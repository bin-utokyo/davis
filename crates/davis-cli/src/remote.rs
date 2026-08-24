use std::collections::HashMap;
use std::fmt::Write as _;

use davis_catalog::{IndexedDataset, IndexedFile};
use davis_core::{
    Catalog, CatalogFile, Dataset, DatasetManifest, FileSchema, LocalObjectStore, ManifestDataset,
    ManifestFile, ObjectRef,
};
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const DEFAULT_UPLOAD_PART_SIZE: usize = 32 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RemoteError {
    #[error("service URL must use http or https")]
    InvalidScheme,
    #[error("invalid service URL: {0}")]
    InvalidUrl(String),
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Davis service returned HTTP {status}: {message}")]
    Api { status: StatusCode, message: String },
    #[error("dataset was not found: {0}")]
    DatasetNotFound(String),
    #[error("catalog file path is outside dataset root: {0}")]
    InvalidCatalogPath(String),
    #[error("catalog response contains an inconsistent dataset ID: {0}")]
    InconsistentDataset(String),
    #[error("download grant was not returned for file: {0}")]
    MissingGrant(String),
    #[error("upload plan did not contain object: {0}")]
    MissingUploadPlan(String),
    #[error("operator upload response is invalid: {0}")]
    InvalidUploadResponse(String),
    #[error("download size mismatch for {file_id}: expected {expected}, found {actual}")]
    DownloadSize {
        file_id: String,
        expected: u64,
        actual: u64,
    },
    #[error("downloaded object mismatch for {file_id}: expected {expected}, found {actual}")]
    DownloadObject {
        file_id: String,
        expected: String,
        actual: String,
    },
    #[error("failed to create a temporary download: {0}")]
    Temporary(#[from] std::io::Error),
    #[error(transparent)]
    Manifest(#[from] davis_core::ManifestError),
    #[error(transparent)]
    Store(#[from] davis_core::StoreError),
}

#[derive(Debug, Clone)]
pub struct DavisService {
    client: Client,
    base_url: String,
    token: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginSession {
    pub token: String,
    pub expires_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadReport {
    pub downloaded: usize,
    pub cached: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatorUploadReport {
    pub missing: usize,
    pub existing: usize,
    pub uploaded: usize,
    pub missing_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ExchangeRequest<'a> {
    invite_code: &'a str,
    client: &'static str,
}

#[derive(Debug, Serialize)]
struct OperatorExchangeRequest<'a> {
    operator_code: &'a str,
    client: &'static str,
}

#[derive(Debug, Deserialize)]
struct ExchangeResponse {
    token: String,
    expires_at: String,
}

#[derive(Debug, Deserialize)]
struct OperatorStatusResponse {
    expires_at: String,
}

#[derive(Debug, Serialize)]
struct GrantRequest {
    file_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GrantResponse {
    grants: Vec<DownloadGrant>,
}

#[derive(Debug, Deserialize)]
struct DownloadGrant {
    file_id: String,
    size: u64,
    url: String,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(default)]
    code: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    details: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ApiErrorEnvelope {
    error: ApiError,
}

#[derive(Debug, Serialize)]
struct OperatorObjectsRequest<'a> {
    objects: &'a [ObjectRef],
}

#[derive(Debug, Deserialize)]
struct OperatorPlanResponse {
    objects: Vec<OperatorPlanObject>,
}

#[derive(Debug, Deserialize)]
struct OperatorPlanObject {
    oid: String,
    size: u64,
    status: String,
}

#[derive(Debug, Deserialize)]
struct OperatorUploadCreateResponse {
    already_present: bool,
    #[serde(default)]
    upload_id: String,
    #[serde(default = "default_upload_part_size")]
    part_size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct UploadedPart {
    part_number: u32,
    etag: String,
    size: u64,
}

#[derive(Debug, Serialize)]
struct CompleteUploadRequest<'a> {
    oid: &'a str,
    size: u64,
    upload_id: &'a str,
    parts: &'a [UploadedPart],
}

#[derive(Debug, Serialize)]
struct AbortUploadRequest<'a> {
    oid: &'a str,
    upload_id: &'a str,
}

#[derive(Debug, Serialize)]
struct PublishCatalogRequest<'a> {
    revision: &'a str,
    documents: &'a HashMap<String, String>,
}

fn default_upload_part_size() -> usize {
    DEFAULT_UPLOAD_PART_SIZE
}

async fn read_upload_chunk(
    input: &mut tokio::fs::File,
    buffer: &mut [u8],
    remaining: u64,
) -> Result<usize, RemoteError> {
    let buffer_size = u64::try_from(buffer.len())
        .map_err(|_| RemoteError::InvalidUploadResponse("multipart chunk is too large".into()))?;
    let read = usize::try_from(remaining.min(buffer_size))
        .map_err(|_| RemoteError::InvalidUploadResponse("multipart chunk is too large".into()))?;
    // Tokio may return a short read (commonly around 2 MiB) even when the
    // supplied buffer is larger. R2 rejects such non-final multipart parts,
    // so fill each declared part completely before uploading it.
    input.read_exact(&mut buffer[..read]).await?;
    Ok(read)
}

impl DavisService {
    pub fn new(service_url: &str, token: Option<String>) -> Result<Self, RemoteError> {
        let mut url = reqwest::Url::parse(service_url)
            .map_err(|error| RemoteError::InvalidUrl(error.to_string()))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(RemoteError::InvalidScheme);
        }
        url.set_query(None);
        url.set_fragment(None);
        let base_url = url.as_str().trim_end_matches('/').to_owned();
        Ok(Self {
            client: Client::builder()
                .user_agent(concat!("davis/", env!("CARGO_PKG_VERSION")))
                .build()?,
            base_url,
            token,
        })
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub async fn exchange_invite_code(
        &self,
        invite_code: &str,
    ) -> Result<LoginSession, RemoteError> {
        let response = self
            .client
            .post(self.endpoint("api/v1/auth/exchange"))
            .json(&ExchangeRequest {
                invite_code,
                client: "cli",
            })
            .send()
            .await?;
        let response: ExchangeResponse = decode(response).await?;
        Ok(LoginSession {
            token: response.token,
            expires_at: response.expires_at,
        })
    }

    pub async fn exchange_operator_code(
        &self,
        operator_code: &str,
    ) -> Result<LoginSession, RemoteError> {
        let response = self
            .client
            .post(self.endpoint("api/v1/operator/auth/exchange"))
            .json(&OperatorExchangeRequest {
                operator_code,
                client: "cli",
            })
            .send()
            .await?;
        let response: ExchangeResponse = decode(response).await?;
        Ok(LoginSession {
            token: response.token,
            expires_at: response.expires_at,
        })
    }

    pub async fn operator_session_status(&self) -> Result<LoginSession, RemoteError> {
        let token = self.operator_token()?;
        let response = self
            .client
            .get(self.endpoint("api/v1/operator/auth/session"))
            .bearer_auth(token)
            .send()
            .await?;
        let response: OperatorStatusResponse = decode(response).await?;
        Ok(LoginSession {
            token: token.to_owned(),
            expires_at: response.expires_at,
        })
    }

    pub async fn upload_operator_objects<F>(
        &self,
        store: &LocalObjectStore,
        objects: &[ObjectRef],
        dry_run: bool,
        mut on_progress: F,
    ) -> Result<OperatorUploadReport, RemoteError>
    where
        F: FnMut(u64, u64, usize, usize),
    {
        let token = self.operator_token()?;
        let response = self
            .client
            .post(self.endpoint("api/v1/operator/uploads/plan"))
            .bearer_auth(token)
            .json(&OperatorObjectsRequest { objects })
            .send()
            .await?;
        let plan: OperatorPlanResponse = decode(response).await?;
        let planned = plan
            .objects
            .into_iter()
            .map(|object| (object.oid.clone(), object))
            .collect::<HashMap<_, _>>();
        let mut missing = Vec::new();
        let mut existing = 0_usize;
        let mut missing_bytes = 0_u64;
        for object in objects {
            let oid = object.oid.to_string();
            let item = planned
                .get(&oid)
                .ok_or_else(|| RemoteError::MissingUploadPlan(oid.clone()))?;
            if item.size != object.size {
                return Err(RemoteError::InvalidUploadResponse(format!(
                    "size mismatch for {oid}"
                )));
            }
            match item.status.as_str() {
                "existing" => existing += 1,
                "missing" => {
                    verify_operator_upload_source(store, object, dry_run)?;
                    missing_bytes = missing_bytes.checked_add(object.size).ok_or_else(|| {
                        RemoteError::InvalidUploadResponse("byte counter overflow".into())
                    })?;
                    missing.push(object);
                }
                status => {
                    return Err(RemoteError::InvalidUploadResponse(format!(
                        "unexpected plan status {status} for {oid}"
                    )));
                }
            }
        }
        let total_objects = missing.len();
        on_progress(0, missing_bytes, 0, total_objects);
        if dry_run {
            return Ok(OperatorUploadReport {
                missing: missing.len(),
                existing,
                uploaded: 0,
                missing_bytes,
            });
        }
        let mut uploaded = 0_usize;
        let mut completed_bytes = 0_u64;
        for object in missing.iter().copied() {
            self.upload_operator_object(
                store,
                object,
                &mut completed_bytes,
                missing_bytes,
                uploaded,
                total_objects,
                &mut on_progress,
            )
            .await?;
            uploaded += 1;
            on_progress(completed_bytes, missing_bytes, uploaded, total_objects);
        }
        Ok(OperatorUploadReport {
            missing: missing.len(),
            existing,
            uploaded,
            missing_bytes,
        })
    }

    pub async fn publish_operator_catalog(
        &self,
        revision: &str,
        documents: &HashMap<String, String>,
    ) -> Result<(), RemoteError> {
        let token = self.operator_token()?;
        let response = self
            .client
            .post(self.endpoint("api/v1/operator/catalog/publish"))
            .bearer_auth(token)
            .json(&PublishCatalogRequest {
                revision,
                documents,
            })
            .send()
            .await?;
        ensure_success(response).await?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn upload_operator_object<F>(
        &self,
        store: &LocalObjectStore,
        object: &ObjectRef,
        completed_bytes: &mut u64,
        total_bytes: u64,
        completed_objects: usize,
        total_objects: usize,
        on_progress: &mut F,
    ) -> Result<(), RemoteError>
    where
        F: FnMut(u64, u64, usize, usize),
    {
        let token = self.operator_token()?;
        let oid = object.oid.to_string();
        let response = self
            .client
            .post(self.endpoint("api/v1/operator/uploads/create"))
            .bearer_auth(token)
            .json(object)
            .send()
            .await?;
        let created: OperatorUploadCreateResponse = decode(response).await?;
        if created.already_present {
            *completed_bytes = completed_bytes.saturating_add(object.size);
            on_progress(
                *completed_bytes,
                total_bytes,
                completed_objects + 1,
                total_objects,
            );
            return Ok(());
        }
        if created.upload_id.is_empty() || created.part_size == 0 {
            return Err(RemoteError::InvalidUploadResponse(format!(
                "missing multipart settings for {oid}"
            )));
        }
        let upload_result = self
            .upload_operator_parts(
                store,
                object,
                &created,
                completed_bytes,
                total_bytes,
                completed_objects,
                total_objects,
                on_progress,
            )
            .await;
        if upload_result.is_err() {
            let _ = self
                .client
                .post(self.endpoint("api/v1/operator/uploads/abort"))
                .bearer_auth(token)
                .json(&AbortUploadRequest {
                    oid: &oid,
                    upload_id: &created.upload_id,
                })
                .send()
                .await;
        }
        let parts = upload_result?;
        let response = self
            .client
            .post(self.endpoint("api/v1/operator/uploads/complete"))
            .bearer_auth(token)
            .json(&CompleteUploadRequest {
                oid: &oid,
                size: object.size,
                upload_id: &created.upload_id,
                parts: &parts,
            })
            .send()
            .await?;
        ensure_success(response).await?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn upload_operator_parts<F>(
        &self,
        store: &LocalObjectStore,
        object: &ObjectRef,
        created: &OperatorUploadCreateResponse,
        completed_bytes: &mut u64,
        total_bytes: u64,
        completed_objects: usize,
        total_objects: usize,
        on_progress: &mut F,
    ) -> Result<Vec<UploadedPart>, RemoteError>
    where
        F: FnMut(u64, u64, usize, usize),
    {
        let token = self.operator_token()?;
        let oid = object.oid.to_string();
        let mut input = tokio::fs::File::open(store.object_path(&object.oid)).await?;
        let mut buffer = vec![0_u8; created.part_size];
        let mut parts = Vec::new();
        let mut remaining = object.size;
        while remaining > 0 {
            let read = read_upload_chunk(&mut input, &mut buffer, remaining).await?;
            let part_number = u32::try_from(parts.len() + 1).map_err(|_| {
                RemoteError::InvalidUploadResponse("too many multipart parts".into())
            })?;
            let mut url = reqwest::Url::parse(&self.endpoint("api/v1/operator/uploads/part"))
                .map_err(|error| RemoteError::InvalidUrl(error.to_string()))?;
            url.query_pairs_mut()
                .append_pair("oid", &oid)
                .append_pair("upload_id", &created.upload_id)
                .append_pair("part_number", &part_number.to_string());
            let response = self
                .client
                .put(url)
                .bearer_auth(token)
                .header(reqwest::header::CONTENT_LENGTH, read)
                .body(buffer[..read].to_vec())
                .send()
                .await?;
            let part: UploadedPart = decode(response).await?;
            if part.part_number != part_number {
                return Err(RemoteError::InvalidUploadResponse(format!(
                    "unexpected multipart part number for {oid}"
                )));
            }
            if part.size
                != u64::try_from(read).map_err(|_| {
                    RemoteError::InvalidUploadResponse("multipart chunk is too large".into())
                })?
            {
                return Err(RemoteError::InvalidUploadResponse(format!(
                    "unexpected multipart part size for {oid}"
                )));
            }
            parts.push(part);
            let read = u64::try_from(read).map_err(|_| {
                RemoteError::InvalidUploadResponse("multipart chunk is too large".into())
            })?;
            remaining -= read;
            *completed_bytes = completed_bytes.saturating_add(read);
            on_progress(
                *completed_bytes,
                total_bytes,
                completed_objects,
                total_objects,
            );
        }
        Ok(parts)
    }

    fn operator_token(&self) -> Result<&str, RemoteError> {
        self.token.as_deref().ok_or_else(|| RemoteError::Api {
            status: StatusCode::UNAUTHORIZED,
            message: "operator login is required; run `davis operator login <URL>`".into(),
        })
    }

    pub async fn catalog(&self) -> Result<Catalog, RemoteError> {
        let datasets: Vec<IndexedDataset> = self.get_json("catalog/datasets.json").await?;
        let files: Vec<IndexedFile> = self.get_json("catalog/files.json").await?;
        build_catalog(datasets, &files)
    }

    pub async fn manifest(&self, dataset_id: &str) -> Result<DatasetManifest, RemoteError> {
        let datasets: Vec<IndexedDataset> = self.get_json("catalog/datasets.json").await?;
        let dataset = datasets
            .into_iter()
            .find(|dataset| dataset.id == dataset_id)
            .ok_or_else(|| RemoteError::DatasetNotFound(dataset_id.to_owned()))?;
        let files: Vec<IndexedFile> = self.get_json("catalog/files.json").await?;
        build_manifest(
            &dataset,
            files
                .into_iter()
                .filter(|file| file.dataset_id == dataset_id),
        )
    }

    pub async fn indexed_files(&self, dataset_id: &str) -> Result<Vec<IndexedFile>, RemoteError> {
        let files: Vec<IndexedFile> = self.get_json("catalog/files.json").await?;
        Ok(files
            .into_iter()
            .filter(|file| file.dataset_id == dataset_id)
            .collect())
    }

    pub async fn download_manifest<F>(
        &self,
        store: &LocalObjectStore,
        manifest: &DatasetManifest,
        mut on_progress: F,
    ) -> Result<DownloadReport, RemoteError>
    where
        F: FnMut(u64, u64, usize, usize),
    {
        let total_bytes = manifest.files.iter().try_fold(0_u64, |sum, file| {
            sum.checked_add(file.object.size)
                .ok_or_else(|| std::io::Error::other("download byte counter overflow"))
        })?;
        let total_objects = manifest.files.len();
        let mut completed_bytes = 0_u64;
        let mut cached = 0_usize;
        let mut missing = Vec::new();
        for file in &manifest.files {
            if store
                .verify_object(&file.object.oid, file.object.size)
                .is_ok()
            {
                cached += 1;
                completed_bytes += file.object.size;
                on_progress(completed_bytes, total_bytes, cached, total_objects);
            } else {
                missing.push(file);
            }
        }
        if missing.is_empty() {
            return Ok(DownloadReport {
                downloaded: 0,
                cached,
                downloaded_bytes: 0,
                total_bytes,
            });
        }
        let mut grants = self.download_grants(manifest, &missing).await?;
        let mut downloaded = 0_usize;
        let mut downloaded_bytes = 0_u64;
        for file in missing {
            let global_id = format!("{}:{}", manifest.dataset.id, file.id);
            let grant = grants
                .remove(&global_id)
                .ok_or_else(|| RemoteError::MissingGrant(global_id.clone()))?;
            if grant.size != file.object.size {
                return Err(RemoteError::DownloadSize {
                    file_id: global_id,
                    expected: file.object.size,
                    actual: grant.size,
                });
            }
            let file_bytes = self
                .download_one(
                    store,
                    file,
                    &global_id,
                    &grant,
                    completed_bytes,
                    total_bytes,
                    cached + downloaded,
                    total_objects,
                    &mut on_progress,
                )
                .await?;
            completed_bytes += file_bytes;
            downloaded_bytes += file_bytes;
            downloaded += 1;
            on_progress(
                completed_bytes,
                total_bytes,
                cached + downloaded,
                total_objects,
            );
        }
        Ok(DownloadReport {
            downloaded,
            cached,
            downloaded_bytes,
            total_bytes,
        })
    }

    async fn download_grants(
        &self,
        manifest: &DatasetManifest,
        missing: &[&ManifestFile],
    ) -> Result<HashMap<String, DownloadGrant>, RemoteError> {
        let token = self.token.as_deref().ok_or_else(|| RemoteError::Api {
            status: StatusCode::UNAUTHORIZED,
            message: "login is required; run `davis login <URL>`".into(),
        })?;
        let file_ids = missing
            .iter()
            .map(|file| format!("{}:{}", manifest.dataset.id, file.id))
            .collect::<Vec<_>>();
        let mut grants = HashMap::new();
        for chunk in file_ids.chunks(256) {
            let request = GrantRequest {
                file_ids: chunk.to_vec(),
            };
            let response = self
                .client
                .post(self.endpoint("api/v1/download-grants"))
                .bearer_auth(token)
                .json(&request)
                .send()
                .await?;
            let response: GrantResponse = decode(response).await?;
            grants.extend(
                response
                    .grants
                    .into_iter()
                    .map(|grant| (grant.file_id.clone(), grant)),
            );
        }
        Ok(grants)
    }

    #[allow(clippy::too_many_arguments)]
    async fn download_one<F>(
        &self,
        store: &LocalObjectStore,
        file: &ManifestFile,
        global_id: &str,
        grant: &DownloadGrant,
        completed_bytes: u64,
        total_bytes: u64,
        completed_objects: usize,
        total_objects: usize,
        on_progress: &mut F,
    ) -> Result<u64, RemoteError>
    where
        F: FnMut(u64, u64, usize, usize),
    {
        let response = self.client.get(&grant.url).send().await?;
        let response = ensure_success(response).await?;
        let temporary = NamedTempFile::new()?;
        let output = temporary.reopen()?;
        let mut output = tokio::fs::File::from_std(output);
        let mut stream = response.bytes_stream();
        let mut file_bytes = 0_u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            output.write_all(&chunk).await?;
            let chunk_size = u64::try_from(chunk.len())
                .map_err(|_| std::io::Error::other("download chunk is too large"))?;
            file_bytes = file_bytes
                .checked_add(chunk_size)
                .ok_or_else(|| std::io::Error::other("download byte counter overflow"))?;
            on_progress(
                completed_bytes + file_bytes,
                total_bytes,
                completed_objects,
                total_objects,
            );
        }
        output.flush().await?;
        output.sync_all().await?;
        drop(output);
        if file_bytes != file.object.size {
            return Err(RemoteError::DownloadSize {
                file_id: global_id.to_owned(),
                expected: file.object.size,
                actual: file_bytes,
            });
        }
        let ingested = store.ingest_file(temporary.path())?;
        if ingested.oid != file.object.oid {
            return Err(RemoteError::DownloadObject {
                file_id: global_id.to_owned(),
                expected: file.object.oid.to_string(),
                actual: ingested.oid.to_string(),
            });
        }
        Ok(file_bytes)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, RemoteError> {
        let response = self.client.get(self.endpoint(path)).send().await?;
        decode(response).await
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }
}

fn verify_operator_upload_source(
    store: &LocalObjectStore,
    object: &ObjectRef,
    dry_run: bool,
) -> Result<(), RemoteError> {
    if dry_run {
        return Ok(());
    }
    store.verify_object(&object.oid, object.size)?;
    Ok(())
}

fn build_catalog(
    datasets: Vec<IndexedDataset>,
    files: &[IndexedFile],
) -> Result<Catalog, RemoteError> {
    let mut result = Vec::with_capacity(datasets.len());
    for indexed_dataset in datasets {
        let dataset_files = files
            .iter()
            .filter(|file| file.dataset_id == indexed_dataset.id)
            .map(indexed_to_catalog_file)
            .collect();
        result.push(Dataset {
            id: indexed_dataset.id,
            root: indexed_dataset.root,
            files: dataset_files,
        });
    }
    let catalog = Catalog { datasets: result };
    for file in files {
        if catalog.dataset(&file.dataset_id).is_none() {
            return Err(RemoteError::InconsistentDataset(file.dataset_id.clone()));
        }
    }
    Ok(catalog)
}

fn indexed_to_catalog_file(file: &IndexedFile) -> CatalogFile {
    let schema = file.name.as_ref().map(|name| FileSchema {
        name: name.clone(),
        description: file.description.clone(),
        city: file.city.clone(),
        year: file.year,
        license: file.license.clone(),
        columns: file.columns.clone(),
    });
    CatalogFile {
        id: file.file_id.clone(),
        path: file.path.clone(),
        object: file.object.oid.clone(),
        size: file.size,
        schema_status: file.schema_status,
        schema_path: file.schema_path.clone(),
        schema_error: file.schema_error.clone(),
        schema,
    }
}

fn build_manifest(
    dataset: &IndexedDataset,
    files: impl IntoIterator<Item = IndexedFile>,
) -> Result<DatasetManifest, RemoteError> {
    let prefix = format!("{}/", dataset.root.trim_end_matches('/'));
    let files = files
        .into_iter()
        .map(|file| {
            let path = file
                .path
                .strip_prefix(&prefix)
                .ok_or_else(|| RemoteError::InvalidCatalogPath(file.path.clone()))?;
            Ok(ManifestFile {
                id: file.file_id,
                path: path.to_owned(),
                object: file.object,
                updated_at: file.updated_at,
                schema_path: file.schema_path,
            })
        })
        .collect::<Result<Vec<_>, RemoteError>>()?;
    let manifest = DatasetManifest {
        version: 1,
        dataset: ManifestDataset {
            id: dataset.id.clone(),
            root: dataset.root.clone(),
        },
        files,
    };
    manifest.validate()?;
    Ok(manifest)
}

async fn decode<T: DeserializeOwned>(response: reqwest::Response) -> Result<T, RemoteError> {
    let response = ensure_success(response).await?;
    Ok(response.json().await?)
}

async fn ensure_success(response: reqwest::Response) -> Result<reqwest::Response, RemoteError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let request_id = response
        .headers()
        .get("x-davis-request-id")
        .or_else(|| response.headers().get("cf-ray"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = response.text().await.unwrap_or_default();
    let message = format_api_error(&body, request_id.as_deref());
    Err(RemoteError::Api { status, message })
}

fn format_api_error(body: &str, request_id: Option<&str>) -> String {
    let parsed = serde_json::from_str::<ApiErrorEnvelope>(body)
        .map(|envelope| envelope.error)
        .or_else(|_| serde_json::from_str::<ApiError>(body));
    let mut message = match parsed {
        Ok(error) => {
            let mut message = if error.message.trim().is_empty() {
                "The service returned an error without a message".to_owned()
            } else {
                error.message.trim().to_owned()
            };
            if !error.code.trim().is_empty() {
                write!(message, " [code: {}]", error.code.trim())
                    .expect("writing to a String cannot fail");
            }
            if let Some(details) = error.details.filter(|value| !value.is_null()) {
                write!(
                    message,
                    "; details: {}",
                    truncate_message(&details.to_string())
                )
                .expect("writing to a String cannot fail");
            }
            message
        }
        Err(_) if body.trim().is_empty() => "The service returned no error details".to_owned(),
        Err(_) => truncate_message(&body.split_whitespace().collect::<Vec<_>>().join(" ")),
    };
    if let Some(request_id) = request_id.filter(|value| !value.trim().is_empty()) {
        write!(message, " [request ID: {}]", request_id.trim())
            .expect("writing to a String cannot fail");
    }
    message
}

fn truncate_message(message: &str) -> String {
    const LIMIT: usize = 800;
    let mut characters = message.chars();
    let truncated = characters.by_ref().take(LIMIT).collect::<String>();
    if characters.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_manifest, format_api_error, read_upload_chunk, verify_operator_upload_source,
    };
    use davis_catalog::{IndexedDataset, IndexedFile};
    use davis_core::{hash_file, LocalObjectStore};

    #[test]
    fn formats_structured_and_empty_service_errors_actionably() {
        assert_eq!(
            format_api_error(
                r#"{"error":{"code":"invalid_request","message":"objects must not be empty","details":{"field":"objects"}}}"#,
                Some("request-123"),
            ),
            "objects must not be empty [code: invalid_request]; details: {\"field\":\"objects\"} [request ID: request-123]"
        );
        assert_eq!(
            format_api_error("", Some("request-456")),
            "The service returned no error details [request ID: request-456]"
        );
    }

    #[test]
    fn remote_catalog_recreates_relative_manifest_paths() {
        let dataset: IndexedDataset = serde_json::from_str(
            r#"{"id":"network/sample","root":"data/network/sample","file_count":1,"schema_ready_count":1,"total_size":12}"#,
        )
        .unwrap();
        let file: IndexedFile = serde_json::from_str(
            r#"{"id":"network/sample:link.csv","dataset_id":"network/sample","file_id":"link.csv","path":"data/network/sample/link.csv","size":12,"object":{"oid":"blake3:aabbcc","size":12},"format":"csv","schema_status":"ready","schema_path":null,"schema_error":null,"name":null,"description":null,"city":null,"year":null,"license":null,"columns":[],"raw_schema":null}"#,
        )
        .unwrap();
        let manifest = build_manifest(&dataset, [file]).unwrap();
        assert_eq!(manifest.dataset.root, "data/network/sample");
        assert_eq!(manifest.files[0].path, "link.csv");
    }

    #[tokio::test]
    async fn multipart_reader_fills_every_non_final_chunk() {
        const PART_SIZE: usize = 5 * 1024 * 1024;
        let source = tempfile::NamedTempFile::new().expect("temporary source should be created");
        std::fs::write(source.path(), vec![7_u8; PART_SIZE + 17])
            .expect("temporary source should be written");
        let mut input = tokio::fs::File::open(source.path())
            .await
            .expect("temporary source should open");
        let mut buffer = vec![0_u8; PART_SIZE];

        let total_size = u64::try_from(PART_SIZE + 17).expect("test size should fit in u64");
        let first = read_upload_chunk(&mut input, &mut buffer, total_size)
            .await
            .expect("first chunk should be read");
        let final_part = read_upload_chunk(&mut input, &mut buffer, 17)
            .await
            .expect("final chunk should be read");

        assert_eq!(first, PART_SIZE);
        assert_eq!(final_part, 17);
    }

    #[test]
    fn dry_run_does_not_require_a_new_object_in_the_local_cache() {
        let temporary = tempfile::tempdir().expect("temporary directory should be created");
        let source = temporary.path().join("changed.csv");
        std::fs::write(&source, b"id,value\n1,changed\n")
            .expect("temporary source should be written");
        let object = hash_file(&source).expect("source should be hashed");
        let empty_store = LocalObjectStore::new(temporary.path().join("empty-cache"));

        verify_operator_upload_source(&empty_store, &object, true)
            .expect("dry run should use the source hash without requiring a cache object");
        assert!(verify_operator_upload_source(&empty_store, &object, false).is_err());
    }
}
