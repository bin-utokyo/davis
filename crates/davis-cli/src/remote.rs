use std::collections::HashMap;

use davis_catalog::{IndexedDataset, IndexedFile};
use davis_core::{
    Catalog, CatalogFile, Dataset, DatasetManifest, FileSchema, LocalObjectStore, ManifestDataset,
    ManifestFile,
};
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::io::AsyncWriteExt;

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

#[derive(Debug, Serialize)]
struct ExchangeRequest<'a> {
    invite_code: &'a str,
    client: &'static str,
}

#[derive(Debug, Deserialize)]
struct ExchangeResponse {
    token: String,
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
    message: String,
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
        let request = GrantRequest {
            file_ids: missing
                .iter()
                .map(|file| format!("{}:{}", manifest.dataset.id, file.id))
                .collect(),
        };
        let response = self
            .client
            .post(self.endpoint("api/v1/download-grants"))
            .bearer_auth(token)
            .json(&request)
            .send()
            .await?;
        let response: GrantResponse = decode(response).await?;
        Ok(response
            .grants
            .into_iter()
            .map(|grant| (grant.file_id.clone(), grant))
            .collect())
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
    let body = response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<ApiError>(&body)
        .map(|error| error.message)
        .unwrap_or(body);
    Err(RemoteError::Api { status, message })
}

#[cfg(test)]
mod tests {
    use super::build_manifest;
    use davis_catalog::{IndexedDataset, IndexedFile};

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
}
