//! `OpenDAL` adapters for local filesystems and S3-compatible object storage.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use davis_core::{object_key, DatasetManifest, LocalObjectStore, ObjectId, ObjectRef};
use futures::TryStreamExt;
use opendal::services::{Fs, S3};
use opendal::{ErrorKind, Operator};
use serde::Deserialize;
use tempfile::NamedTempFile;
use thiserror::Error;
use tokio::io::AsyncReadExt;

const TRANSFER_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// Configuration shared by AWS S3, Cloudflare R2, and compatible services.
///
/// Credentials are deliberately not serializable or printable. Callers should
/// resolve them from a credential store or environment at runtime.
pub struct S3StorageConfig {
    pub bucket: String,
    pub endpoint: String,
    pub region: String,
    pub root: String,
    pub access_key_id: String,
    pub secret_access_key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfiguration {
    pub version: u32,
    pub remote: BTreeMap<String, RemoteConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RemoteConfig {
    Fs {
        root: PathBuf,
    },
    S3 {
        bucket: String,
        endpoint: String,
        #[serde(default = "default_region")]
        region: String,
        #[serde(default)]
        root: String,
    },
}

pub struct S3Credentials {
    pub access_key_id: String,
    pub secret_access_key: String,
}

#[derive(Debug, Clone)]
pub struct ObjectStorage {
    operator: Operator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadReport {
    pub uploaded: usize,
    pub skipped: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadPlan {
    pub missing: usize,
    pub existing: usize,
    pub missing_bytes: u64,
    missing_objects: Vec<ObjectRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteCoverage {
    pub existing: usize,
    pub missing: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadReport {
    pub downloaded: usize,
    pub cached: usize,
    pub bytes: u64,
}

/// Aggregate progress for one storage operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferProgress {
    pub completed_bytes: u64,
    pub total_bytes: u64,
    pub completed_objects: usize,
    pub total_objects: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadOutcome {
    Uploaded,
    AlreadyPresent,
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error(transparent)]
    Backend(#[from] opendal::Error),
    #[error("filesystem path is not valid UTF-8: {0}")]
    NonUtf8Path(PathBuf),
    #[error("failed to read local object {path}: {source}")]
    LocalRead {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("remote object size mismatch for {oid}: expected {expected}, found {actual}")]
    SizeMismatch {
        oid: ObjectId,
        expected: u64,
        actual: u64,
    },
    #[error("conflicting sizes for object {oid}: {first} and {second}")]
    ConflictingObjectSize {
        oid: ObjectId,
        first: u64,
        second: u64,
    },
    #[error(transparent)]
    LocalStore(#[from] davis_core::StoreError),
    #[error("failed to read storage configuration {path}: {source}")]
    ConfigRead {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid storage configuration {path}: {source}")]
    ConfigParse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("unsupported storage configuration version: {0}")]
    ConfigVersion(u32),
    #[error("S3 credentials are required for this remote")]
    MissingS3Credentials,
    #[error("size counter overflow")]
    SizeOverflow,
    #[error("downloaded object ID mismatch: expected {expected}, found {actual}")]
    DownloadIntegrity {
        expected: ObjectId,
        actual: ObjectId,
    },
    #[error("failed to write downloaded object {path}: {source}")]
    DownloadWrite {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("remote key must be a safe relative path: {0}")]
    InvalidRemoteKey(String),
}

/// Reads a versioned `.davis/config.toml` file.
///
/// # Errors
///
/// Returns an error when the file cannot be read or parsed, or when its version
/// is unsupported.
pub fn read_storage_configuration(path: &Path) -> Result<StorageConfiguration, StorageError> {
    let contents = fs::read_to_string(path).map_err(|source| StorageError::ConfigRead {
        path: path.to_path_buf(),
        source,
    })?;
    let config: StorageConfiguration =
        toml::from_str(&contents).map_err(|source| StorageError::ConfigParse {
            path: path.to_path_buf(),
            source,
        })?;
    if config.version != 1 {
        return Err(StorageError::ConfigVersion(config.version));
    }
    Ok(config)
}

impl ObjectStorage {
    /// Creates an `OpenDAL` filesystem backend for integration tests and local use.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is not UTF-8 or `OpenDAL` cannot initialize.
    pub fn filesystem(root: &Path) -> Result<Self, StorageError> {
        let root = root
            .to_str()
            .ok_or_else(|| StorageError::NonUtf8Path(root.to_path_buf()))?;
        let builder = Fs::default().root(root);
        Ok(Self {
            operator: Operator::new(builder)?,
        })
    }

    /// Creates an S3-compatible backend, including Cloudflare R2.
    ///
    /// # Errors
    ///
    /// Returns an error when `OpenDAL` rejects the backend configuration.
    pub fn s3(config: &S3StorageConfig) -> Result<Self, StorageError> {
        // Default features are disabled to keep Davis binaries small, so the
        // HTTP transport must be installed explicitly before any S3 request.
        opendal::install_default();
        let builder = S3::default()
            .bucket(&config.bucket)
            .endpoint(&config.endpoint)
            .region(&config.region)
            .root(&config.root)
            .access_key_id(&config.access_key_id)
            .secret_access_key(&config.secret_access_key);
        Ok(Self {
            operator: Operator::new(builder)?,
        })
    }

    /// Builds a backend from public configuration and separately supplied secrets.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid backend configuration or missing S3 secrets.
    pub fn from_config(
        config: &RemoteConfig,
        credentials: Option<&S3Credentials>,
    ) -> Result<Self, StorageError> {
        match config {
            RemoteConfig::Fs { root } => Self::filesystem(root),
            RemoteConfig::S3 {
                bucket,
                endpoint,
                region,
                root,
            } => {
                let credentials = credentials.ok_or(StorageError::MissingS3Credentials)?;
                Self::s3(&S3StorageConfig {
                    bucket: bucket.clone(),
                    endpoint: endpoint.clone(),
                    region: region.clone(),
                    root: root.clone(),
                    access_key_id: credentials.access_key_id.clone(),
                    secret_access_key: credentials.secret_access_key.clone(),
                })
            }
        }
    }

    /// Writes a small derived document to a stable remote key.
    ///
    /// This is intended for catalog metadata. Content-addressed dataset objects
    /// continue to use the verified streaming upload methods below.
    ///
    /// # Errors
    ///
    /// Returns an error when the key is unsafe or the backend rejects the write.
    pub async fn write_document(&self, key: &str, contents: Vec<u8>) -> Result<(), StorageError> {
        validate_remote_key(key)?;
        self.operator.write(key, contents).await?;
        Ok(())
    }

    /// Checks whether every content-addressed object referenced by manifests
    /// exists remotely, without reading the local object cache.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid manifests, conflicting object sizes, or a
    /// backend failure. Existing objects with an unexpected size also fail.
    pub async fn remote_coverage(
        &self,
        manifests: &[DatasetManifest],
    ) -> Result<RemoteCoverage, StorageError> {
        let objects = unique_objects(manifests)?;
        let mut coverage = RemoteCoverage {
            existing: 0,
            missing: 0,
        };
        for object in objects {
            let key = object_key(&object.oid);
            match self.operator.stat(&key).await {
                Ok(metadata) => {
                    if metadata.content_length() != object.size {
                        return Err(StorageError::SizeMismatch {
                            oid: object.oid,
                            expected: object.size,
                            actual: metadata.content_length(),
                        });
                    }
                    coverage.existing += 1;
                }
                Err(error) if error.kind() == ErrorKind::NotFound => coverage.missing += 1,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(coverage)
    }

    /// Calculates which manifest objects are missing without uploading anything.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation, local verification, or remote
    /// metadata access fails.
    pub async fn plan_upload(
        &self,
        local: &LocalObjectStore,
        manifest: &DatasetManifest,
    ) -> Result<UploadPlan, StorageError> {
        self.plan_upload_manifests(local, std::slice::from_ref(manifest))
            .await
    }

    /// Calculates missing objects across multiple manifests without double counting.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation, local verification, or remote
    /// metadata access fails.
    pub async fn plan_upload_manifests(
        &self,
        local: &LocalObjectStore,
        manifests: &[DatasetManifest],
    ) -> Result<UploadPlan, StorageError> {
        self.plan_upload_manifests_with_progress(local, manifests, |_| {})
            .await
    }

    /// Calculates missing objects and reports local verification progress.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation, local verification, or remote
    /// metadata access fails.
    pub async fn plan_upload_manifests_with_progress<F>(
        &self,
        local: &LocalObjectStore,
        manifests: &[DatasetManifest],
        on_progress: F,
    ) -> Result<UploadPlan, StorageError>
    where
        F: FnMut(TransferProgress),
    {
        self.plan_upload_manifests_internal(Some(local), manifests, on_progress)
            .await
    }

    /// Calculates missing objects using remote metadata only.
    ///
    /// This read-only variant is intended for dry runs that hash changed source
    /// files without writing them into the local object cache.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation or remote metadata access fails.
    pub async fn plan_remote_upload_manifests_with_progress<F>(
        &self,
        manifests: &[DatasetManifest],
        on_progress: F,
    ) -> Result<UploadPlan, StorageError>
    where
        F: FnMut(TransferProgress),
    {
        self.plan_upload_manifests_internal(None, manifests, on_progress)
            .await
    }

    async fn plan_upload_manifests_internal<F>(
        &self,
        local: Option<&LocalObjectStore>,
        manifests: &[DatasetManifest],
        mut on_progress: F,
    ) -> Result<UploadPlan, StorageError>
    where
        F: FnMut(TransferProgress),
    {
        let objects = unique_objects(manifests)?;
        let total_bytes = total_object_bytes(&objects)?;
        let mut progress = TransferProgress {
            completed_bytes: 0,
            total_bytes,
            completed_objects: 0,
            total_objects: objects.len(),
        };
        on_progress(progress);
        let mut plan = UploadPlan {
            missing: 0,
            existing: 0,
            missing_bytes: 0,
            missing_objects: Vec::new(),
        };
        for object in objects {
            let key = object_key(&object.oid);
            match self.operator.stat(&key).await {
                Ok(metadata) => {
                    if metadata.content_length() != object.size {
                        return Err(StorageError::SizeMismatch {
                            oid: object.oid,
                            expected: object.size,
                            actual: metadata.content_length(),
                        });
                    }
                    plan.existing += 1;
                    progress.completed_bytes = progress.completed_bytes.saturating_add(object.size);
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    if let Some(local) = local {
                        local.verify_object_with_progress(&object.oid, object.size, |bytes| {
                            progress.completed_bytes =
                                progress.completed_bytes.saturating_add(bytes);
                            on_progress(progress);
                        })?;
                    } else {
                        progress.completed_bytes =
                            progress.completed_bytes.saturating_add(object.size);
                    }
                    plan.missing += 1;
                    plan.missing_bytes = plan
                        .missing_bytes
                        .checked_add(object.size)
                        .ok_or(StorageError::SizeOverflow)?;
                    plan.missing_objects.push(object.clone());
                }
                Err(error) => return Err(error.into()),
            }
            progress.completed_objects += 1;
            on_progress(progress);
        }
        Ok(plan)
    }

    /// Streams one verified local object to remote storage without buffering it.
    ///
    /// # Errors
    ///
    /// Returns an error when the local object is corrupt, reading fails, remote
    /// storage fails, or an existing remote object has a different size.
    pub async fn upload_object(
        &self,
        local: &LocalObjectStore,
        oid: &ObjectId,
        size: u64,
    ) -> Result<UploadOutcome, StorageError> {
        self.upload_object_with_progress(local, oid, size, |_| {})
            .await
    }

    /// Streams one verified object and reports bytes handled by the push.
    ///
    /// Existing objects report their full size immediately. Newly uploaded
    /// objects report each chunk after it has been accepted by the backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the local object is corrupt, reading fails, remote
    /// storage fails, or an existing remote object has a different size.
    pub async fn upload_object_with_progress<F>(
        &self,
        local: &LocalObjectStore,
        oid: &ObjectId,
        size: u64,
        mut on_progress: F,
    ) -> Result<UploadOutcome, StorageError>
    where
        F: FnMut(u64),
    {
        local.verify_object(oid, size)?;
        let key = object_key(oid);
        match self.operator.stat(&key).await {
            Ok(metadata) => {
                if metadata.content_length() != size {
                    return Err(StorageError::SizeMismatch {
                        oid: oid.clone(),
                        expected: size,
                        actual: metadata.content_length(),
                    });
                }
                on_progress(size);
                return Ok(UploadOutcome::AlreadyPresent);
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        let source = local.object_path(oid);
        let mut input = tokio::fs::File::open(&source)
            .await
            .map_err(|source_error| StorageError::LocalRead {
                path: source.clone(),
                source: source_error,
            })?;
        let mut writer = self
            .operator
            .writer_with(&key)
            .if_not_exists(true)
            .chunk(TRANSFER_CHUNK_SIZE)
            .await?;
        let mut buffer = vec![0_u8; TRANSFER_CHUNK_SIZE];
        loop {
            let read =
                input
                    .read(&mut buffer)
                    .await
                    .map_err(|source_error| StorageError::LocalRead {
                        path: source.clone(),
                        source: source_error,
                    })?;
            if read == 0 {
                break;
            }
            writer.write(buffer[..read].to_vec()).await?;
            on_progress(u64::try_from(read).map_err(|_| StorageError::SizeOverflow)?);
        }
        writer.close().await?;

        let metadata = self.operator.stat(&key).await?;
        if metadata.content_length() != size {
            return Err(StorageError::SizeMismatch {
                oid: oid.clone(),
                expected: size,
                actual: metadata.content_length(),
            });
        }
        Ok(UploadOutcome::Uploaded)
    }

    /// Uploads every object referenced by a manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation or any object upload fails.
    pub async fn upload_manifest(
        &self,
        local: &LocalObjectStore,
        manifest: &DatasetManifest,
    ) -> Result<UploadReport, StorageError> {
        self.upload_manifests(local, std::slice::from_ref(manifest))
            .await
    }

    /// Uploads every distinct object referenced by multiple manifests.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation or any object upload fails.
    pub async fn upload_manifests(
        &self,
        local: &LocalObjectStore,
        manifests: &[DatasetManifest],
    ) -> Result<UploadReport, StorageError> {
        self.upload_manifests_with_progress(local, manifests, |_| {})
            .await
    }

    /// Uploads distinct objects and reports aggregate handled bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when manifest validation or any object upload fails.
    pub async fn upload_manifests_with_progress<F>(
        &self,
        local: &LocalObjectStore,
        manifests: &[DatasetManifest],
        mut on_progress: F,
    ) -> Result<UploadReport, StorageError>
    where
        F: FnMut(TransferProgress),
    {
        let objects = unique_objects(manifests)?;
        let mut progress = TransferProgress {
            completed_bytes: 0,
            total_bytes: total_object_bytes(&objects)?,
            completed_objects: 0,
            total_objects: objects.len(),
        };
        on_progress(progress);
        let mut report = UploadReport {
            uploaded: 0,
            skipped: 0,
            bytes: 0,
        };
        for object in objects {
            let outcome = self
                .upload_object_with_progress(local, &object.oid, object.size, |bytes| {
                    progress.completed_bytes = progress.completed_bytes.saturating_add(bytes);
                    on_progress(progress);
                })
                .await?;
            match outcome {
                UploadOutcome::Uploaded => report.uploaded += 1,
                UploadOutcome::AlreadyPresent => report.skipped += 1,
            }
            report.bytes = report
                .bytes
                .checked_add(object.size)
                .ok_or(StorageError::SizeOverflow)?;
            progress.completed_objects += 1;
            on_progress(progress);
        }
        Ok(report)
    }

    /// Uploads only the missing objects captured by a previous upload plan.
    ///
    /// # Errors
    ///
    /// Returns an error when a planned object is no longer valid locally or its
    /// upload fails. Objects created remotely after planning are safely skipped.
    pub async fn upload_plan(
        &self,
        local: &LocalObjectStore,
        plan: &UploadPlan,
    ) -> Result<UploadReport, StorageError> {
        self.upload_plan_with_progress(local, plan, |_| {}).await
    }

    /// Uploads a prepared plan and reports progress over its missing objects only.
    ///
    /// # Errors
    ///
    /// Returns an error when a planned object is no longer valid locally or its
    /// upload fails. Objects created remotely after planning are safely skipped.
    pub async fn upload_plan_with_progress<F>(
        &self,
        local: &LocalObjectStore,
        plan: &UploadPlan,
        mut on_progress: F,
    ) -> Result<UploadReport, StorageError>
    where
        F: FnMut(TransferProgress),
    {
        let mut progress = TransferProgress {
            completed_bytes: 0,
            total_bytes: plan.missing_bytes,
            completed_objects: 0,
            total_objects: plan.missing_objects.len(),
        };
        on_progress(progress);
        let mut report = UploadReport {
            uploaded: 0,
            skipped: 0,
            bytes: 0,
        };
        for object in &plan.missing_objects {
            let outcome = self
                .upload_object_with_progress(local, &object.oid, object.size, |bytes| {
                    progress.completed_bytes = progress.completed_bytes.saturating_add(bytes);
                    on_progress(progress);
                })
                .await?;
            match outcome {
                UploadOutcome::Uploaded => report.uploaded += 1,
                UploadOutcome::AlreadyPresent => report.skipped += 1,
            }
            report.bytes = report
                .bytes
                .checked_add(object.size)
                .ok_or(StorageError::SizeOverflow)?;
            progress.completed_objects += 1;
            on_progress(progress);
        }
        Ok(report)
    }

    /// Downloads missing manifest objects into the local content cache.
    ///
    /// # Errors
    ///
    /// Returns an error when a remote object is missing or has the wrong size,
    /// streaming fails, or the downloaded BLAKE3 ID does not match the manifest.
    pub async fn download_manifest(
        &self,
        local: &LocalObjectStore,
        manifest: &DatasetManifest,
    ) -> Result<DownloadReport, StorageError> {
        self.download_manifest_with_progress(local, manifest, |_| {})
            .await
    }

    /// Downloads missing objects and reports aggregate downloaded or cached bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when a remote object is missing or has the wrong size,
    /// streaming fails, or the downloaded BLAKE3 ID does not match the manifest.
    pub async fn download_manifest_with_progress<F>(
        &self,
        local: &LocalObjectStore,
        manifest: &DatasetManifest,
        mut on_progress: F,
    ) -> Result<DownloadReport, StorageError>
    where
        F: FnMut(TransferProgress),
    {
        manifest.validate().map_err(davis_core::StoreError::from)?;
        let total_bytes = manifest.files.iter().try_fold(0_u64, |total, file| {
            total
                .checked_add(file.object.size)
                .ok_or(StorageError::SizeOverflow)
        })?;
        let mut progress = TransferProgress {
            completed_bytes: 0,
            total_bytes,
            completed_objects: 0,
            total_objects: manifest.files.len(),
        };
        on_progress(progress);
        let mut report = DownloadReport {
            downloaded: 0,
            cached: 0,
            bytes: 0,
        };
        for file in &manifest.files {
            if local
                .verify_object(&file.object.oid, file.object.size)
                .is_ok()
            {
                report.cached += 1;
                progress.completed_bytes =
                    progress.completed_bytes.saturating_add(file.object.size);
                progress.completed_objects += 1;
                on_progress(progress);
                continue;
            }
            let key = object_key(&file.object.oid);
            let metadata = self.operator.stat(&key).await?;
            if metadata.content_length() != file.object.size {
                return Err(StorageError::SizeMismatch {
                    oid: file.object.oid.clone(),
                    expected: file.object.size,
                    actual: metadata.content_length(),
                });
            }
            let mut temporary =
                NamedTempFile::new().map_err(|source| StorageError::DownloadWrite {
                    path: std::env::temp_dir(),
                    source,
                })?;
            let mut stream = self.operator.reader(&key).await?.into_stream(..).await?;
            while let Some(buffer) = stream.try_next().await? {
                for bytes in buffer {
                    let transferred =
                        u64::try_from(bytes.len()).map_err(|_| StorageError::SizeOverflow)?;
                    temporary
                        .write_all(&bytes)
                        .map_err(|source| StorageError::DownloadWrite {
                            path: temporary.path().to_path_buf(),
                            source,
                        })?;
                    progress.completed_bytes = progress.completed_bytes.saturating_add(transferred);
                    on_progress(progress);
                }
            }
            temporary
                .as_file()
                .sync_all()
                .map_err(|source| StorageError::DownloadWrite {
                    path: temporary.path().to_path_buf(),
                    source,
                })?;
            let ingested = local.ingest_file(temporary.path())?;
            if ingested.oid != file.object.oid {
                return Err(StorageError::DownloadIntegrity {
                    expected: file.object.oid.clone(),
                    actual: ingested.oid,
                });
            }
            report.downloaded += 1;
            report.bytes = report
                .bytes
                .checked_add(ingested.size)
                .ok_or(StorageError::SizeOverflow)?;
            progress.completed_objects += 1;
            on_progress(progress);
        }
        Ok(report)
    }
}

fn default_region() -> String {
    "auto".into()
}

fn validate_remote_key(key: &str) -> Result<(), StorageError> {
    let path = Path::new(key);
    if key.is_empty()
        || key.contains('\\')
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::CurDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(StorageError::InvalidRemoteKey(key.to_owned()));
    }
    Ok(())
}

fn unique_objects(manifests: &[DatasetManifest]) -> Result<Vec<ObjectRef>, StorageError> {
    let mut sizes = HashMap::<ObjectId, u64>::new();
    let mut objects = Vec::new();
    for manifest in manifests {
        manifest.validate().map_err(davis_core::StoreError::from)?;
        for file in &manifest.files {
            match sizes.get(&file.object.oid) {
                Some(size) if *size != file.object.size => {
                    return Err(StorageError::ConflictingObjectSize {
                        oid: file.object.oid.clone(),
                        first: *size,
                        second: file.object.size,
                    });
                }
                Some(_) => {}
                None => {
                    sizes.insert(file.object.oid.clone(), file.object.size);
                    objects.push(file.object.clone());
                }
            }
        }
    }
    Ok(objects)
}

fn total_object_bytes(objects: &[ObjectRef]) -> Result<u64, StorageError> {
    objects.iter().try_fold(0_u64, |total, object| {
        total
            .checked_add(object.size)
            .ok_or(StorageError::SizeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use davis_core::{
        hash_file, DatasetManifest, LocalObjectStore, ManifestDataset, ManifestFile, ObjectRef,
    };
    use tempfile::tempdir;

    use super::{ObjectStorage, StorageError, UploadOutcome};

    #[tokio::test]
    async fn writes_catalog_documents_only_to_safe_relative_keys() {
        let temporary = tempdir().unwrap();
        let remote_root = temporary.path().join("remote");
        let remote = ObjectStorage::filesystem(&remote_root).unwrap();

        remote
            .write_document("catalog/revisions/abc/files.json", b"[]\n".to_vec())
            .await
            .unwrap();
        assert_eq!(
            fs::read(remote_root.join("catalog/revisions/abc/files.json")).unwrap(),
            b"[]\n"
        );
        assert!(matches!(
            remote.write_document("../outside", Vec::new()).await,
            Err(StorageError::InvalidRemoteKey(_))
        ));
    }

    #[tokio::test]
    async fn uploads_to_filesystem_without_overwriting_existing_object() {
        let temporary = tempdir().unwrap();
        let source = temporary.path().join("source.csv");
        fs::write(&source, b"id,value\n1,example\n").unwrap();
        let local = LocalObjectStore::new(temporary.path().join("local"));
        let object = local.ingest_file(&source).unwrap();
        let remote = ObjectStorage::filesystem(&temporary.path().join("remote")).unwrap();

        let initial_plan = remote
            .plan_upload(
                &local,
                &DatasetManifest {
                    version: 1,
                    dataset: ManifestDataset {
                        id: "sample/tiny".into(),
                        root: "data/sample/tiny".into(),
                    },
                    files: vec![ManifestFile {
                        id: "source.csv".into(),
                        path: "source.csv".into(),
                        object: ObjectRef {
                            oid: object.oid.clone(),
                            size: object.size,
                        },
                        schema_path: None,
                    }],
                },
            )
            .await
            .unwrap();
        assert_eq!(initial_plan.missing, 1);

        assert_eq!(
            remote
                .upload_object(&local, &object.oid, object.size)
                .await
                .unwrap(),
            UploadOutcome::Uploaded
        );
        assert_eq!(
            remote
                .upload_object(&local, &object.oid, object.size)
                .await
                .unwrap(),
            UploadOutcome::AlreadyPresent
        );

        let manifest = DatasetManifest {
            version: 1,
            dataset: ManifestDataset {
                id: "sample/tiny".into(),
                root: "data/sample/tiny".into(),
            },
            files: vec![ManifestFile {
                id: "source.csv".into(),
                path: "source.csv".into(),
                object: ObjectRef {
                    oid: object.oid,
                    size: object.size,
                },
                schema_path: None,
            }],
        };
        let report = remote.upload_manifest(&local, &manifest).await.unwrap();
        assert_eq!(report.uploaded, 0);
        assert_eq!(report.skipped, 1);
        let final_plan = remote.plan_upload(&local, &manifest).await.unwrap();
        assert_eq!(final_plan.missing, 0);
        assert_eq!(final_plan.existing, 1);
        let final_coverage = remote
            .remote_coverage(std::slice::from_ref(&manifest))
            .await
            .unwrap();
        assert_eq!(final_coverage.missing, 0);
        assert_eq!(final_coverage.existing, 1);

        fs::remove_file(local.object_path(&manifest.files[0].object.oid)).unwrap();
        let remote_only_plan = remote.plan_upload(&local, &manifest).await.unwrap();
        assert_eq!(remote_only_plan.missing, 0);
        assert_eq!(remote_only_plan.existing, 1);

        let empty_cache = LocalObjectStore::new(temporary.path().join("empty-cache"));
        let mut download_progress = Vec::new();
        let download = remote
            .download_manifest_with_progress(&empty_cache, &manifest, |progress| {
                download_progress.push(progress);
            })
            .await
            .unwrap();
        assert_eq!(download.downloaded, 1);
        let final_progress = download_progress.last().unwrap();
        assert_eq!(final_progress.completed_bytes, object.size);
        assert_eq!(final_progress.completed_objects, 1);
        empty_cache
            .verify_object(&manifest.files[0].object.oid, object.size)
            .unwrap();
    }

    #[tokio::test]
    async fn remote_only_plan_supports_a_dry_run_without_cache_objects() {
        let temporary = tempdir().unwrap();
        let source = temporary.path().join("changed.csv");
        fs::write(&source, b"id,value\n1,changed\n").unwrap();
        let object = hash_file(&source).unwrap();
        let manifest = DatasetManifest {
            version: 1,
            dataset: ManifestDataset {
                id: "sample/changed".into(),
                root: "data/sample/changed".into(),
            },
            files: vec![ManifestFile {
                id: "changed.csv".into(),
                path: "changed.csv".into(),
                object: object.clone(),
                schema_path: None,
            }],
        };
        let remote = ObjectStorage::filesystem(&temporary.path().join("remote")).unwrap();
        let mut progress = Vec::new();

        let plan = remote
            .plan_remote_upload_manifests_with_progress(&[manifest], |state| {
                progress.push(state);
            })
            .await
            .unwrap();

        assert_eq!(plan.missing, 1);
        assert_eq!(plan.existing, 0);
        assert_eq!(plan.missing_bytes, object.size);
        assert_eq!(progress.last().unwrap().completed_bytes, object.size);
        assert_eq!(progress.last().unwrap().completed_objects, 1);
    }

    #[tokio::test]
    async fn deduplicates_objects_shared_by_multiple_manifests() {
        let temporary = tempdir().unwrap();
        let source = temporary.path().join("shared.csv");
        fs::write(&source, b"id,value\n1,shared\n").unwrap();
        let local = LocalObjectStore::new(temporary.path().join("local"));
        let object = local.ingest_file(&source).unwrap();
        let remote = ObjectStorage::filesystem(&temporary.path().join("remote")).unwrap();

        let manifests: Vec<DatasetManifest> = ["sample/first", "sample/second"]
            .into_iter()
            .map(|dataset_id| DatasetManifest {
                version: 1,
                dataset: ManifestDataset {
                    id: dataset_id.into(),
                    root: format!("data/{dataset_id}"),
                },
                files: vec![ManifestFile {
                    id: "shared.csv".into(),
                    path: "shared.csv".into(),
                    object: ObjectRef {
                        oid: object.oid.clone(),
                        size: object.size,
                    },
                    schema_path: None,
                }],
            })
            .collect();

        let mut plan_progress = Vec::new();
        let initial = remote
            .plan_upload_manifests_with_progress(&local, &manifests, |progress| {
                plan_progress.push(progress);
            })
            .await
            .unwrap();
        assert_eq!(initial.missing, 1);
        assert_eq!(initial.missing_bytes, object.size);
        assert_eq!(plan_progress.last().unwrap().completed_bytes, object.size);
        assert_eq!(plan_progress.last().unwrap().completed_objects, 1);

        let mut upload_progress = Vec::new();
        let uploaded = remote
            .upload_plan_with_progress(&local, &initial, |progress| {
                upload_progress.push(progress);
            })
            .await
            .unwrap();
        assert_eq!(uploaded.uploaded, 1);
        assert_eq!(uploaded.skipped, 0);
        assert_eq!(uploaded.bytes, object.size);
        assert_eq!(upload_progress.last().unwrap().completed_bytes, object.size);
        assert_eq!(upload_progress.last().unwrap().completed_objects, 1);

        let final_plan = remote
            .plan_upload_manifests(&local, &manifests)
            .await
            .unwrap();
        assert_eq!(final_plan.missing, 0);
        assert_eq!(final_plan.existing, 1);
        let no_upload = remote.upload_plan(&local, &final_plan).await.unwrap();
        assert_eq!(no_upload.uploaded, 0);
        assert_eq!(no_upload.skipped, 0);
        assert_eq!(no_upload.bytes, 0);
    }
}
