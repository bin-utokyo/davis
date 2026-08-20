//! `OpenDAL` adapters for local filesystems and S3-compatible object storage.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use davis_core::{object_key, DatasetManifest, LocalObjectStore, ObjectId};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadPlan {
    pub missing: usize,
    pub existing: usize,
    pub missing_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadReport {
    pub downloaded: usize,
    pub cached: usize,
    pub bytes: u64,
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
        manifest.validate().map_err(davis_core::StoreError::from)?;
        let mut plan = UploadPlan {
            missing: 0,
            existing: 0,
            missing_bytes: 0,
        };
        for file in &manifest.files {
            local.verify_object(&file.object.oid, file.object.size)?;
            let key = object_key(&file.object.oid);
            match self.operator.stat(&key).await {
                Ok(metadata) => {
                    if metadata.content_length() != file.object.size {
                        return Err(StorageError::SizeMismatch {
                            oid: file.object.oid.clone(),
                            expected: file.object.size,
                            actual: metadata.content_length(),
                        });
                    }
                    plan.existing += 1;
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    plan.missing += 1;
                    plan.missing_bytes = plan
                        .missing_bytes
                        .checked_add(file.object.size)
                        .ok_or(StorageError::SizeOverflow)?;
                }
                Err(error) => return Err(error.into()),
            }
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
        manifest.validate().map_err(davis_core::StoreError::from)?;
        let mut report = UploadReport {
            uploaded: 0,
            skipped: 0,
            bytes: 0,
        };
        for file in &manifest.files {
            match self
                .upload_object(local, &file.object.oid, file.object.size)
                .await?
            {
                UploadOutcome::Uploaded => report.uploaded += 1,
                UploadOutcome::AlreadyPresent => report.skipped += 1,
            }
            report.bytes = report
                .bytes
                .checked_add(file.object.size)
                .ok_or(StorageError::SizeOverflow)?;
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
        manifest.validate().map_err(davis_core::StoreError::from)?;
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
                    temporary
                        .write_all(&bytes)
                        .map_err(|source| StorageError::DownloadWrite {
                            path: temporary.path().to_path_buf(),
                            source,
                        })?;
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
        }
        Ok(report)
    }
}

fn default_region() -> String {
    "auto".into()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use davis_core::{DatasetManifest, LocalObjectStore, ManifestDataset, ManifestFile, ObjectRef};
    use tempfile::tempdir;

    use super::{ObjectStorage, UploadOutcome};

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

        let empty_cache = LocalObjectStore::new(temporary.path().join("empty-cache"));
        let download = remote
            .download_manifest(&empty_cache, &manifest)
            .await
            .unwrap();
        assert_eq!(download.downloaded, 1);
        empty_cache
            .verify_object(&manifest.files[0].object.oid, object.size)
            .unwrap();
    }
}
