use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::{
    hash_file, read_manifest, write_manifest, DatasetManifest, LocalObjectStore, ManifestError,
    ManifestFile, StoreError,
};

/// Shared on-disk cache used by the CLI, Desktop app, and runtime.
#[derive(Debug, Clone)]
pub struct CatalogCache {
    root: PathBuf,
}

#[derive(Debug, Error)]
pub enum CatalogCacheError {
    #[error("the Davis user data directory could not be determined")]
    DirectoryUnavailable,
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("cached manifest dataset ID mismatch: expected {expected}, found {actual}")]
    DatasetMismatch { expected: String, actual: String },
    #[error("catalog file was not found in the cached manifest: {dataset_id}/{file_id}")]
    FileNotFound { dataset_id: String, file_id: String },
    #[error("catalog revision pins are not supported by this cache yet: {0}")]
    UnsupportedRevision(String),
}

impl CatalogCache {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Opens the platform-wide Davis catalog cache.
    ///
    /// # Errors
    ///
    /// Returns an error when the platform data directory cannot be determined.
    pub fn for_user() -> Result<Self, CatalogCacheError> {
        Ok(Self::new(user_data_directory()?.join("catalog")))
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn object_store(&self) -> LocalObjectStore {
        LocalObjectStore::new(self.root.join("objects"))
    }

    /// Records a remotely obtained manifest so every Davis frontend can resolve
    /// the same logical dataset/file identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest is invalid or cannot be written.
    pub fn store_manifest(&self, manifest: &DatasetManifest) -> Result<PathBuf, CatalogCacheError> {
        let path = self.manifest_path(&manifest.dataset.id);
        write_manifest(&path, manifest)?;
        Ok(path)
    }

    /// Materializes one downloaded object into the shared readable-file area.
    /// Existing valid materializations are reused.
    ///
    /// # Errors
    ///
    /// Returns an error when the file is absent, the object is unavailable, or
    /// the destination cannot be written.
    pub fn materialize_file(
        &self,
        manifest: &DatasetManifest,
        file_id: &str,
    ) -> Result<PathBuf, CatalogCacheError> {
        let selected = manifest.select_files(&[file_id.to_owned()])?;
        let file = selected
            .files
            .first()
            .ok_or_else(|| CatalogCacheError::FileNotFound {
                dataset_id: manifest.dataset.id.clone(),
                file_id: file_id.to_owned(),
            })?;
        let destination = self.materialized_path(&selected, file);
        if destination.is_file() && hash_file(&destination)? == file.object {
            self.store_manifest(manifest)?;
            return Ok(destination);
        }
        self.object_store()
            .materialize(&selected, &self.materialized_root(), true)?;
        self.store_manifest(manifest)?;
        Ok(destination)
    }

    /// Resolves a catalog reference without network access.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest or materialized file is missing or
    /// fails its content-integrity check.
    pub fn resolve_file(
        &self,
        dataset_id: &str,
        file_id: &str,
        revision: Option<&str>,
    ) -> Result<PathBuf, CatalogCacheError> {
        if let Some(revision) = revision {
            return Err(CatalogCacheError::UnsupportedRevision(revision.to_owned()));
        }
        let manifest = read_manifest(&self.manifest_path(dataset_id))?;
        if manifest.dataset.id != dataset_id {
            return Err(CatalogCacheError::DatasetMismatch {
                expected: dataset_id.to_owned(),
                actual: manifest.dataset.id,
            });
        }
        let file = manifest
            .files
            .iter()
            .find(|file| file.id == file_id)
            .ok_or_else(|| CatalogCacheError::FileNotFound {
                dataset_id: dataset_id.to_owned(),
                file_id: file_id.to_owned(),
            })?;
        let path = self.materialized_path(&manifest, file);
        let actual = hash_file(&path)?;
        if actual != file.object {
            return Err(StoreError::Integrity {
                expected: file.object.oid.clone(),
                actual: actual.oid,
            }
            .into());
        }
        Ok(path)
    }

    #[must_use]
    fn manifest_path(&self, dataset_id: &str) -> PathBuf {
        let key = blake3::hash(dataset_id.as_bytes()).to_hex();
        self.root.join("manifests").join(format!("{key}.yaml"))
    }

    #[must_use]
    fn materialized_root(&self) -> PathBuf {
        self.root.join("files")
    }

    #[must_use]
    fn materialized_path(&self, manifest: &DatasetManifest, file: &ManifestFile) -> PathBuf {
        self.materialized_root()
            .join(&manifest.dataset.root)
            .join(&file.path)
    }
}

/// Returns the per-user Davis data directory.
///
/// `DAVIS_DATA_HOME` overrides the platform default for tests and portable
/// installations.
///
/// # Errors
///
/// Returns an error when no platform home/data directory is available.
pub fn user_data_directory() -> Result<PathBuf, CatalogCacheError> {
    if let Some(directory) = std::env::var_os("DAVIS_DATA_HOME") {
        return Ok(PathBuf::from(directory));
    }
    #[cfg(target_os = "windows")]
    if let Some(directory) =
        std::env::var_os("LOCALAPPDATA").or_else(|| std::env::var_os("APPDATA"))
    {
        return Ok(PathBuf::from(directory).join("Davis"));
    }
    #[cfg(target_os = "macos")]
    if let Some(directory) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(directory).join("Library/Application Support/Davis"));
    }
    if let Some(directory) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(directory).join("davis"));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|directory| directory.join(".local/share/davis"))
        .ok_or(CatalogCacheError::DirectoryUnavailable)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::{LocalObjectStore, ManifestDataset};

    #[test]
    fn materialized_catalog_file_can_be_resolved_by_logical_id() {
        let temporary = tempdir().unwrap();
        let cache = CatalogCache::new(temporary.path().join("catalog"));
        let source = temporary.path().join("source.csv");
        std::fs::write(&source, "id,value\n1,2\n").unwrap();
        let store: LocalObjectStore = cache.object_store();
        let ingested = store.ingest_file(&source).unwrap();
        let manifest = DatasetManifest {
            version: 1,
            dataset: ManifestDataset {
                id: "example/dataset".to_owned(),
                root: "example".to_owned(),
            },
            files: vec![ManifestFile {
                id: "table".to_owned(),
                path: "table.csv".to_owned(),
                object: crate::ObjectRef {
                    oid: ingested.oid,
                    size: ingested.size,
                },
                updated_at: None,
                schema_path: None,
            }],
        };

        let path = cache.materialize_file(&manifest, "table").unwrap();
        assert_eq!(
            cache
                .resolve_file("example/dataset", "table", None)
                .unwrap(),
            path
        );
        assert_eq!(std::fs::read_to_string(path).unwrap(), "id,value\n1,2\n");
    }
}
