use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;
use thiserror::Error;

use crate::{object_key, DatasetManifest, ObjectId};

const BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestedObject {
    pub oid: ObjectId,
    pub size: u64,
    pub already_present: bool,
}

#[derive(Debug, Clone)]
pub struct LocalObjectStore {
    root: PathBuf,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("source file was not found: {0}")]
    SourceNotFound(PathBuf),
    #[error("object was not found: {0}")]
    ObjectNotFound(ObjectId),
    #[error("object failed integrity verification: expected {expected}, found {actual}")]
    Integrity {
        expected: ObjectId,
        actual: ObjectId,
    },
    #[error("object size mismatch: expected {expected}, found {actual}")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("destination already exists: {0}")]
    DestinationExists(PathBuf),
    #[error("file is too large to represent: {0}")]
    SizeOverflow(PathBuf),
    #[error("destination path has no parent: {0}")]
    InvalidDestination(PathBuf),
    #[error(transparent)]
    Manifest(#[from] crate::ManifestError),
}

impl LocalObjectStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn object_path(&self, oid: &ObjectId) -> PathBuf {
        self.root.join(object_key(oid))
    }

    /// Copies a source file into content-addressed local storage while hashing it.
    ///
    /// # Errors
    ///
    /// Returns an error when the source cannot be read, the store cannot be
    /// written, or an existing object is corrupt.
    pub fn ingest_file(&self, source: &Path) -> Result<IngestedObject, StoreError> {
        if !source.is_file() {
            return Err(StoreError::SourceNotFound(source.to_path_buf()));
        }
        let temporary_directory = self.root.join(".tmp");
        fs::create_dir_all(&temporary_directory)
            .map_err(|source_error| io_error(&temporary_directory, source_error))?;
        let mut temporary = NamedTempFile::new_in(&temporary_directory)
            .map_err(|source_error| io_error(&temporary_directory, source_error))?;
        let mut input =
            File::open(source).map_err(|source_error| io_error(source, source_error))?;
        let mut hasher = blake3::Hasher::new();
        let mut size = 0_u64;
        let mut buffer = vec![0_u8; BUFFER_SIZE];

        loop {
            let read = input
                .read(&mut buffer)
                .map_err(|source_error| io_error(source, source_error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            temporary
                .write_all(&buffer[..read])
                .map_err(|source_error| io_error(temporary.path(), source_error))?;
            size = add_read_size(size, read, source)?;
        }
        temporary
            .as_file()
            .sync_all()
            .map_err(|source_error| io_error(temporary.path(), source_error))?;

        let oid = ObjectId::from_blake3_digest(hasher.finalize().to_hex().to_string());
        let destination = self.object_path(&oid);
        if destination.exists() {
            self.verify_object(&oid, size)?;
            return Ok(IngestedObject {
                oid,
                size,
                already_present: true,
            });
        }
        let parent = destination
            .parent()
            .ok_or_else(|| StoreError::InvalidDestination(destination.clone()))?;
        fs::create_dir_all(parent).map_err(|source_error| io_error(parent, source_error))?;
        temporary
            .persist(&destination)
            .map_err(|error| io_error(&destination, error.error))?;
        Ok(IngestedObject {
            oid,
            size,
            already_present: false,
        })
    }

    /// Verifies an object against its ID and expected size.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, unreadable, truncated, or corrupt objects.
    pub fn verify_object(&self, oid: &ObjectId, expected_size: u64) -> Result<(), StoreError> {
        let path = self.object_path(oid);
        if !path.is_file() {
            return Err(StoreError::ObjectNotFound(oid.clone()));
        }
        let (actual, size) = hash_file(&path)?;
        if size != expected_size {
            return Err(StoreError::SizeMismatch {
                expected: expected_size,
                actual: size,
            });
        }
        if &actual != oid {
            return Err(StoreError::Integrity {
                expected: oid.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Materializes all manifest files under an output root.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest is invalid, an object fails
    /// verification, or a destination cannot be written.
    pub fn materialize(
        &self,
        manifest: &DatasetManifest,
        output_root: &Path,
        force: bool,
    ) -> Result<(), StoreError> {
        manifest.validate()?;
        for file in &manifest.files {
            self.verify_object(&file.object.oid, file.object.size)?;
            let source = self.object_path(&file.object.oid);
            let destination = output_root.join(&manifest.dataset.root).join(&file.path);
            if destination.exists() && !force {
                return Err(StoreError::DestinationExists(destination));
            }
            let parent = destination
                .parent()
                .ok_or_else(|| StoreError::InvalidDestination(destination.clone()))?;
            fs::create_dir_all(parent).map_err(|source_error| io_error(parent, source_error))?;
            let mut temporary = NamedTempFile::new_in(parent)
                .map_err(|source_error| io_error(parent, source_error))?;
            let mut input =
                File::open(&source).map_err(|source_error| io_error(&source, source_error))?;
            std::io::copy(&mut input, &mut temporary)
                .map_err(|source_error| io_error(&destination, source_error))?;
            temporary
                .as_file()
                .sync_all()
                .map_err(|source_error| io_error(&destination, source_error))?;
            if destination.exists() {
                fs::remove_file(&destination)
                    .map_err(|source_error| io_error(&destination, source_error))?;
            }
            temporary
                .persist(&destination)
                .map_err(|error| io_error(&destination, error.error))?;
        }
        Ok(())
    }
}

fn hash_file(path: &Path) -> Result<(ObjectId, u64), StoreError> {
    let mut input = File::open(path).map_err(|source_error| io_error(path, source_error))?;
    let mut hasher = blake3::Hasher::new();
    let mut size = 0_u64;
    let mut buffer = vec![0_u8; BUFFER_SIZE];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|source_error| io_error(path, source_error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size = add_read_size(size, read, path)?;
    }
    Ok((
        ObjectId::from_blake3_digest(hasher.finalize().to_hex().to_string()),
        size,
    ))
}

fn io_error(path: &Path, source: std::io::Error) -> StoreError {
    StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn add_read_size(current: u64, read: usize, path: &Path) -> Result<u64, StoreError> {
    let read = u64::try_from(read).map_err(|_| StoreError::SizeOverflow(path.to_path_buf()))?;
    current
        .checked_add(read)
        .ok_or_else(|| StoreError::SizeOverflow(path.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use crate::{DatasetManifest, ManifestDataset, ManifestFile, ObjectRef};

    use super::LocalObjectStore;

    #[test]
    fn ingests_and_materializes_without_changing_content() {
        let temporary = tempdir().unwrap();
        let source = temporary.path().join("source.csv");
        fs::write(&source, b"id,value\n1,example\n").unwrap();
        let store = LocalObjectStore::new(temporary.path().join("store"));
        let object = store.ingest_file(&source).unwrap();
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

        let output = temporary.path().join("output");
        store.materialize(&manifest, &output, false).unwrap();
        assert_eq!(
            fs::read(output.join("data/sample/tiny/source.csv")).unwrap(),
            fs::read(source).unwrap()
        );
    }
}
