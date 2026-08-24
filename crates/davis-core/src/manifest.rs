use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;

use crate::ObjectId;

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub version: u32,
    pub dataset: ManifestDataset,
    pub files: Vec<ManifestFile>,
}

impl DatasetManifest {
    /// Validates invariants required before storage or materialization.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported versions, unsafe paths, duplicate file
    /// IDs, or duplicate paths.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.version != MANIFEST_VERSION {
            return Err(ManifestError::UnsupportedVersion(self.version));
        }
        validate_relative_path(&self.dataset.root)?;

        let mut ids = HashSet::new();
        let mut paths = HashSet::new();
        for file in &self.files {
            validate_relative_path(&file.path)?;
            if let Some(updated_at) = &file.updated_at {
                let canonical = updated_at
                    .parse::<jiff::civil::Date>()
                    .map(|date| date.to_string())
                    .map_err(|_| ManifestError::InvalidUpdatedAt {
                        file_id: file.id.clone(),
                        value: updated_at.clone(),
                    })?;
                if canonical != *updated_at {
                    return Err(ManifestError::InvalidUpdatedAt {
                        file_id: file.id.clone(),
                        value: updated_at.clone(),
                    });
                }
            }
            if file.id.is_empty() || !ids.insert(&file.id) {
                return Err(ManifestError::DuplicateFileId(file.id.clone()));
            }
            if !paths.insert(&file.path) {
                return Err(ManifestError::DuplicateFilePath(file.path.clone()));
            }
        }
        Ok(())
    }

    /// Creates a manifest containing only the requested logical file IDs.
    ///
    /// # Errors
    ///
    /// Returns an error when a requested ID is absent or the selection is empty.
    pub fn select_files(&self, file_ids: &[String]) -> Result<Self, ManifestError> {
        if file_ids.is_empty() {
            return Err(ManifestError::EmptyFileSelection);
        }
        let requested: HashSet<&str> = file_ids.iter().map(String::as_str).collect();
        for file_id in &requested {
            if !self.files.iter().any(|file| file.id == *file_id) {
                return Err(ManifestError::FileNotFound((*file_id).to_owned()));
            }
        }
        let selected = Self {
            version: self.version,
            dataset: self.dataset.clone(),
            files: self
                .files
                .iter()
                .filter(|file| requested.contains(file.id.as_str()))
                .cloned()
                .collect(),
        };
        selected.validate()?;
        Ok(selected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestDataset {
    pub id: String,
    pub root: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    pub id: String,
    pub path: String,
    pub object: ObjectRef,
    /// Date when this object ID was first recorded by an updating Davis operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectRef {
    pub oid: ObjectId,
    pub size: u64,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("failed to read manifest {path}: {source}")]
    Read {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write manifest {path}: {source}")]
    Write {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("invalid manifest YAML in {path}: {source}")]
    InvalidYaml {
        path: std::path::PathBuf,
        source: serde_yaml::Error,
    },
    #[error("unsupported manifest version: {0}")]
    UnsupportedVersion(u32),
    #[error("manifest path must be a safe relative path: {0}")]
    UnsafePath(String),
    #[error("duplicate or empty file ID: {0}")]
    DuplicateFileId(String),
    #[error("duplicate file path: {0}")]
    DuplicateFilePath(String),
    #[error("invalid updated_at date for file {file_id}: {value} (expected YYYY-MM-DD)")]
    InvalidUpdatedAt { file_id: String, value: String },
    #[error("file selection must not be empty")]
    EmptyFileSelection,
    #[error("file was not found in the manifest: {0}")]
    FileNotFound(String),
}

/// Reads and validates a `DatasetManifest` YAML file.
///
/// # Errors
///
/// Returns an error when the file cannot be read, parsed, or validated.
pub fn read_manifest(path: &Path) -> Result<DatasetManifest, ManifestError> {
    let contents = fs::read_to_string(path).map_err(|source| ManifestError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let manifest: DatasetManifest =
        serde_yaml::from_str(&contents).map_err(|source| ManifestError::InvalidYaml {
            path: path.to_path_buf(),
            source,
        })?;
    manifest.validate()?;
    Ok(manifest)
}

/// Atomically writes a validated `DatasetManifest` as YAML.
///
/// # Errors
///
/// Returns an error when validation, serialization, or writing fails.
pub fn write_manifest(path: &Path, manifest: &DatasetManifest) -> Result<(), ManifestError> {
    manifest.validate()?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| ManifestError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    let contents =
        serde_yaml::to_string(manifest).map_err(|source| ManifestError::InvalidYaml {
            path: path.to_path_buf(),
            source,
        })?;
    let mut temporary = NamedTempFile::new_in(parent).map_err(|source| ManifestError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    std::io::Write::write_all(&mut temporary, contents.as_bytes()).map_err(|source| {
        ManifestError::Write {
            path: path.to_path_buf(),
            source,
        }
    })?;
    temporary
        .persist(path)
        .map_err(|error| ManifestError::Write {
            path: path.to_path_buf(),
            source: error.error,
        })?;
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), ManifestError> {
    let path_value = Path::new(path);
    if path_value.as_os_str().is_empty()
        || !path_value
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err(ManifestError::UnsafePath(path.to_owned()));
    }
    Ok(())
}
