//! Domain types and storage-independent use cases shared by Davis clients.

mod local_store;
mod manifest;

use std::fmt;
use std::str::FromStr;

pub use local_store::{IngestedObject, LocalObjectStore, StoreError};
pub use manifest::{
    read_manifest, write_manifest, DatasetManifest, ManifestDataset, ManifestError, ManifestFile,
    ObjectRef,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A storage-independent identifier for immutable content.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectId {
    algorithm: String,
    digest: String,
}

impl ObjectId {
    #[must_use]
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub(crate) fn from_blake3_digest(digest: String) -> Self {
        Self {
            algorithm: "blake3".into(),
            digest,
        }
    }
}

/// Returns the backend-independent key for a content-addressed object.
#[must_use]
pub fn object_key(oid: &ObjectId) -> String {
    format!(
        "objects/{}/{}/{}",
        oid.algorithm(),
        &oid.digest()[..2],
        &oid.digest()[2..]
    )
}

impl fmt::Display for ObjectId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.algorithm, self.digest)
    }
}

impl FromStr for ObjectId {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (algorithm, digest) = value
            .split_once(':')
            .ok_or("object ID must contain an algorithm and digest")?;
        if algorithm.is_empty()
            || digest.len() < 2
            || !algorithm
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("object ID contains invalid characters");
        }
        Ok(Self {
            algorithm: algorithm.to_owned(),
            digest: digest.to_ascii_lowercase(),
        })
    }
}

impl Serialize for ObjectId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ObjectId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Bilingual text used by the current file schemas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalizedText {
    pub ja: String,
    pub en: String,
}

/// Whether metadata for a catalog file can be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchemaStatus {
    Ready,
    Missing,
    Invalid,
}

/// A column declared by a file-level schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: String,
    pub description: Option<LocalizedText>,
}

/// Searchable metadata extracted from a file-level schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSchema {
    pub name: LocalizedText,
    pub description: Option<LocalizedText>,
    pub city: Option<LocalizedText>,
    pub year: Option<i64>,
    pub license: Option<LocalizedText>,
    pub columns: Vec<ColumnSchema>,
}

/// A logical file and the immutable object that contains it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogFile {
    pub id: String,
    pub path: String,
    pub object: ObjectId,
    pub size: u64,
    pub schema_status: SchemaStatus,
    pub schema_path: Option<String>,
    pub schema_error: Option<String>,
    pub schema: Option<FileSchema>,
}

/// A user-facing logical dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub root: String,
    pub files: Vec<CatalogFile>,
}

impl Dataset {
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|file| file.size).sum()
    }

    #[must_use]
    pub fn schema_ready_count(&self) -> usize {
        self.files
            .iter()
            .filter(|file| file.schema_status == SchemaStatus::Ready)
            .count()
    }
}

/// A catalog generated from versioned metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Catalog {
    pub datasets: Vec<Dataset>,
}

impl Catalog {
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.datasets
            .iter()
            .map(|dataset| dataset.files.len())
            .sum()
    }

    #[must_use]
    pub fn schema_ready_count(&self) -> usize {
        self.datasets.iter().map(Dataset::schema_ready_count).sum()
    }

    #[must_use]
    pub fn dataset(&self, id: &str) -> Option<&Dataset> {
        self.datasets.iter().find(|dataset| dataset.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::{DatasetManifest, ManifestDataset, ManifestFile, ObjectId, ObjectRef};

    #[test]
    fn object_id_uses_portable_string_representation() {
        let id: ObjectId = "blake3:aabbcc".parse().unwrap();
        assert_eq!(id.algorithm(), "blake3");
        assert_eq!(id.digest(), "aabbcc");
        assert_eq!(serde_yaml::to_string(&id).unwrap(), "blake3:aabbcc\n");
    }

    #[test]
    fn manifest_file_selection_preserves_manifest_order() {
        let object: ObjectId = "blake3:aabbcc".parse().unwrap();
        let manifest = DatasetManifest {
            version: 1,
            dataset: ManifestDataset {
                id: "sample".into(),
                root: "data/sample".into(),
            },
            files: ["first.csv", "second.csv", "third.csv"]
                .into_iter()
                .map(|id| ManifestFile {
                    id: id.into(),
                    path: id.into(),
                    object: ObjectRef {
                        oid: object.clone(),
                        size: 1,
                    },
                    schema_path: None,
                })
                .collect(),
        };
        let selected = manifest
            .select_files(&["third.csv".into(), "first.csv".into()])
            .unwrap();
        assert_eq!(
            selected
                .files
                .iter()
                .map(|file| file.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first.csv", "third.csv"]
        );
        assert!(manifest.select_files(&["missing.csv".into()]).is_err());
    }
}
