//! Catalog generation and compatibility adapters.

mod index;

pub use index::{
    build_catalog_index, write_catalog_index, CatalogFacets, CatalogIndex, CatalogSummary,
    IndexedColumn, IndexedDataset, IndexedFile,
};

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use davis_core::{
    Catalog, CatalogFile, ColumnSchema, Dataset, DatasetManifest, FileSchema, LocalObjectStore,
    LocalizedText, ManifestDataset, ManifestFile, ObjectId, ObjectRef, SchemaStatus,
};
use md5::{Digest, Md5};
use serde::Deserialize;
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("data directory was not found: {0}")]
    DataDirectoryNotFound(PathBuf),
    #[error("failed to walk the data directory: {0}")]
    Walk(#[from] walkdir::Error),
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize catalog index: {0}")]
    SerializeIndex(#[from] serde_json::Error),
    #[error(transparent)]
    Manifest(#[from] davis_core::ManifestError),
    #[error("invalid DVC metadata in {path}: {source}")]
    InvalidDvc {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    #[error("DVC metadata has no output: {0}")]
    MissingDvcOutput(PathBuf),
    #[error("DVC output path is unsafe in {dvc_path}: {output_path}")]
    UnsafeOutputPath {
        dvc_path: PathBuf,
        output_path: String,
    },
    #[error("catalog path is outside the repository: {0}")]
    PathOutsideRepository(PathBuf),
    #[error("cannot infer a dataset from path: {0}")]
    CannotInferDataset(PathBuf),
    #[error("DVC metadata path has no parent: {0}")]
    InvalidDvcPath(PathBuf),
    #[error("invalid legacy object ID in {path}: {value}")]
    InvalidObjectId { path: PathBuf, value: String },
    #[error("legacy file size mismatch for {path}: expected {expected}, found {actual}")]
    LegacySizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("legacy MD5 mismatch for {path}: expected {expected}, found {actual}")]
    LegacyDigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(transparent)]
    Store(#[from] davis_core::StoreError),
}

#[derive(Debug)]
pub struct IngestReport {
    pub manifest: DatasetManifest,
    pub added_objects: usize,
    pub existing_objects: usize,
    pub reused_files: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditReport {
    pub files: usize,
    pub bytes: u64,
}

#[derive(Debug, Deserialize)]
struct LegacyDvcFile {
    outs: Vec<LegacyDvcOutput>,
}

#[derive(Debug, Deserialize)]
struct LegacyDvcOutput {
    md5: String,
    size: u64,
    path: String,
}

#[derive(Debug, Deserialize)]
struct LegacyFileSchema {
    name: LocalizedText,
    description: Option<LocalizedText>,
    city: Option<LocalizedText>,
    year: Option<i64>,
    #[serde(rename = "license_")]
    license: Option<LocalizedText>,
    #[serde(default)]
    columns: Vec<LegacyColumnSchema>,
}

#[derive(Debug, Deserialize)]
struct LegacyColumnSchema {
    name: String,
    #[serde(rename = "type_")]
    data_type: LegacyDataType,
    description: Option<LocalizedText>,
}

#[derive(Debug, Deserialize)]
struct LegacyDataType {
    name: String,
}

impl From<LegacyFileSchema> for FileSchema {
    fn from(schema: LegacyFileSchema) -> Self {
        Self {
            name: schema.name,
            description: schema.description,
            city: schema.city,
            year: schema.year,
            license: schema.license,
            columns: schema
                .columns
                .into_iter()
                .map(|column| ColumnSchema {
                    name: column.name,
                    data_type: column.data_type.name,
                    description: column.description,
                })
                .collect(),
        }
    }
}

/// Scans the current repository layout without requiring DVC or remote credentials.
///
/// This is a compatibility adapter for P0. New Davis manifests will become the
/// primary input after the R2 bootstrap is available.
///
/// # Errors
///
/// Returns an error when the data directory cannot be read, DVC metadata is
/// malformed, or a path would escape the repository.
pub fn scan_legacy_repository(repository_root: &Path) -> Result<Catalog, CatalogError> {
    let data_root = repository_root.join("data");
    if !data_root.is_dir() {
        return Err(CatalogError::DataDirectoryNotFound(data_root));
    }

    let mut datasets: BTreeMap<String, Dataset> = BTreeMap::new();

    for entry in WalkDir::new(&data_root).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "dvc") {
            continue;
        }

        let dvc_path = entry.path();
        let dvc: LegacyDvcFile = serde_yaml::from_str(&read_text(dvc_path)?).map_err(|source| {
            CatalogError::InvalidDvc {
                path: dvc_path.to_path_buf(),
                source,
            }
        })?;
        if dvc.outs.is_empty() {
            return Err(CatalogError::MissingDvcOutput(dvc_path.to_path_buf()));
        }

        for output in dvc.outs {
            if !is_safe_relative_path(Path::new(&output.path)) {
                return Err(CatalogError::UnsafeOutputPath {
                    dvc_path: dvc_path.to_path_buf(),
                    output_path: output.path,
                });
            }

            let logical_path = dvc_path
                .parent()
                .ok_or_else(|| CatalogError::InvalidDvcPath(dvc_path.to_path_buf()))?
                .join(&output.path);
            let relative_path = logical_path
                .strip_prefix(repository_root)
                .map_err(|_| CatalogError::PathOutsideRepository(logical_path.clone()))?;
            let dataset_id = infer_dataset_id(relative_path)?;
            let dataset_root = repository_root.join("data").join(&dataset_id);
            let dataset_root_relative = dataset_root
                .strip_prefix(repository_root)
                .map(path_to_slash)
                .map_err(|_| CatalogError::PathOutsideRepository(dataset_root.clone()))?;
            let file_id = logical_path
                .strip_prefix(&dataset_root)
                .map(path_to_slash)
                .map_err(|_| CatalogError::CannotInferDataset(relative_path.to_path_buf()))?;
            let schema_path = PathBuf::from(format!("{}.schema.yaml", logical_path.display()));
            let (schema_status, schema, schema_error) = read_schema(&schema_path)?;

            let dataset = datasets
                .entry(dataset_id.clone())
                .or_insert_with(|| Dataset {
                    id: dataset_id.clone(),
                    root: dataset_root_relative,
                    files: Vec::new(),
                });
            dataset.files.push(CatalogFile {
                id: file_id,
                path: path_to_slash(relative_path),
                object: format!("md5:{}", output.md5).parse().map_err(|_| {
                    CatalogError::InvalidObjectId {
                        path: dvc_path.to_path_buf(),
                        value: output.md5,
                    }
                })?,
                size: output.size,
                schema_status,
                schema_path: schema_path
                    .exists()
                    .then(|| {
                        schema_path
                            .strip_prefix(repository_root)
                            .map(path_to_slash)
                            .map_err(|_| CatalogError::PathOutsideRepository(schema_path.clone()))
                    })
                    .transpose()?,
                schema_error,
                schema,
            });
        }
    }

    for dataset in datasets.values_mut() {
        dataset
            .files
            .sort_by(|left, right| left.path.cmp(&right.path));
    }

    Ok(Catalog {
        datasets: datasets.into_values().collect(),
    })
}

/// Verifies and ingests all files in a legacy dataset, then builds Manifest v1.
///
/// # Errors
///
/// Returns an error when a source differs from its DVC metadata, storage fails,
/// or a catalog path is outside the declared dataset root.
pub fn ingest_legacy_dataset(
    repository_root: &Path,
    dataset: &Dataset,
    store: &LocalObjectStore,
) -> Result<IngestReport, CatalogError> {
    refresh_legacy_dataset(repository_root, dataset, store, None, true)
}

/// Refreshes a legacy dataset manifest while reusing unchanged local objects.
///
/// The fast path compares source and cached-object metadata. `rehash` disables
/// that path and verifies every source against its DVC digest.
///
/// # Errors
///
/// Returns an error when a changed source differs from its DVC metadata,
/// storage fails, or a catalog path is outside the declared dataset root.
pub fn refresh_legacy_dataset(
    repository_root: &Path,
    dataset: &Dataset,
    store: &LocalObjectStore,
    previous: Option<&DatasetManifest>,
    rehash: bool,
) -> Result<IngestReport, CatalogError> {
    let mut files = Vec::with_capacity(dataset.files.len());
    let mut added_objects = 0;
    let mut existing_objects = 0;
    let mut reused_files = 0;
    let mut bytes = 0_u64;

    for catalog_file in &dataset.files {
        let source = repository_root.join(&catalog_file.path);
        if !rehash {
            let previous_file = previous
                .filter(|manifest| {
                    manifest.dataset.id == dataset.id && manifest.dataset.root == dataset.root
                })
                .and_then(|manifest| {
                    manifest
                        .files
                        .iter()
                        .find(|file| file.id == catalog_file.id)
                });
            if let Some(previous_file) = previous_file
                .filter(|file| reusable_source(&source, catalog_file.size, store, &file.object))
            {
                bytes = bytes
                    .checked_add(previous_file.object.size)
                    .ok_or_else(|| CatalogError::LegacySizeMismatch {
                        path: source.clone(),
                        expected: u64::MAX,
                        actual: u64::MAX,
                    })?;
                files.push(ManifestFile {
                    id: catalog_file.id.clone(),
                    path: catalog_file.id.clone(),
                    object: previous_file.object.clone(),
                    schema_path: catalog_file.schema_path.clone(),
                });
                reused_files += 1;
                continue;
            }
        }
        verify_legacy_object(&source, &catalog_file.object, catalog_file.size)?;
        let ingested = store.ingest_file(&source)?;
        if ingested.already_present {
            existing_objects += 1;
        } else {
            added_objects += 1;
        }
        bytes += ingested.size;
        files.push(ManifestFile {
            id: catalog_file.id.clone(),
            path: catalog_file.id.clone(),
            object: ObjectRef {
                oid: ingested.oid,
                size: ingested.size,
            },
            schema_path: catalog_file.schema_path.clone(),
        });
    }

    let manifest = DatasetManifest {
        version: 1,
        dataset: ManifestDataset {
            id: dataset.id.clone(),
            root: dataset.root.clone(),
        },
        files,
    };
    manifest.validate().map_err(davis_core::StoreError::from)?;

    Ok(IngestReport {
        manifest,
        added_objects,
        existing_objects,
        reused_files,
        bytes,
    })
}

fn reusable_source(
    source: &Path,
    expected_size: u64,
    store: &LocalObjectStore,
    object: &ObjectRef,
) -> bool {
    if object.size != expected_size {
        return false;
    }
    let Ok(source_metadata) = source.metadata() else {
        return false;
    };
    let Ok(object_metadata) = store.object_path(&object.oid).metadata() else {
        return false;
    };
    if !source_metadata.is_file()
        || !object_metadata.is_file()
        || source_metadata.len() != expected_size
        || object_metadata.len() != object.size
    {
        return false;
    }
    let (Ok(source_modified), Ok(object_modified)) =
        (source_metadata.modified(), object_metadata.modified())
    else {
        return false;
    };
    source_modified <= object_modified
}

/// Verifies local files against the MD5 and size recorded by DVC.
///
/// # Errors
///
/// Returns an error at the first missing, unreadable, truncated, or modified
/// file.
pub fn audit_legacy_datasets(
    repository_root: &Path,
    datasets: &[&Dataset],
) -> Result<AuditReport, CatalogError> {
    let mut files = 0_usize;
    let mut bytes = 0_u64;
    for dataset in datasets {
        for catalog_file in &dataset.files {
            let source = repository_root.join(&catalog_file.path);
            verify_legacy_object(&source, &catalog_file.object, catalog_file.size)?;
            files += 1;
            bytes =
                bytes
                    .checked_add(catalog_file.size)
                    .ok_or(CatalogError::LegacySizeMismatch {
                        path: source,
                        expected: u64::MAX,
                        actual: u64::MAX,
                    })?;
        }
    }
    Ok(AuditReport { files, bytes })
}

fn read_text(path: &Path) -> Result<String, CatalogError> {
    fs::read_to_string(path).map_err(|source| CatalogError::Read {
        path: path.to_path_buf(),
        source,
    })
}

fn verify_legacy_object(
    path: &Path,
    expected_oid: &ObjectId,
    expected_size: u64,
) -> Result<(), CatalogError> {
    let mut input = fs::File::open(path).map_err(|source| CatalogError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Md5::new();
    let mut size = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|source| CatalogError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size = size
            .checked_add(
                u64::try_from(read).map_err(|_| CatalogError::LegacySizeMismatch {
                    path: path.to_path_buf(),
                    expected: expected_size,
                    actual: u64::MAX,
                })?,
            )
            .ok_or_else(|| CatalogError::LegacySizeMismatch {
                path: path.to_path_buf(),
                expected: expected_size,
                actual: u64::MAX,
            })?;
    }
    if size != expected_size {
        return Err(CatalogError::LegacySizeMismatch {
            path: path.to_path_buf(),
            expected: expected_size,
            actual: size,
        });
    }
    let actual = format!("{:x}", hasher.finalize());
    if expected_oid.algorithm() != "md5" || actual != expected_oid.digest() {
        return Err(CatalogError::LegacyDigestMismatch {
            path: path.to_path_buf(),
            expected: expected_oid.to_string(),
            actual: format!("md5:{actual}"),
        });
    }
    Ok(())
}

fn read_schema(
    schema_path: &Path,
) -> Result<(SchemaStatus, Option<FileSchema>, Option<String>), CatalogError> {
    if !schema_path.is_file() {
        return Ok((SchemaStatus::Missing, None, None));
    }

    let contents = read_text(schema_path)?;
    match serde_yaml::from_str::<LegacyFileSchema>(&contents) {
        Ok(schema) => Ok((SchemaStatus::Ready, Some(schema.into()), None)),
        Err(error) => Ok((SchemaStatus::Invalid, None, Some(error.to_string()))),
    }
}

fn is_safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn infer_dataset_id(path: &Path) -> Result<String, CatalogError> {
    let components: Vec<String> = path
        .strip_prefix("data")
        .map_err(|_| CatalogError::CannotInferDataset(path.to_path_buf()))?
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();

    let first = components
        .first()
        .ok_or_else(|| CatalogError::CannotInferDataset(path.to_path_buf()))?;
    if first == "PT_data" || first == "Tohoku_History" || components.len() == 1 {
        return Ok(first.clone());
    }

    Ok(format!("{first}/{}", components[1]))
}

fn path_to_slash(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy()),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => None,
            Component::ParentDir => Some("..".into()),
        })
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::{infer_dataset_id, refresh_legacy_dataset};
    use davis_core::{CatalogFile, Dataset, LocalObjectStore, SchemaStatus};
    use md5::{Digest, Md5};
    use std::fs;
    use std::path::Path;

    fn legacy_object(contents: &[u8]) -> davis_core::ObjectId {
        format!("md5:{:x}", Md5::digest(contents)).parse().unwrap()
    }

    #[test]
    fn infers_category_and_dataset() {
        assert_eq!(
            infer_dataset_id(Path::new("data/routes/Shibuya-2021/car/path.csv")).unwrap(),
            "routes/Shibuya-2021"
        );
    }

    #[test]
    fn keeps_legacy_top_level_datasets() {
        assert_eq!(
            infer_dataset_id(Path::new("data/PT_data/PTdata.csv")).unwrap(),
            "PT_data"
        );
        assert_eq!(
            infer_dataset_id(Path::new("data/Tohoku_History/df_individual.csv")).unwrap(),
            "Tohoku_History"
        );
    }

    #[test]
    fn refresh_reuses_unchanged_files_and_ingests_changed_files() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("data/routes/sample/source.csv");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        let first_contents = b"id,value\n1,first\n";
        fs::write(&source, first_contents).unwrap();
        let store = LocalObjectStore::new(temporary.path().join(".davis/cache"));
        let mut dataset = Dataset {
            id: "routes/sample".into(),
            root: "data/routes/sample".into(),
            files: vec![CatalogFile {
                id: "source.csv".into(),
                path: "data/routes/sample/source.csv".into(),
                object: legacy_object(first_contents),
                size: u64::try_from(first_contents.len()).unwrap(),
                schema_status: SchemaStatus::Missing,
                schema_path: None,
                schema_error: None,
                schema: None,
            }],
        };

        let first =
            refresh_legacy_dataset(temporary.path(), &dataset, &store, None, false).unwrap();
        assert_eq!(first.added_objects, 1);
        assert_eq!(first.reused_files, 0);

        let second = refresh_legacy_dataset(
            temporary.path(),
            &dataset,
            &store,
            Some(&first.manifest),
            false,
        )
        .unwrap();
        assert_eq!(second.added_objects, 0);
        assert_eq!(second.existing_objects, 0);
        assert_eq!(second.reused_files, 1);

        let changed_contents = b"id,value\n1,changed-and-longer\n";
        fs::write(&source, changed_contents).unwrap();
        dataset.files[0].object = legacy_object(changed_contents);
        dataset.files[0].size = u64::try_from(changed_contents.len()).unwrap();
        let changed = refresh_legacy_dataset(
            temporary.path(),
            &dataset,
            &store,
            Some(&second.manifest),
            false,
        )
        .unwrap();
        assert_eq!(changed.added_objects, 1);
        assert_eq!(changed.reused_files, 0);
        assert_ne!(
            changed.manifest.files[0].object.oid,
            first.manifest.files[0].object.oid
        );
    }
}
