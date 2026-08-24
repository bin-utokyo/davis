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
    current_local_date, hash_file, Catalog, CatalogFile, ColumnSchema, Dataset, DatasetManifest,
    FileSchema, LocalObjectStore, LocalizedText, ManifestDataset, ManifestFile, ObjectId,
    ObjectRef, SchemaStatus,
};
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
    #[error("schema was not found: {0}")]
    SchemaNotFound(PathBuf),
    #[error("invalid schema in {path}: {message}")]
    InvalidSchema { path: PathBuf, message: String },
    #[error(transparent)]
    Manifest(#[from] davis_core::ManifestError),
    #[error("catalog path is outside the repository: {0}")]
    PathOutsideRepository(PathBuf),
    #[error("cannot infer a dataset from path: {0}")]
    CannotInferDataset(PathBuf),
    #[error("file size mismatch for {path}: expected {expected}, found {actual}")]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("BLAKE3 mismatch for {path}: expected {expected}, found {actual}")]
    DigestMismatch {
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

/// Builds the catalog from versioned Davis manifests and companion schemas.
///
/// # Errors
///
/// Returns an error when a Manifest or schema cannot be read or a path would
/// escape the repository.
pub fn scan_repository(repository_root: &Path) -> Result<Catalog, CatalogError> {
    let manifest_root = repository_root.join(".davis/datasets");
    if !manifest_root.is_dir() {
        return Err(CatalogError::DataDirectoryNotFound(manifest_root));
    }

    let mut datasets = BTreeMap::new();
    for entry in WalkDir::new(&manifest_root).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file()
            || entry
                .path()
                .extension()
                .is_none_or(|extension| extension != "yaml")
        {
            continue;
        }
        let manifest = davis_core::read_manifest(entry.path())?;
        let mut files = Vec::with_capacity(manifest.files.len());
        for file in manifest.files {
            let logical_path = repository_root
                .join(&manifest.dataset.root)
                .join(&file.path);
            let schema_path = file.schema_path.as_ref().map_or_else(
                || PathBuf::from(format!("{}.schema.yaml", logical_path.display())),
                |path| repository_root.join(path),
            );
            let (schema_status, schema, schema_error) = read_schema(&schema_path)?;
            files.push(CatalogFile {
                id: file.id,
                path: path_to_slash(&Path::new(&manifest.dataset.root).join(&file.path)),
                object: file.object.oid,
                size: file.object.size,
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
        files.sort_by(|left, right| left.path.cmp(&right.path));
        datasets.insert(
            manifest.dataset.id.clone(),
            Dataset {
                id: manifest.dataset.id,
                root: manifest.dataset.root,
                files,
            },
        );
    }

    Ok(Catalog {
        datasets: datasets.into_values().collect(),
    })
}

/// Ingests every primary file below a dataset root and builds Manifest v1.
///
/// # Errors
///
/// Returns an error when a source cannot be hashed, storage fails, or a catalog
/// path is outside the declared dataset root.
pub fn ingest_dataset(
    repository_root: &Path,
    dataset: &Dataset,
    store: &LocalObjectStore,
) -> Result<IngestReport, CatalogError> {
    let updated_on = current_local_date();
    refresh_dataset(
        repository_root,
        &dataset.id,
        &dataset.root,
        store,
        RefreshOptions {
            previous: None,
            rehash: true,
            write_objects: true,
            updated_on: Some(&updated_on),
        },
    )
}

/// Controls how a dataset Manifest is refreshed from local files.
#[derive(Debug, Clone, Copy)]
pub struct RefreshOptions<'a> {
    pub previous: Option<&'a DatasetManifest>,
    pub rehash: bool,
    pub write_objects: bool,
    pub updated_on: Option<&'a str>,
}

/// Rebuilds a dataset manifest directly from local files without DVC metadata.
///
/// # Errors
///
/// Returns an error when a source cannot be hashed, storage fails, or a catalog
/// path is outside the declared dataset root.
pub fn refresh_dataset(
    repository_root: &Path,
    dataset_id: &str,
    dataset_root: &str,
    store: &LocalObjectStore,
    options: RefreshOptions<'_>,
) -> Result<IngestReport, CatalogError> {
    let root = repository_root.join(dataset_root);
    if !root.is_dir() {
        return Err(CatalogError::DataDirectoryNotFound(root));
    }
    let mut sources = WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_map(|entry| match entry {
            Ok(entry) if entry.file_type().is_file() && is_primary_file(entry.path()) => {
                Some(Ok(entry.into_path()))
            }
            Ok(_) => None,
            Err(error) => Some(Err(CatalogError::Walk(error))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    sources.sort();

    let mut files = Vec::with_capacity(sources.len());
    let mut added_objects = 0;
    let mut existing_objects = 0;
    let mut reused_files = 0;
    let mut bytes = 0_u64;
    for source in sources {
        let file_id = source
            .strip_prefix(&root)
            .map(path_to_slash)
            .map_err(|_| CatalogError::PathOutsideRepository(source.clone()))?;
        let previous_file = options
            .previous
            .filter(|manifest| {
                manifest.dataset.id == dataset_id && manifest.dataset.root == dataset_root
            })
            .and_then(|manifest| manifest.files.iter().find(|file| file.id == file_id));
        let reusable = previous_file
            .filter(|file| !options.rehash && reusable_source(&source, store, &file.object));
        let object = if let Some(file) = reusable {
            reused_files += 1;
            file.object.clone()
        } else if options.write_objects {
            let ingested = store.ingest_file(&source)?;
            if ingested.already_present {
                existing_objects += 1;
            } else {
                added_objects += 1;
            }
            ObjectRef {
                oid: ingested.oid,
                size: ingested.size,
            }
        } else {
            added_objects += 1;
            hash_file(&source)?
        };
        bytes = bytes
            .checked_add(object.size)
            .ok_or_else(|| CatalogError::SizeMismatch {
                path: source.clone(),
                expected: u64::MAX,
                actual: u64::MAX,
            })?;
        let schema_path = PathBuf::from(format!("{}.schema.yaml", source.display()));
        let updated_at = match previous_file {
            Some(previous) if previous.object == object => previous.updated_at.clone(),
            Some(_) | None => options.updated_on.map(str::to_owned),
        };
        files.push(ManifestFile {
            id: file_id.clone(),
            path: file_id,
            object,
            updated_at,
            schema_path: schema_path
                .is_file()
                .then(|| {
                    schema_path
                        .strip_prefix(repository_root)
                        .map(path_to_slash)
                        .map_err(|_| CatalogError::PathOutsideRepository(schema_path.clone()))
                })
                .transpose()?,
        });
    }

    let manifest = DatasetManifest {
        version: 1,
        dataset: ManifestDataset {
            id: dataset_id.to_owned(),
            root: dataset_root.to_owned(),
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

fn reusable_source(source: &Path, store: &LocalObjectStore, object: &ObjectRef) -> bool {
    let Ok(source_metadata) = source.metadata() else {
        return false;
    };
    let Ok(object_metadata) = store.object_path(&object.oid).metadata() else {
        return false;
    };
    if !source_metadata.is_file()
        || !object_metadata.is_file()
        || source_metadata.len() != object.size
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

/// Verifies local files against the BLAKE3 object IDs in Davis manifests.
///
/// # Errors
///
/// Returns an error at the first missing, unreadable, truncated, or modified
/// file.
pub fn audit_datasets(
    repository_root: &Path,
    datasets: &[&Dataset],
) -> Result<AuditReport, CatalogError> {
    let mut files = 0_usize;
    let mut bytes = 0_u64;
    for dataset in datasets {
        for catalog_file in &dataset.files {
            let source = repository_root.join(&catalog_file.path);
            verify_object(&source, &catalog_file.object, catalog_file.size)?;
            files += 1;
            bytes = bytes
                .checked_add(catalog_file.size)
                .ok_or(CatalogError::SizeMismatch {
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

fn verify_object(
    path: &Path,
    expected_oid: &ObjectId,
    expected_size: u64,
) -> Result<(), CatalogError> {
    let mut input = fs::File::open(path).map_err(|source| CatalogError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = blake3::Hasher::new();
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
            .checked_add(u64::try_from(read).map_err(|_| CatalogError::SizeMismatch {
                path: path.to_path_buf(),
                expected: expected_size,
                actual: u64::MAX,
            })?)
            .ok_or_else(|| CatalogError::SizeMismatch {
                path: path.to_path_buf(),
                expected: expected_size,
                actual: u64::MAX,
            })?;
    }
    if size != expected_size {
        return Err(CatalogError::SizeMismatch {
            path: path.to_path_buf(),
            expected: expected_size,
            actual: size,
        });
    }
    let actual = hasher.finalize().to_hex().to_string();
    if expected_oid.algorithm() != "blake3" || actual != expected_oid.digest() {
        return Err(CatalogError::DigestMismatch {
            path: path.to_path_buf(),
            expected: expected_oid.to_string(),
            actual: format!("blake3:{actual}"),
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

/// Reads one file schema using the repository's current YAML compatibility format.
///
/// # Errors
///
/// Returns an error when the schema is missing, unreadable, or invalid.
pub fn read_file_schema(schema_path: &Path) -> Result<FileSchema, CatalogError> {
    let (status, schema, error) = read_schema(schema_path)?;
    match (status, schema) {
        (SchemaStatus::Ready, Some(schema)) => Ok(schema),
        (SchemaStatus::Missing, _) => Err(CatalogError::SchemaNotFound(schema_path.to_path_buf())),
        _ => Err(CatalogError::InvalidSchema {
            path: schema_path.to_path_buf(),
            message: error.unwrap_or_else(|| "schema could not be parsed".to_owned()),
        }),
    }
}

fn is_primary_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name != ".DS_Store"
        && name != ".gitignore"
        && !path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dvc"))
        && !name.ends_with(".schema.yaml")
        && !name.ends_with(".ja.pdf")
        && !name.ends_with(".en.pdf")
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
    use super::{refresh_dataset, RefreshOptions};
    use davis_core::LocalObjectStore;
    use std::fs;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn refresh_hashes_primary_files_without_dvc_metadata() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("data/routes/sample/source.csv");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        let first_contents = b"id,value\n1,first\n";
        fs::write(&source, first_contents).unwrap();
        fs::write(source.with_extension("csv.dvc"), "legacy").unwrap();
        fs::write(
            temporary
                .path()
                .join("data/routes/sample/source.csv.schema.yaml"),
            "schema",
        )
        .unwrap();
        let store = LocalObjectStore::new(temporary.path().join(".davis/cache"));

        let first = refresh_dataset(
            temporary.path(),
            "routes/sample",
            "data/routes/sample",
            &store,
            RefreshOptions {
                previous: None,
                rehash: false,
                write_objects: true,
                updated_on: Some("2026-08-20"),
            },
        )
        .unwrap();
        assert_eq!(first.added_objects, 1);
        assert_eq!(first.manifest.files.len(), 1);
        assert_eq!(first.manifest.files[0].id, "source.csv");
        assert_eq!(first.manifest.files[0].object.oid.algorithm(), "blake3");
        assert_eq!(
            first.manifest.files[0].updated_at.as_deref(),
            Some("2026-08-20")
        );
        assert!(first.manifest.files[0].schema_path.is_some());

        let unchanged = refresh_dataset(
            temporary.path(),
            "routes/sample",
            "data/routes/sample",
            &store,
            RefreshOptions {
                previous: Some(&first.manifest),
                rehash: false,
                write_objects: true,
                updated_on: Some("2026-08-21"),
            },
        )
        .unwrap();
        assert_eq!(unchanged.reused_files, 1);
        assert_eq!(unchanged.added_objects, 0);
        assert_eq!(
            unchanged.manifest.files[0].updated_at.as_deref(),
            Some("2026-08-20")
        );

        let changed_contents = b"id,value\n1,changed-and-longer\n";
        fs::write(&source, changed_contents).unwrap();
        let changed = refresh_dataset(
            temporary.path(),
            "routes/sample",
            "data/routes/sample",
            &store,
            RefreshOptions {
                previous: Some(&unchanged.manifest),
                rehash: false,
                write_objects: true,
                updated_on: Some("2026-08-22"),
            },
        )
        .unwrap();
        assert_eq!(changed.added_objects, 1);
        assert_ne!(
            changed.manifest.files[0].object.oid,
            first.manifest.files[0].object.oid
        );
        assert_eq!(
            changed.manifest.files[0].updated_at.as_deref(),
            Some("2026-08-22")
        );

        let mut unknown = changed.manifest.clone();
        unknown.files[0].updated_at = None;
        let preserved_unknown = refresh_dataset(
            temporary.path(),
            "routes/sample",
            "data/routes/sample",
            &store,
            RefreshOptions {
                previous: Some(&unknown),
                rehash: false,
                write_objects: true,
                updated_on: Some("2026-08-23"),
            },
        )
        .unwrap();
        assert_eq!(preserved_unknown.manifest.files[0].updated_at, None);

        let dry_run_store = LocalObjectStore::new(temporary.path().join("dry-run-cache"));
        let dry_run = refresh_dataset(
            temporary.path(),
            "routes/sample",
            "data/routes/sample",
            &dry_run_store,
            RefreshOptions {
                previous: Some(&changed.manifest),
                rehash: false,
                write_objects: false,
                updated_on: None,
            },
        )
        .unwrap();
        assert_eq!(dry_run.added_objects, 1);
        assert_eq!(
            dry_run.manifest.files[0].updated_at.as_deref(),
            Some("2026-08-22")
        );
        assert!(!dry_run_store.root().exists());
    }
}
