use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use davis_core::{
    read_manifest, Catalog, ColumnSchema, FileSchema, LocalizedText, ObjectRef, SchemaStatus,
};
use serde::Serialize;

use crate::CatalogError;

/// Stable metadata consumed by static clients and future HTTP APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CatalogIndex {
    pub version: u32,
    pub summary: CatalogSummary,
    pub datasets: Vec<IndexedDataset>,
    pub files: Vec<IndexedFile>,
    pub columns: Vec<IndexedColumn>,
    pub facets: CatalogFacets,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CatalogSummary {
    pub dataset_count: usize,
    pub file_count: usize,
    pub schema_ready_count: usize,
    pub total_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexedDataset {
    pub id: String,
    pub root: String,
    pub file_count: usize,
    pub schema_ready_count: usize,
    pub total_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexedFile {
    pub id: String,
    pub dataset_id: String,
    pub file_id: String,
    pub path: String,
    pub size: u64,
    pub object: ObjectRef,
    pub format: String,
    pub schema_status: SchemaStatus,
    pub schema_path: Option<String>,
    pub schema_error: Option<String>,
    pub name: Option<LocalizedText>,
    pub description: Option<LocalizedText>,
    pub city: Option<LocalizedText>,
    pub year: Option<i64>,
    pub license: Option<LocalizedText>,
    pub columns: Vec<ColumnSchema>,
    pub raw_schema: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexedColumn {
    pub dataset_id: String,
    pub file_id: String,
    pub name: String,
    pub data_type: String,
    pub description: Option<LocalizedText>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CatalogFacets {
    pub cities: Vec<LocalizedText>,
    pub years: Vec<i64>,
    pub formats: Vec<String>,
    pub licenses: Vec<LocalizedText>,
    pub schema_statuses: Vec<SchemaStatus>,
}

/// Builds a deterministic, derived search index from the catalog and source schemas.
///
/// # Errors
///
/// Returns an error when a referenced schema cannot be read or is outside the
/// repository.
pub fn build_catalog_index(
    repository_root: &Path,
    catalog: &Catalog,
) -> Result<CatalogIndex, CatalogError> {
    let mut datasets = Vec::with_capacity(catalog.datasets.len());
    let mut files = Vec::with_capacity(catalog.file_count());
    let mut columns = Vec::new();
    let mut cities = BTreeMap::new();
    let mut years = BTreeSet::new();
    let mut formats = BTreeSet::new();
    let mut licenses = BTreeMap::new();
    let mut schema_statuses = BTreeSet::new();
    let mut total_size = 0_u64;

    for dataset in &catalog.datasets {
        let manifest_path = repository_root
            .join(".davis/datasets")
            .join(format!("{}.yaml", dataset.id));
        let manifest = manifest_path
            .is_file()
            .then(|| read_manifest(&manifest_path))
            .transpose()?;
        datasets.push(IndexedDataset {
            id: dataset.id.clone(),
            root: dataset.root.clone(),
            file_count: dataset.files.len(),
            schema_ready_count: dataset.schema_ready_count(),
            total_size: dataset.total_size(),
        });
        total_size = total_size
            .checked_add(dataset.total_size())
            .ok_or_else(|| CatalogError::Write {
                path: repository_root.to_path_buf(),
                source: std::io::Error::other("catalog size overflow"),
            })?;

        for file in &dataset.files {
            let format = file_format(&file.id);
            formats.insert(format.clone());
            schema_statuses.insert(schema_status_key(file.schema_status));
            let raw_schema = file
                .schema_path
                .as_deref()
                .map(|path| read_repository_text(repository_root, path))
                .transpose()?;
            let schema = file.schema.as_ref();
            collect_schema_facets(schema, &mut cities, &mut years, &mut licenses);
            if let Some(schema) = schema {
                columns.extend(schema.columns.iter().map(|column| IndexedColumn {
                    dataset_id: dataset.id.clone(),
                    file_id: file.id.clone(),
                    name: column.name.clone(),
                    data_type: column.data_type.clone(),
                    description: column.description.clone(),
                }));
            }
            let object = manifest
                .as_ref()
                .and_then(|value| value.files.iter().find(|item| item.id == file.id))
                .map_or_else(
                    || ObjectRef {
                        oid: file.object.clone(),
                        size: file.size,
                    },
                    |item| item.object.clone(),
                );
            files.push(index_file(
                dataset.id.as_str(),
                file,
                object,
                format,
                raw_schema,
            ));
        }
    }

    Ok(CatalogIndex {
        version: 1,
        summary: CatalogSummary {
            dataset_count: datasets.len(),
            file_count: files.len(),
            schema_ready_count: catalog.schema_ready_count(),
            total_size,
        },
        datasets,
        files,
        columns,
        facets: CatalogFacets {
            cities: cities.into_values().collect(),
            years: years.into_iter().collect(),
            formats: formats.into_iter().collect(),
            licenses: licenses.into_values().collect(),
            schema_statuses: schema_statuses
                .into_iter()
                .map(schema_status_from_key)
                .collect(),
        },
    })
}

/// Writes the complete index and split static API documents.
///
/// # Errors
///
/// Returns an error when JSON serialization or filesystem writes fail.
pub fn write_catalog_index(
    output_directory: &Path,
    index: &CatalogIndex,
) -> Result<(), CatalogError> {
    fs::create_dir_all(output_directory).map_err(|source| CatalogError::Write {
        path: output_directory.to_path_buf(),
        source,
    })?;
    write_json(output_directory.join("index.json"), index)?;
    write_json(output_directory.join("datasets.json"), &index.datasets)?;
    write_json(output_directory.join("files.json"), &index.files)?;
    write_json(output_directory.join("columns.json"), &index.columns)?;
    write_json(output_directory.join("facets.json"), &index.facets)?;
    Ok(())
}

fn index_file(
    dataset_id: &str,
    file: &davis_core::CatalogFile,
    object: ObjectRef,
    format: String,
    raw_schema: Option<String>,
) -> IndexedFile {
    let schema = file.schema.as_ref();
    IndexedFile {
        id: format!("{dataset_id}:{}", file.id),
        dataset_id: dataset_id.to_owned(),
        file_id: file.id.clone(),
        path: file.path.clone(),
        size: file.size,
        object,
        format,
        schema_status: file.schema_status,
        schema_path: file.schema_path.clone(),
        schema_error: file.schema_error.clone(),
        name: schema.map(|value| value.name.clone()),
        description: schema.and_then(|value| value.description.clone()),
        city: schema.and_then(|value| value.city.clone()),
        year: schema.and_then(|value| value.year),
        license: schema.and_then(|value| value.license.clone()),
        columns: schema.map_or_else(Vec::new, |value| value.columns.clone()),
        raw_schema,
    }
}

fn collect_schema_facets(
    schema: Option<&FileSchema>,
    cities: &mut BTreeMap<String, LocalizedText>,
    years: &mut BTreeSet<i64>,
    licenses: &mut BTreeMap<String, LocalizedText>,
) {
    let Some(schema) = schema else {
        return;
    };
    if let Some(city) = &schema.city {
        cities.insert(localized_key(city), city.clone());
    }
    if let Some(year) = schema.year {
        years.insert(year);
    }
    if let Some(license) = &schema.license {
        licenses.insert(localized_key(license), license.clone());
    }
}

fn localized_key(value: &LocalizedText) -> String {
    format!("{}\0{}", value.ja, value.en)
}

fn file_format(file_id: &str) -> String {
    Path::new(file_id)
        .extension()
        .and_then(|extension| extension.to_str())
        .map_or_else(|| "unknown".to_owned(), str::to_ascii_lowercase)
}

fn read_repository_text(
    repository_root: &Path,
    relative_path: &str,
) -> Result<String, CatalogError> {
    let path = repository_root.join(relative_path);
    if !path.starts_with(repository_root) {
        return Err(CatalogError::PathOutsideRepository(path));
    }
    fs::read_to_string(&path).map_err(|source| CatalogError::Read { path, source })
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), CatalogError> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&path, bytes).map_err(|source| CatalogError::Write { path, source })
}

fn schema_status_key(status: SchemaStatus) -> u8 {
    match status {
        SchemaStatus::Ready => 0,
        SchemaStatus::Missing => 1,
        SchemaStatus::Invalid => 2,
    }
}

fn schema_status_from_key(key: u8) -> SchemaStatus {
    match key {
        0 => SchemaStatus::Ready,
        1 => SchemaStatus::Missing,
        _ => SchemaStatus::Invalid,
    }
}
