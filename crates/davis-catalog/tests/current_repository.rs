use std::path::PathBuf;

use davis_catalog::{build_catalog_index, scan_repository, write_catalog_index};
use davis_core::{read_manifest, SchemaStatus};

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

#[test]
fn current_repository_has_no_catalog_coverage_regression() {
    let catalog = scan_repository(&repository_root()).unwrap();

    assert_eq!(catalog.file_count(), 258);
    assert_eq!(catalog.schema_ready_count(), 176);
    assert_eq!(
        catalog
            .datasets
            .iter()
            .flat_map(|dataset| &dataset.files)
            .filter(|file| file.schema_status == SchemaStatus::Invalid)
            .count(),
        0
    );
    assert!(catalog.dataset("PT_data").is_some());
    assert!(catalog.dataset("Tohoku_History").is_some());
    assert!(catalog.dataset("routes/Toyosu-2018-2021").is_some());
}

#[test]
fn generated_manifests_cover_the_current_catalog() {
    let root = repository_root();
    let catalog = scan_repository(&root).unwrap();
    let mut manifest_files = 0_usize;
    let mut schema_references = 0_usize;

    for dataset in &catalog.datasets {
        let manifest_path = root
            .join(".davis/datasets")
            .join(format!("{}.yaml", dataset.id));
        let manifest = read_manifest(&manifest_path).unwrap();
        assert_eq!(manifest.dataset.id, dataset.id);
        assert_eq!(manifest.dataset.root, dataset.root);
        assert_eq!(manifest.files.len(), dataset.files.len());
        assert!(manifest
            .files
            .iter()
            .all(|file| file.object.oid.algorithm() == "blake3"));
        assert!(manifest
            .files
            .iter()
            .all(|file| file.updated_at.as_deref() == Some("2026-08-24")));
        manifest_files += manifest.files.len();
        schema_references += manifest
            .files
            .iter()
            .filter(|file| file.schema_path.is_some())
            .count();
    }

    assert_eq!(manifest_files, 258);
    assert_eq!(schema_references, 176);
}

#[test]
fn static_index_covers_every_file_and_schema() {
    let root = repository_root();
    let catalog = scan_repository(&root).unwrap();
    let index = build_catalog_index(&root, &catalog).unwrap();

    assert_eq!(index.summary.dataset_count, 15);
    assert_eq!(index.summary.file_count, 258);
    assert_eq!(index.summary.schema_ready_count, 176);
    assert_eq!(index.files.len(), 258);
    assert!(index
        .files
        .iter()
        .all(|file| file.object.oid.algorithm() == "blake3"));
    assert!(index.files.iter().all(|file| file.updated_at.is_some()));
    assert!(index.datasets.iter().all(|dataset| {
        dataset.updated_at
            == index
                .files
                .iter()
                .filter(|file| file.dataset_id == dataset.id)
                .filter_map(|file| file.updated_at.as_ref())
                .max()
                .cloned()
    }));
    assert!(index.columns.len() > 1_000);
    assert!(index.files.iter().any(|file| {
        file.dataset_id == "network/matsuyama"
            && file
                .raw_schema
                .as_deref()
                .is_some_and(|raw| raw.contains("columns:"))
    }));
    assert!(index.facets.formats.contains(&"csv".to_owned()));
    assert!(!index.facets.licenses.is_empty());

    let output = tempfile::tempdir().unwrap();
    write_catalog_index(output.path(), &index).unwrap();
    for name in [
        "index.json",
        "datasets.json",
        "files.json",
        "columns.json",
        "facets.json",
    ] {
        assert!(output.path().join(name).is_file());
    }
}
