mod remote;
mod session;

use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use davis_catalog::{
    audit_legacy_datasets, build_catalog_index, ingest_legacy_dataset, scan_legacy_repository,
    write_catalog_index,
};
use davis_core::{read_manifest, write_manifest, Dataset, LocalObjectStore, SchemaStatus};
use davis_storage::{
    read_storage_configuration, ObjectStorage, RemoteConfig, S3Credentials, StorageError,
    TransferProgress,
};
use indicatif::{ProgressBar, ProgressStyle};
use remote::DavisService;

#[derive(Debug, Parser)]
#[command(name = "davis", version, about = "Davis data catalog client")]
struct Cli {
    /// Davis repository to read.
    #[arg(long, global = true, default_value = ".")]
    repository: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Exchange a participant invite code for a CLI session.
    Login {
        /// Davis Web service URL, for example <https://davis.example.pages.dev>.
        service_url: String,
        /// Read the invite code from standard input instead of prompting.
        #[arg(long)]
        invite_code_stdin: bool,
    },
    /// Remove the locally stored CLI session.
    Logout,
    /// List available datasets.
    List {
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show files and schema coverage for one dataset.
    Info {
        dataset_id: String,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Generate the static catalog API consumed by Web clients.
    Index {
        /// Directory where JSON index files are written.
        #[arg(short, long, default_value = "web/davis-web/public/catalog")]
        out: PathBuf,
    },
    /// Verify legacy files and ingest them into a local content-addressed store.
    Ingest {
        /// One dataset to ingest.
        dataset_id: Option<String>,
        /// Ingest every dataset in the current catalog.
        #[arg(long, conflicts_with = "dataset_id")]
        all: bool,
        /// Local object storage root.
        #[arg(long, default_value = ".davis/cache")]
        store: PathBuf,
        /// Directory where `DatasetManifest` YAML files are written.
        #[arg(long, default_value = ".davis/datasets")]
        manifest_directory: PathBuf,
    },
    /// Materialize a dataset from its Manifest and local object storage.
    Get {
        dataset_id: String,
        /// Davis Web service URL. When no matching CLI session exists, prompt for an invite code.
        #[arg(long)]
        service_url: Option<String>,
        /// Materialize only these logical file IDs. Repeat for multiple files.
        #[arg(long = "file")]
        files: Vec<String>,
        /// Local object storage root.
        #[arg(long, default_value = ".davis/cache")]
        store: PathBuf,
        /// Directory containing `DatasetManifest` YAML files.
        #[arg(long, default_value = ".davis/datasets")]
        manifest_directory: PathBuf,
        /// Output root. Dataset paths are recreated below this directory.
        #[arg(short, long, default_value = ".")]
        out: PathBuf,
        /// Replace files that already exist.
        #[arg(long)]
        force: bool,
        /// Storage configuration used to fill missing cache objects.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Named remote from the configuration.
        #[arg(long, default_value = "default")]
        remote: String,
    },
    /// Verify local files against the size and MD5 recorded by DVC.
    Verify {
        /// Verify only one dataset. All datasets are checked when omitted.
        dataset_id: Option<String>,
    },
    /// Upload missing content-addressed objects without deleting remote data.
    Push {
        /// One dataset to upload.
        dataset_id: Option<String>,
        /// Upload every dataset in the current catalog.
        #[arg(long, conflicts_with = "dataset_id")]
        all: bool,
        /// Local object storage root populated by `davis ingest`.
        #[arg(long, default_value = ".davis/cache")]
        store: PathBuf,
        /// Directory containing `DatasetManifest` YAML files.
        #[arg(long, default_value = ".davis/datasets")]
        manifest_directory: PathBuf,
        /// Versioned storage configuration.
        #[arg(long, default_value = ".davis/config.toml")]
        config: PathBuf,
        /// Named remote from the configuration.
        #[arg(long, default_value = "default")]
        remote: String,
        /// Show missing objects without uploading them.
        #[arg(long)]
        dry_run: bool,
    },
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(Cli::parse()).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Login {
            service_url,
            invite_code_stdin,
        } => handle_login(&service_url, invite_code_stdin).await?,
        Command::Logout => handle_logout()?,
        Command::List { json } => handle_list(&cli.repository, json).await?,
        Command::Info { dataset_id, json } => {
            handle_info(&cli.repository, &dataset_id, json).await?;
        }
        Command::Index { out } => handle_index(&cli.repository, &out)?,
        Command::Ingest {
            dataset_id,
            all,
            store,
            manifest_directory,
        } => {
            handle_ingest(
                &cli.repository,
                dataset_id.as_deref(),
                all,
                &store,
                &manifest_directory,
            )?;
        }
        Command::Get {
            dataset_id,
            service_url,
            files,
            store,
            manifest_directory,
            out,
            force,
            config,
            remote,
        } => {
            handle_get(GetRequest {
                repository: cli.repository,
                dataset_id,
                service_url,
                files,
                store,
                manifest_directory,
                out,
                force,
                config,
                remote,
            })
            .await?;
        }
        Command::Verify { dataset_id } => handle_verify(&cli.repository, dataset_id.as_deref())?,
        Command::Push {
            dataset_id,
            all,
            store,
            manifest_directory,
            config,
            remote,
            dry_run,
        } => {
            handle_push(PushRequest {
                repository: cli.repository,
                dataset_id,
                all,
                store,
                manifest_directory,
                config,
                remote,
                dry_run,
            })
            .await?;
        }
    }

    Ok(())
}

struct PushRequest {
    repository: PathBuf,
    dataset_id: Option<String>,
    all: bool,
    store: PathBuf,
    manifest_directory: PathBuf,
    config: PathBuf,
    remote: String,
    dry_run: bool,
}

struct GetRequest {
    repository: PathBuf,
    dataset_id: String,
    service_url: Option<String>,
    files: Vec<String>,
    store: PathBuf,
    manifest_directory: PathBuf,
    out: PathBuf,
    force: bool,
    config: Option<PathBuf>,
    remote: String,
}

const CATALOG_DOCUMENTS: [&str; 5] = [
    "index.json",
    "datasets.json",
    "files.json",
    "columns.json",
    "facets.json",
];

async fn handle_login(
    service_url: &str,
    invite_code_stdin: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let invite_code = if invite_code_stdin {
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input)?;
        input.trim_end_matches(['\r', '\n']).to_owned()
    } else if std::io::stdin().is_terminal() {
        rpassword::prompt_password("Invite code: ")?
    } else {
        return Err("standard input is not a terminal; use --invite-code-stdin".into());
    };
    if invite_code.is_empty() {
        return Err("invite code must not be empty".into());
    }
    let service = DavisService::new(service_url, None)?;
    let login = service.exchange_invite_code(&invite_code).await?;
    let stored =
        session::Session::new(service.base_url().to_owned(), login.token, login.expires_at);
    let path = session::save(&stored)?;
    println!("Logged in to {}", stored.service_url);
    println!("Session expires: {}", stored.expires_at);
    println!("Session: {}", path.display());
    Ok(())
}

fn handle_logout() -> Result<(), Box<dyn std::error::Error>> {
    if session::clear()? {
        println!("Logged out");
    } else {
        println!("No stored session");
    }
    Ok(())
}

async fn handle_list(
    repository: &std::path::Path,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = if let Some(stored) = session::load()? {
        DavisService::new(&stored.service_url, Some(stored.token))?
            .catalog()
            .await?
    } else {
        scan_legacy_repository(repository)?
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&catalog)?);
    } else {
        print_dataset_list(&catalog.datasets);
    }
    Ok(())
}

async fn handle_info(
    repository: &std::path::Path,
    dataset_id: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = if let Some(stored) = session::load()? {
        DavisService::new(&stored.service_url, Some(stored.token))?
            .catalog()
            .await?
    } else {
        scan_legacy_repository(repository)?
    };
    let dataset = catalog
        .dataset(dataset_id)
        .ok_or_else(|| format!("dataset was not found: {dataset_id}"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(dataset)?);
    } else {
        print_dataset_info(dataset);
    }
    Ok(())
}

fn handle_index(
    repository: &std::path::Path,
    output_directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = scan_legacy_repository(repository)?;
    let index = build_catalog_index(repository, &catalog)?;
    let output_directory = resolve(repository, output_directory);
    write_catalog_index(&output_directory, &index)?;
    println!("Catalog index: {}", output_directory.display());
    println!("Datasets: {}", index.summary.dataset_count);
    println!("Files: {}", index.summary.file_count);
    println!("Columns: {}", index.columns.len());
    Ok(())
}

fn handle_ingest(
    repository: &std::path::Path,
    dataset_id: Option<&str>,
    all: bool,
    store: &std::path::Path,
    manifest_directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if dataset_id.is_none() && !all {
        return Err("provide a dataset ID or use --all".into());
    }
    let catalog = scan_legacy_repository(repository)?;
    let object_store = LocalObjectStore::new(resolve(repository, store));
    let selected: Vec<&Dataset> = if all {
        catalog.datasets.iter().collect()
    } else {
        let dataset_id = dataset_id.ok_or("provide a dataset ID or use --all")?;
        vec![catalog
            .dataset(dataset_id)
            .ok_or_else(|| format!("dataset was not found: {dataset_id}"))?]
    };
    let mut total_files = 0_usize;
    let mut total_bytes = 0_u64;
    for dataset in selected {
        let report = ingest_legacy_dataset(repository, dataset, &object_store)?;
        let manifest_path =
            resolve(repository, manifest_directory).join(format!("{}.yaml", dataset.id));
        write_manifest(&manifest_path, &report.manifest)?;
        total_files += report.manifest.files.len();
        total_bytes = total_bytes
            .checked_add(report.bytes)
            .ok_or("ingested byte counter overflow")?;
        println!(
            "Ingested {}: {} files, {}",
            dataset.id,
            report.manifest.files.len(),
            human_size(report.bytes)
        );
    }
    println!(
        "Manifest directory: {}",
        resolve(repository, manifest_directory).display()
    );
    println!("Store: {}", object_store.root().display());
    println!("Total: {total_files} files, {}", human_size(total_bytes));
    Ok(())
}

async fn handle_get(request: GetRequest) -> Result<(), Box<dyn std::error::Error>> {
    let stored_session = if request.config.is_none() {
        get_session(request.service_url.as_deref()).await?
    } else {
        None
    };
    let manifest = if let Some(stored) = &stored_session {
        DavisService::new(&stored.service_url, Some(stored.token.clone()))?
            .manifest(&request.dataset_id)
            .await?
    } else {
        let manifest_path = resolve(&request.repository, &request.manifest_directory)
            .join(format!("{}.yaml", request.dataset_id));
        read_manifest(&manifest_path)?
    };
    if manifest.dataset.id != request.dataset_id {
        return Err(format!(
            "manifest dataset ID mismatch: expected {}, found {}",
            request.dataset_id, manifest.dataset.id
        )
        .into());
    }
    let manifest = if request.files.is_empty() {
        manifest
    } else {
        manifest.select_files(&request.files)?
    };
    let object_store = LocalObjectStore::new(resolve(&request.repository, &request.store));
    if let Some(config) = &request.config {
        let remote_store = open_remote(&request.repository, config, &request.remote)?;
        let progress_bar = transfer_progress_bar("Download");
        let result = remote_store
            .download_manifest_with_progress(&object_store, &manifest, |progress| {
                update_transfer_progress(&progress_bar, "Download", progress);
            })
            .await;
        let report = match result {
            Ok(report) => {
                progress_bar.finish_with_message("Download complete");
                report
            }
            Err(error) => {
                progress_bar.abandon_with_message("Download failed");
                return Err(error.into());
            }
        };
        println!("Downloaded objects: {}", report.downloaded);
        println!("Cached objects: {}", report.cached);
    } else if let Some(stored) = stored_session {
        let service = DavisService::new(&stored.service_url, Some(stored.token))?;
        let progress_bar = transfer_progress_bar("Download");
        let result = service
            .download_manifest(
                &object_store,
                &manifest,
                |completed_bytes, total_bytes, completed_objects, total_objects| {
                    progress_bar.set_length(total_bytes);
                    progress_bar.set_position(completed_bytes.min(total_bytes));
                    progress_bar
                        .set_message(format!("Download {completed_objects}/{total_objects}"));
                },
            )
            .await;
        let report = match result {
            Ok(report) => {
                progress_bar.finish_with_message("Download complete");
                report
            }
            Err(error) => {
                progress_bar.abandon_with_message("Download failed");
                return Err(error.into());
            }
        };
        println!("Downloaded objects: {}", report.downloaded);
        println!("Cached objects: {}", report.cached);
    }
    let output = resolve(&request.repository, &request.out);
    object_store.materialize(&manifest, &output, request.force)?;
    println!(
        "Materialized {} files under {}",
        manifest.files.len(),
        output.join(&manifest.dataset.root).display()
    );
    Ok(())
}

async fn get_session(
    service_url: Option<&str>,
) -> Result<Option<session::Session>, Box<dyn std::error::Error>> {
    let stored = session::load()?;
    let Some(service_url) = service_url else {
        return Ok(stored);
    };
    let service = DavisService::new(service_url, None)?;
    if stored
        .as_ref()
        .is_some_and(|session| session.service_url == service.base_url())
    {
        return Ok(stored);
    }
    if !std::io::stdin().is_terminal() {
        return Err(format!(
            "no CLI session for {}; run `davis login {}` first",
            service.base_url(),
            service.base_url()
        )
        .into());
    }
    println!("CLI login is required for {}", service.base_url());
    let invite_code = rpassword::prompt_password("Invite code: ")?;
    if invite_code.is_empty() {
        return Err("invite code must not be empty".into());
    }
    let login = service.exchange_invite_code(&invite_code).await?;
    let stored =
        session::Session::new(service.base_url().to_owned(), login.token, login.expires_at);
    session::save(&stored)?;
    println!("Logged in to {}", stored.service_url);
    Ok(Some(stored))
}

fn handle_verify(
    repository: &std::path::Path,
    dataset_id: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = scan_legacy_repository(repository)?;
    let selected: Vec<&Dataset> = if let Some(dataset_id) = dataset_id {
        vec![catalog
            .dataset(dataset_id)
            .ok_or_else(|| format!("dataset was not found: {dataset_id}"))?]
    } else {
        catalog.datasets.iter().collect()
    };
    let report = audit_legacy_datasets(repository, &selected)?;
    println!(
        "Verified {} files ({}) against DVC metadata",
        report.files,
        human_size(report.bytes)
    );
    Ok(())
}

async fn handle_push(request: PushRequest) -> Result<(), Box<dyn std::error::Error>> {
    if request.dataset_id.is_none() && !request.all {
        return Err("provide a dataset ID or use --all".into());
    }
    let catalog = scan_legacy_repository(&request.repository)?;
    let dataset_ids: Vec<String> = if request.all {
        catalog
            .datasets
            .iter()
            .map(|dataset| dataset.id.clone())
            .collect()
    } else {
        let dataset_id = request
            .dataset_id
            .as_deref()
            .ok_or("provide a dataset ID or use --all")?;
        if catalog.dataset(dataset_id).is_none() {
            return Err(format!("dataset was not found: {dataset_id}").into());
        }
        vec![dataset_id.to_owned()]
    };
    let manifest_directory = resolve(&request.repository, &request.manifest_directory);
    let mut manifests = Vec::with_capacity(dataset_ids.len());
    for dataset_id in &dataset_ids {
        let manifest = read_manifest(&manifest_directory.join(format!("{dataset_id}.yaml")))?;
        if manifest.dataset.id != *dataset_id {
            return Err(format!(
                "manifest dataset ID mismatch: expected {dataset_id}, found {}",
                manifest.dataset.id
            )
            .into());
        }
        manifests.push(manifest);
    }
    let remote_store = open_remote(&request.repository, &request.config, &request.remote)?;
    let local_store = LocalObjectStore::new(resolve(&request.repository, &request.store));
    if request.all {
        println!("Datasets: {}", manifests.len());
    } else {
        println!("Dataset: {}", dataset_ids[0]);
    }
    println!("Remote: {}", request.remote);
    let check_bar = transfer_progress_bar("Check");
    let plan_result = remote_store
        .plan_upload_manifests_with_progress(&local_store, &manifests, |progress| {
            update_transfer_progress(&check_bar, "Check", progress);
        })
        .await;
    let plan = match plan_result {
        Ok(plan) => {
            check_bar.finish_with_message("Check complete");
            plan
        }
        Err(error) => {
            check_bar.abandon_with_message("Check failed");
            return Err(error.into());
        }
    };
    println!("Missing objects: {}", plan.missing);
    println!("Existing objects: {}", plan.existing);
    println!("Upload size: {}", human_size(plan.missing_bytes));
    if request.dry_run {
        println!("Dry run: no objects were uploaded");
    } else if plan.missing == 0 {
        println!("Nothing to upload");
    } else {
        let upload_bar = transfer_progress_bar("Push");
        let upload_result = remote_store
            .upload_plan_with_progress(&local_store, &plan, |progress| {
                update_transfer_progress(&upload_bar, "Push", progress);
            })
            .await;
        let report = match upload_result {
            Ok(report) => {
                upload_bar.finish_with_message("Push complete");
                report
            }
            Err(error) => {
                upload_bar.abandon_with_message("Push failed");
                return Err(error.into());
            }
        };
        println!("Uploaded objects: {}", report.uploaded);
        println!("Skipped objects: {}", report.skipped);
    }
    let revision = publish_catalog(
        &request.repository,
        &catalog,
        &remote_store,
        request.dry_run,
    )
    .await?;
    if request.dry_run {
        println!("Catalog revision: {revision} (dry run, not published)");
    } else {
        println!("Catalog revision: {revision}");
        println!("Catalog published: yes");
    }
    Ok(())
}

async fn publish_catalog(
    repository: &std::path::Path,
    catalog: &davis_core::Catalog,
    remote: &ObjectStorage,
    dry_run: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let index = build_catalog_index(repository, catalog)?;
    let temporary = tempfile::tempdir()?;
    write_catalog_index(temporary.path(), &index)?;
    let index_bytes = std::fs::read(temporary.path().join("index.json"))?;
    let revision = blake3::hash(&index_bytes).to_hex().to_string();
    if dry_run {
        return Ok(revision);
    }

    let manifest_root = repository.join(".davis/datasets");
    let manifests = catalog
        .datasets
        .iter()
        .map(|dataset| read_manifest(&manifest_root.join(format!("{}.yaml", dataset.id))))
        .collect::<Result<Vec<_>, _>>()?;
    let coverage = remote.remote_coverage(&manifests).await?;
    if coverage.missing > 0 {
        return Err(format!(
            "catalog was not published because {} referenced remote objects are missing",
            coverage.missing
        )
        .into());
    }

    for name in CATALOG_DOCUMENTS {
        let contents = std::fs::read(temporary.path().join(name))?;
        let key = format!("catalog/revisions/{revision}/{name}");
        remote.write_document(&key, contents).await?;
    }
    let pointer = serde_json::to_vec(&serde_json::json!({
        "version": 1,
        "revision": revision.clone(),
    }))?;
    remote
        .write_document("catalog/current.json", pointer)
        .await?;
    Ok(revision)
}

fn open_remote(
    repository: &std::path::Path,
    config: &std::path::Path,
    remote: &str,
) -> Result<ObjectStorage, Box<dyn std::error::Error>> {
    let config_path = resolve(repository, config);
    let configuration = read_storage_configuration(&config_path)?;
    let remote_config = configuration
        .remote
        .get(remote)
        .ok_or_else(|| format!("remote was not found in configuration: {remote}"))?;
    let resolved_remote = resolve_filesystem_remote(repository, remote_config);
    let credentials = credentials_for(&resolved_remote)?;
    Ok(ObjectStorage::from_config(
        &resolved_remote,
        credentials.as_ref(),
    )?)
}

fn resolve_filesystem_remote(repository: &std::path::Path, remote: &RemoteConfig) -> RemoteConfig {
    match remote {
        RemoteConfig::Fs { root } => RemoteConfig::Fs {
            root: resolve(repository, root),
        },
        RemoteConfig::S3 {
            bucket,
            endpoint,
            region,
            root,
        } => RemoteConfig::S3 {
            bucket: bucket.clone(),
            endpoint: endpoint.clone(),
            region: region.clone(),
            root: root.clone(),
        },
    }
}

fn credentials_for(remote: &RemoteConfig) -> Result<Option<S3Credentials>, StorageError> {
    if !matches!(remote, RemoteConfig::S3 { .. }) {
        return Ok(None);
    }
    let access_key_id =
        std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| StorageError::MissingS3Credentials)?;
    let secret_access_key =
        std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| StorageError::MissingS3Credentials)?;
    Ok(Some(S3Credentials {
        access_key_id,
        secret_access_key,
    }))
}

fn resolve(repository: &std::path::Path, path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repository.join(path)
    }
}

fn print_dataset_list(datasets: &[Dataset]) {
    println!("DATASET\tFILES\tSCHEMAS\tSIZE");
    for dataset in datasets {
        println!(
            "{}\t{}\t{}/{}\t{}",
            dataset.id,
            dataset.files.len(),
            dataset.schema_ready_count(),
            dataset.files.len(),
            human_size(dataset.total_size())
        );
    }
}

fn print_dataset_info(dataset: &Dataset) {
    println!("Dataset: {}", dataset.id);
    println!("Root: {}", dataset.root);
    println!("Files: {}", dataset.files.len());
    println!("Size: {}", human_size(dataset.total_size()));
    println!();
    println!("STATUS\tSIZE\tFILE");
    for file in &dataset.files {
        let status = match file.schema_status {
            SchemaStatus::Ready => "schema-ready",
            SchemaStatus::Missing => "schema-missing",
            SchemaStatus::Invalid => "schema-invalid",
        };
        println!("{status}\t{}\t{}", human_size(file.size), file.path);
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut scaled = bytes;
    let mut unit = 0;
    let mut divisor = 1_u64;
    while scaled >= 1024 && unit < UNITS.len() - 1 {
        scaled /= 1024;
        divisor *= 1024;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        let fraction = (bytes % divisor) * 100 / divisor;
        format!("{}.{fraction:02} {}", bytes / divisor, UNITS[unit])
    }
}

fn transfer_progress_bar(label: &'static str) -> ProgressBar {
    let progress_bar = if std::io::stderr().is_terminal() {
        ProgressBar::new(0)
    } else {
        ProgressBar::hidden()
    };
    let style = ProgressStyle::with_template(
        "{spinner:.green} {msg:20} [{bar:36.cyan/blue}] {bytes}/{total_bytes} ({percent}%) {eta}",
    )
    .expect("the built-in progress template must be valid")
    .progress_chars("=>-");
    progress_bar.set_style(style);
    progress_bar.set_message(label);
    progress_bar.enable_steady_tick(Duration::from_millis(100));
    progress_bar
}

fn update_transfer_progress(progress_bar: &ProgressBar, label: &str, progress: TransferProgress) {
    progress_bar.set_length(progress.total_bytes);
    progress_bar.set_position(progress.completed_bytes.min(progress.total_bytes));
    progress_bar.set_message(format!(
        "{label} {}/{}",
        progress.completed_objects, progress.total_objects
    ));
}
