mod component;
mod component_pack;
mod component_registry;
mod git_workflow;
mod model;
mod remote;
mod session;
mod software;
mod update;

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use davis_catalog::{
    audit_datasets, build_catalog_index, ingest_dataset, read_file_schema, refresh_dataset,
    scan_repository, write_catalog_index, RefreshOptions,
};
use davis_core::{
    current_local_date, read_manifest, write_manifest, Dataset, LocalObjectStore, LocalizedText,
    ObjectRef, SchemaStatus,
};
use davis_document::{render_schema_pdf, write_pdf_if_changed, Language};
use davis_storage::{
    read_storage_configuration, ObjectStorage, RemoteConfig, S3Credentials, StorageError,
    TransferProgress,
};
use git_workflow::{
    commit_and_push_operator_changes, git_output, verify_operator_worktree,
    verify_publish_git_state,
};
use indicatif::{ProgressBar, ProgressStyle};
use remote::{DavisService, RemoteError};

#[derive(Debug, Parser)]
#[command(name = "davis", version, about = "Davis data catalog client")]
struct Cli {
    /// Davis repository or local analysis workspace to read.
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
    /// Check for and install a newer Davis release.
    Update {
        /// Install the update without asking for confirmation.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Authenticate and manage organizer access.
    Operator {
        #[command(subcommand)]
        command: OperatorCommand,
    },
    /// Validate and run local analysis components.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Install optional Davis applications and components.
    Install {
        #[command(subcommand)]
        command: InstallCommand,
    },
    /// Launch the installed Davis desktop application.
    Desktop {
        /// Select an exact installed desktop version.
        #[arg(long)]
        version: Option<String>,
    },
    /// List software and components managed by this Davis installation.
    Installed {
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Inspect and manage installed analysis components.
    Component {
        #[command(subcommand)]
        command: ComponentCommand,
    },
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
    /// Hash local files and ingest them into the Davis content-addressed store.
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
    /// Regenerate schema PDFs without uploading or changing Git history (maintainers only).
    Documents {
        /// One dataset whose documents are regenerated.
        dataset_id: Option<String>,
        /// Regenerate documents for every dataset.
        #[arg(long, conflicts_with = "dataset_id")]
        all: bool,
    },
    /// Materialize a dataset from its Manifest and local object storage.
    Get {
        dataset_id: String,
        /// Davis Web service URL. When no matching CLI session exists, prompt for an invite code.
        #[arg(long)]
        service_url: Option<String>,
        /// Materialize only these file IDs or directory prefixes. Repeat for multiple selections.
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
        /// Replace files that already exist without prompting.
        #[arg(long)]
        force: bool,
        /// Do not save the schema.yaml companion files.
        #[arg(long)]
        no_schema: bool,
        /// Save Japanese PDF documentation when available.
        #[arg(long)]
        pdf_ja: bool,
        /// Save English PDF documentation when available.
        #[arg(long)]
        pdf_en: bool,
        /// Storage configuration used to fill missing cache objects.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Named remote from the configuration.
        #[arg(long, default_value = "default")]
        remote: String,
    },
    /// Synchronize a dataset to the current Manifest, or retrieve it for the first time.
    Pull {
        #[command(flatten)]
        args: PullArgs,
    },
    /// Verify local files against the BLAKE3 IDs recorded by Davis.
    Verify {
        /// Verify only one dataset. All datasets are checked when omitted.
        dataset_id: Option<String>,
    },
    /// Prepare, upload, commit, and push dataset updates from a personal branch.
    Push {
        /// One dataset to upload. Every dataset is selected when omitted.
        dataset_id: Option<String>,
        /// Explicitly upload every dataset (compatibility alias for omitting the ID).
        #[arg(long, conflicts_with = "dataset_id")]
        all: bool,
        /// Local content-addressed cache.
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
        /// Re-read every selected source file instead of reusing unchanged Manifest entries.
        #[arg(long)]
        rehash: bool,
        /// Git commit message. Defaults to `data: update <dataset>`.
        #[arg(short, long)]
        message: Option<String>,
    },
    /// Publish the reviewed `CatalogIndex` from the current main branch.
    Publish {
        /// Versioned storage configuration used without an operator session.
        #[arg(long, default_value = ".davis/config.toml")]
        config: PathBuf,
        /// Named remote from the configuration.
        #[arg(long, default_value = "default")]
        remote: String,
    },
}

#[derive(Debug, Args)]
struct PullArgs {
    /// One dataset to synchronize. Omit to synchronize every dataset.
    dataset_id: Option<String>,
    /// Davis Web service URL. When no matching CLI session exists, prompt for an invite code.
    #[arg(long)]
    service_url: Option<String>,
    /// Local object storage root.
    #[arg(long, default_value = ".davis/cache")]
    store: PathBuf,
    /// Directory containing `DatasetManifest` YAML files.
    #[arg(long, default_value = ".davis/datasets")]
    manifest_directory: PathBuf,
    /// Output root. Dataset paths are recreated below this directory.
    #[arg(short, long, default_value = ".")]
    out: PathBuf,
    /// Do not save the schema.yaml companion files.
    #[arg(long)]
    no_schema: bool,
    /// Save Japanese PDF documentation when available.
    #[arg(long)]
    pdf_ja: bool,
    /// Save English PDF documentation when available.
    #[arg(long)]
    pdf_en: bool,
    /// Storage configuration used to fill missing cache objects.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Named remote from the configuration.
    #[arg(long, default_value = "default")]
    remote: String,
}

#[derive(Debug, Subcommand)]
enum OperatorCommand {
    /// Exchange the shared operator code for a short-lived upload session.
    Login {
        /// Davis Web service URL.
        service_url: String,
        /// Read the operator code from standard input instead of prompting.
        #[arg(long)]
        operator_code_stdin: bool,
    },
    /// Show whether the stored operator session is still valid.
    Status,
    /// Remove the locally stored operator session.
    Logout,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Inspect the encoding, delimiter, and inferred columns of a local CSV file.
    Inspect {
        path: PathBuf,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Validate an analysis plan and resolve its component.
    Validate {
        plan: PathBuf,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Resolve inputs and print the exact request without running the component.
    Plan {
        plan: PathBuf,
        /// Root where the eventual run directory will be created.
        #[arg(long, default_value = "davis-runs")]
        run_root: PathBuf,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Resolve inputs and execute a local component.
    Run {
        plan: PathBuf,
        /// Directory below which immutable run records are written.
        #[arg(long, default_value = "davis-runs")]
        run_root: PathBuf,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum InstallCommand {
    /// Install an official or local component package.
    Component {
        /// Official component name, ID, or local package directory.
        source: String,
        /// Select an exact official component version.
        #[arg(long)]
        version: Option<String>,
        /// Override the official component registry URL.
        #[arg(long)]
        registry: Option<String>,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Install the Davis desktop application for this computer.
    #[command(alias = "app")]
    Desktop {
        /// Select an exact desktop version.
        #[arg(long)]
        version: Option<String>,
        /// Override the official software registry URL.
        #[arg(long)]
        registry: Option<String>,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ComponentCommand {
    /// Create a minimal self-contained component package.
    Scaffold {
        /// New component directory. It must not already exist.
        path: PathBuf,
        /// Stable component ID, for example `example/my-component`.
        #[arg(long)]
        id: String,
        /// Human-readable name. Defaults to the final ID segment.
        #[arg(long)]
        name: Option<String>,
        /// Component role.
        #[arg(long, value_enum, default_value_t = ScaffoldKind::Model)]
        kind: ScaffoldKind,
        /// One runtime command argument. Repeat this option for every argument.
        #[arg(long = "command", required = true, allow_hyphen_values = true)]
        runtime_command: Vec<String>,
        /// Supported operation. Repeat to declare multiple operations.
        #[arg(long = "operation")]
        operations: Vec<String>,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Validate a component package without installing it.
    Validate {
        path: PathBuf,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// List installed model components.
    List {
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show one installed model component.
    Inspect {
        id: String,
        /// Select an exact installed version.
        #[arg(long)]
        version: Option<String>,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Remove one installed model component.
    Remove {
        id: String,
        /// Select an exact installed version.
        #[arg(long)]
        version: Option<String>,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Build a deterministic release bundle and registry entry.
    Pack {
        path: PathBuf,
        /// Directory where the bundle and entry JSON are written.
        #[arg(long)]
        out: PathBuf,
        /// Short official install name. Defaults to the final ID segment.
        #[arg(long)]
        name: Option<String>,
        /// Compatible Davis `SemVer` requirement. Defaults to the manifest declaration.
        #[arg(long)]
        requires_davis: Option<String>,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
    /// Combine component entry files into a versioned registry.
    Registry {
        /// Entry JSON files emitted by `davis component pack`.
        entries: Vec<PathBuf>,
        /// Registry JSON destination.
        #[arg(long)]
        out: PathBuf,
        /// Print structured JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ScaffoldKind {
    Model,
    Transform,
    Visualize,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let checks_update_explicitly = matches!(cli.command, Command::Update { .. });
    if let Err(error) = run(cli).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    if !checks_update_explicitly {
        update::check_automatically().await;
    }
}

// The top-level dispatch intentionally keeps every public CLI command visible in one match.
#[allow(clippy::too_many_lines)]
async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Login {
            service_url,
            invite_code_stdin,
        } => handle_login(&service_url, invite_code_stdin).await?,
        Command::Logout => handle_logout()?,
        Command::Update { yes } => update::check_explicitly(yes).await?,
        Command::Operator { command } => handle_operator(command).await?,
        Command::Model { command } => model::handle(&cli.repository, command)?,
        Command::Install { command } => handle_install(command).await?,
        Command::Desktop { version } => software::launch_desktop(version.as_deref())?,
        Command::Installed { json } => software::print_installed(json)?,
        Command::Component { command } => component::handle_component(command)?,
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
        Command::Documents { dataset_id, all } => {
            handle_documents(&cli.repository, dataset_id.as_deref(), all)?;
        }
        Command::Get {
            dataset_id,
            service_url,
            files,
            store,
            manifest_directory,
            out,
            force,
            no_schema,
            pdf_ja,
            pdf_en,
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
                documents: DocumentSelection {
                    schema: !no_schema,
                    pdf_ja,
                    pdf_en,
                },
                config,
                remote,
            })
            .await?;
        }
        Command::Pull { args } => handle_pull(PullRequest::new(cli.repository, args)).await?,
        Command::Verify { dataset_id } => handle_verify(&cli.repository, dataset_id.as_deref())?,
        Command::Push {
            dataset_id,
            all,
            store,
            manifest_directory,
            config,
            remote,
            dry_run,
            rehash,
            message,
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
                rehash,
                message,
            })
            .await?;
        }
        Command::Publish { config, remote } => {
            handle_publish(&cli.repository, &config, &remote).await?;
        }
    }

    Ok(())
}

async fn handle_install(command: InstallCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        InstallCommand::Component {
            source,
            version,
            registry,
            json,
        } => component::handle_install(source, version, registry, json).await?,
        InstallCommand::Desktop {
            version,
            registry,
            json,
        } => software::install_desktop(version.as_deref(), registry.as_deref(), json).await?,
    }
    Ok(())
}

async fn handle_operator(command: OperatorCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        OperatorCommand::Login {
            service_url,
            operator_code_stdin,
        } => handle_operator_login(&service_url, operator_code_stdin).await,
        OperatorCommand::Status => handle_operator_status().await,
        OperatorCommand::Logout => {
            if session::clear_operator()? {
                println!("Operator session removed");
            } else {
                println!("No stored operator session");
            }
            Ok(())
        }
    }
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
    rehash: bool,
    message: Option<String>,
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
    documents: DocumentSelection,
    config: Option<PathBuf>,
    remote: String,
}

struct PullRequest {
    repository: PathBuf,
    dataset_id: Option<String>,
    service_url: Option<String>,
    store: PathBuf,
    manifest_directory: PathBuf,
    out: PathBuf,
    documents: DocumentSelection,
    config: Option<PathBuf>,
    remote: String,
}

impl PullRequest {
    fn new(repository: PathBuf, args: PullArgs) -> Self {
        Self {
            repository,
            dataset_id: args.dataset_id,
            service_url: args.service_url,
            store: args.store,
            manifest_directory: args.manifest_directory,
            out: args.out,
            documents: DocumentSelection {
                schema: !args.no_schema,
                pdf_ja: args.pdf_ja,
                pdf_en: args.pdf_en,
            },
            config: args.config,
            remote: args.remote,
        }
    }
}

#[derive(Clone, Copy)]
struct DocumentSelection {
    schema: bool,
    pdf_ja: bool,
    pdf_en: bool,
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

async fn handle_operator_login(
    service_url: &str,
    operator_code_stdin: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let operator_code = if operator_code_stdin {
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input)?;
        input.trim_end_matches(['\r', '\n']).to_owned()
    } else if std::io::stdin().is_terminal() {
        rpassword::prompt_password("Operator code: ")?
    } else {
        return Err("standard input is not a terminal; use --operator-code-stdin".into());
    };
    if operator_code.is_empty() {
        return Err("operator code must not be empty".into());
    }
    let service = DavisService::new(service_url, None)?;
    let login = service.exchange_operator_code(&operator_code).await?;
    let stored =
        session::Session::new(service.base_url().to_owned(), login.token, login.expires_at);
    let path = session::save_operator(&stored)?;
    println!("Operator login: {}", stored.service_url);
    println!("Session expires: {}", stored.expires_at);
    println!("Session: {}", path.display());
    Ok(())
}

async fn handle_operator_status() -> Result<(), Box<dyn std::error::Error>> {
    let stored = session::load_operator()?.ok_or("no stored operator session")?;
    let status = DavisService::new(&stored.service_url, Some(stored.token.clone()))?
        .operator_session_status()
        .await?;
    println!("Operator session: active");
    println!("Service: {}", stored.service_url);
    println!("Session expires: {}", status.expires_at);
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
        scan_repository(repository)?
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
        scan_repository(repository)?
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
    let catalog = scan_repository(repository)?;
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
    let catalog = scan_repository(repository)?;
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
        let report = ingest_dataset(repository, dataset, &object_store)?;
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

async fn handle_get(mut request: GetRequest) -> Result<(), Box<dyn std::error::Error>> {
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
    let manifest = select_download_files(&manifest, &request.files)?;
    let output = resolve(&request.repository, &request.out);
    if !confirm_get_overwrite(&mut request, &manifest, &output)? {
        println!("Get cancelled; existing files were not changed");
        return Ok(());
    }
    print_download_terms(&request, &manifest, stored_session.as_ref()).await?;
    if !request.documents.schema {
        eprintln!("Warning: schema.yaml will not be saved; future Davis formatting and modeling workflows may require it.");
    }
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
    } else if let Some(stored) = &stored_session {
        let service = DavisService::new(&stored.service_url, Some(stored.token.clone()))?;
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
    object_store.materialize(&manifest, &output, request.force)?;
    materialize_companion_documents(&request, &manifest, stored_session.as_ref(), &output).await?;
    println!(
        "Materialized {} files under {}",
        manifest.files.len(),
        output.join(&manifest.dataset.root).display()
    );
    Ok(())
}

fn confirm_get_overwrite(
    request: &mut GetRequest,
    manifest: &davis_core::DatasetManifest,
    output: &std::path::Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    if request.force {
        return Ok(true);
    }
    let existing_destinations = existing_manifest_destinations(manifest, output);
    if existing_destinations.is_empty() {
        return Ok(true);
    }
    eprintln!(
        "{} selected file(s) already exist:",
        existing_destinations.len()
    );
    for destination in existing_destinations.iter().take(5) {
        eprintln!("- {}", destination.display());
    }
    if existing_destinations.len() > 5 {
        eprintln!("- ... and {} more", existing_destinations.len() - 5);
    }
    if !std::io::stdin().is_terminal() {
        return Err("cannot ask whether to replace existing files because standard input is not a terminal; rerun with --force to replace them".into());
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let replace = prompt_to_replace(
        &mut stdin.lock(),
        &mut stdout.lock(),
        existing_destinations.len(),
    )?;
    request.force = replace;
    Ok(replace)
}

fn existing_manifest_destinations(
    manifest: &davis_core::DatasetManifest,
    output: &std::path::Path,
) -> Vec<PathBuf> {
    manifest
        .files
        .iter()
        .map(|file| output.join(&manifest.dataset.root).join(&file.path))
        .filter(|destination| destination.exists())
        .collect()
}

fn prompt_to_replace(
    input: &mut impl BufRead,
    output: &mut impl Write,
    existing_count: usize,
) -> std::io::Result<bool> {
    loop {
        write!(
            output,
            "Replace all {existing_count} existing file(s)? [y/N] "
        )?;
        output.flush()?;
        let mut answer = String::new();
        if input.read_line(&mut answer)? == 0 {
            return Ok(false);
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "" | "n" | "no" => return Ok(false),
            _ => writeln!(output, "Please answer y or n.")?,
        }
    }
}

async fn print_download_terms(
    request: &GetRequest,
    manifest: &davis_core::DatasetManifest,
    stored_session: Option<&session::Session>,
) -> Result<(), Box<dyn std::error::Error>> {
    let selected_ids = manifest
        .files
        .iter()
        .map(|file| file.id.as_str())
        .collect::<Vec<_>>();
    let terms = if let Some(stored) = stored_session {
        DavisService::new(&stored.service_url, Some(stored.token.clone()))?
            .indexed_files(&manifest.dataset.id)
            .await?
            .into_iter()
            .filter(|file| selected_ids.contains(&file.file_id.as_str()))
            .filter_map(|file| file.license)
            .collect::<Vec<_>>()
    } else {
        scan_repository(&request.repository)?
            .dataset(&manifest.dataset.id)
            .ok_or_else(|| format!("dataset was not found: {}", manifest.dataset.id))?
            .files
            .iter()
            .filter(|file| selected_ids.contains(&file.id.as_str()))
            .filter_map(|file| file.schema.as_ref()?.license.clone())
            .collect::<Vec<_>>()
    };
    let unique_terms = terms.into_iter().fold(Vec::new(), |mut unique, terms| {
        if !unique.contains(&terms) {
            unique.push(terms);
        }
        unique
    });
    if unique_terms.is_empty() {
        println!("Terms of use: not specified; confirm with the organizers before use");
    } else {
        println!("Terms of use:");
        for LocalizedText { ja, en } in unique_terms {
            println!("- {ja}");
            if en != ja {
                println!("  {en}");
            }
        }
    }
    Ok(())
}

async fn handle_pull(request: PullRequest) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = if request.config.is_none() {
        if let Some(stored) = get_session(request.service_url.as_deref()).await? {
            DavisService::new(&stored.service_url, Some(stored.token))?
                .catalog()
                .await?
        } else {
            scan_repository(&request.repository)?
        }
    } else {
        scan_repository(&request.repository)?
    };
    let dataset_ids = select_dataset_ids(&catalog, request.dataset_id.as_deref())?;
    if request.dataset_id.is_none() {
        println!("Datasets: {}", dataset_ids.len());
    }
    for (index, dataset_id) in dataset_ids.into_iter().enumerate() {
        if request.dataset_id.is_none() {
            println!(
                "Dataset {}/{}: {dataset_id}",
                index + 1,
                catalog.datasets.len()
            );
        }
        handle_get(GetRequest {
            repository: request.repository.clone(),
            dataset_id,
            service_url: request.service_url.clone(),
            files: Vec::new(),
            store: request.store.clone(),
            manifest_directory: request.manifest_directory.clone(),
            out: request.out.clone(),
            force: true,
            documents: request.documents,
            config: request.config.clone(),
            remote: request.remote.clone(),
        })
        .await?;
    }
    Ok(())
}

fn select_dataset_ids(
    catalog: &davis_core::Catalog,
    dataset_id: Option<&str>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if let Some(dataset_id) = dataset_id {
        if catalog.dataset(dataset_id).is_none() {
            return Err(format!("dataset was not found: {dataset_id}").into());
        }
        Ok(vec![dataset_id.to_owned()])
    } else {
        Ok(catalog
            .datasets
            .iter()
            .map(|dataset| dataset.id.clone())
            .collect())
    }
}

fn select_download_files(
    manifest: &davis_core::DatasetManifest,
    requested: &[String],
) -> Result<davis_core::DatasetManifest, Box<dyn std::error::Error>> {
    let is_document = |id: &str| {
        id.ends_with(".schema.yaml") || id.ends_with(".ja.pdf") || id.ends_with(".en.pdf")
    };
    let primary_ids = if requested.is_empty() {
        manifest
            .files
            .iter()
            .filter(|file| !is_document(&file.id))
            .map(|file| file.id.clone())
            .collect::<Vec<_>>()
    } else {
        requested.to_vec()
    };
    let selected = if requested.is_empty() {
        primary_ids
    } else {
        let mut selected = Vec::new();
        for selector in requested {
            let directory_prefix = format!("{}/", selector.trim_end_matches('/'));
            let matches = manifest
                .files
                .iter()
                .filter(|file| {
                    !is_document(&file.id)
                        && (file.id == *selector
                            || file.path == *selector
                            || file.id.starts_with(&directory_prefix)
                            || file.path.starts_with(&directory_prefix))
                })
                .map(|file| file.id.clone())
                .collect::<Vec<_>>();
            if matches.is_empty() {
                return Err(format!("file or directory was not found: {selector}").into());
            }
            for file_id in matches {
                if !selected.contains(&file_id) {
                    selected.push(file_id);
                }
            }
        }
        selected
    };
    manifest.select_files(&selected).map_err(Into::into)
}

async fn materialize_companion_documents(
    request: &GetRequest,
    manifest: &davis_core::DatasetManifest,
    stored_session: Option<&session::Session>,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if !request.documents.schema && !request.documents.pdf_ja && !request.documents.pdf_en {
        return Ok(());
    }
    let destination_root = output.join(&manifest.dataset.root);
    if let Some(stored) = stored_session {
        let service = DavisService::new(&stored.service_url, Some(stored.token.clone()))?;
        let indexed = service.indexed_files(&manifest.dataset.id).await?;
        for primary in &manifest.files {
            let Some(file) = indexed.iter().find(|file| file.file_id == primary.id) else {
                continue;
            };
            if request.documents.schema {
                if let (Some(document), Some(contents)) =
                    (&file.documents.schema, file.raw_schema.as_deref())
                {
                    write_companion(
                        &destination_root.join(&document.id),
                        contents.as_bytes(),
                        request.force,
                    )?;
                }
            }
            for document in [
                request
                    .documents
                    .pdf_ja
                    .then_some(file.documents.pdf_ja.as_ref())
                    .flatten(),
                request
                    .documents
                    .pdf_en
                    .then_some(file.documents.pdf_en.as_ref())
                    .flatten(),
            ]
            .into_iter()
            .flatten()
            {
                let response = reqwest::get(&document.url).await?.error_for_status()?;
                let bytes = response.bytes().await?;
                write_companion(&destination_root.join(&document.id), &bytes, request.force)?;
            }
        }
    } else {
        for primary in &manifest.files {
            for (enabled, suffix) in [
                (request.documents.schema, ".schema.yaml"),
                (request.documents.pdf_ja, ".ja.pdf"),
                (request.documents.pdf_en, ".en.pdf"),
            ] {
                if !enabled {
                    continue;
                }
                let id = format!("{}{suffix}", primary.id);
                let source = request.repository.join(&manifest.dataset.root).join(&id);
                if source.is_file() {
                    write_companion(
                        &destination_root.join(id),
                        &std::fs::read(source)?,
                        request.force,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn write_companion(
    destination: &std::path::Path,
    contents: &[u8],
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if destination.exists() && !force {
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(destination, contents)?;
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
    let catalog = scan_repository(repository)?;
    let selected: Vec<&Dataset> = if let Some(dataset_id) = dataset_id {
        vec![catalog
            .dataset(dataset_id)
            .ok_or_else(|| format!("dataset was not found: {dataset_id}"))?]
    } else {
        catalog.datasets.iter().collect()
    };
    let report = audit_datasets(repository, &selected)?;
    println!(
        "Verified {} files ({}) against Davis BLAKE3 metadata",
        report.files,
        human_size(report.bytes)
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
async fn handle_push(request: PushRequest) -> Result<(), Box<dyn std::error::Error>> {
    if request.rehash {
        println!("Rehashing every selected source file");
    }
    let catalog = scan_repository(&request.repository)?;
    let dataset_ids = select_dataset_ids(&catalog, request.dataset_id.as_deref())?;
    let operator_session = session::load_operator()?;
    if operator_session.is_some() {
        verify_operator_worktree(&request.repository, &dataset_ids)?;
    }
    let manifest_directory = resolve(&request.repository, &request.manifest_directory);
    let local_store = LocalObjectStore::new(resolve(&request.repository, &request.store));
    let manifests = prepare_push_manifests(
        &request.repository,
        &catalog,
        &dataset_ids,
        &manifest_directory,
        &local_store,
        request.rehash,
        !request.dry_run,
    )?;
    if let Some(operator_session) = operator_session {
        let operator_session = ensure_operator_session(operator_session).await?;
        handle_operator_push(&request, &manifests, &local_store, operator_session).await?;
        if !request.dry_run {
            prepare_changed_pdfs(&request.repository, &manifests)?;
            commit_and_push_operator_changes(
                &request.repository,
                &dataset_ids,
                request.dataset_id.as_deref(),
                request.message.as_deref(),
            )?;
        }
        return Ok(());
    }
    let remote_store = open_remote(&request.repository, &request.config, &request.remote)?;
    if request.all || request.dataset_id.is_none() {
        println!("Datasets: {}", manifests.len());
    } else {
        println!("Dataset: {}", dataset_ids[0]);
    }
    println!("Remote: {}", request.remote);
    let check_bar = transfer_progress_bar("Check");
    let plan_result = if request.dry_run {
        remote_store
            .plan_remote_upload_manifests_with_progress(&manifests, |progress| {
                update_transfer_progress(&check_bar, "Check", progress);
            })
            .await
    } else {
        remote_store
            .plan_upload_manifests_with_progress(&local_store, &manifests, |progress| {
                update_transfer_progress(&check_bar, "Check", progress);
            })
            .await
    };
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
    if !request.dry_run {
        println!("Objects synchronized: yes");
        println!("Catalog published: no (run `davis publish` from the latest main branch)");
    }
    Ok(())
}

fn prepare_pdfs(
    repository: &std::path::Path,
    manifests: &[davis_core::DatasetManifest],
    changed_inputs: Option<&HashSet<String>>,
    write_changes: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut generated = 0_usize;
    let mut changed = 0_usize;
    for manifest in manifests {
        for file in &manifest.files {
            let Some(schema_path) = &file.schema_path else {
                continue;
            };
            let base = repository.join(&manifest.dataset.root).join(&file.path);
            let manifest_path = format!(".davis/datasets/{}.yaml", manifest.dataset.id);
            let pdf_missing = [Language::Japanese, Language::English]
                .iter()
                .any(|language| {
                    !std::path::PathBuf::from(format!("{}{}", base.display(), language.suffix()))
                        .is_file()
                });
            if changed_inputs.is_some_and(|paths| {
                !paths.contains(&manifest_path) && !paths.contains(schema_path) && !pdf_missing
            }) {
                continue;
            }
            let schema = read_file_schema(&repository.join(schema_path))?;
            for language in [Language::Japanese, Language::English] {
                let contents = render_schema_pdf(&schema, &file.object, language)?;
                let destination =
                    std::path::PathBuf::from(format!("{}{}", base.display(), language.suffix()));
                let differs =
                    std::fs::read(&destination).map_or(true, |current| current != contents);
                if differs {
                    changed += 1;
                    if write_changes {
                        write_pdf_if_changed(&destination, &contents)?;
                    }
                }
                generated += 1;
            }
        }
    }
    println!("PDF documents checked: {generated}");
    if write_changes {
        println!("PDF documents updated: {changed}");
    } else {
        println!("PDF documents to update: {changed}");
    }
    Ok(())
}

fn prepare_changed_pdfs(
    repository: &std::path::Path,
    manifests: &[davis_core::DatasetManifest],
) -> Result<(), Box<dyn std::error::Error>> {
    let tracked = git_output(
        repository,
        &["diff", "--name-only", "-z", "--no-renames", "HEAD", "--"],
    )?;
    let untracked = git_output(
        repository,
        &["ls-files", "--others", "--exclude-standard", "-z", "--"],
    )?;
    let changed_inputs = tracked
        .split('\0')
        .chain(untracked.split('\0'))
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect();
    prepare_pdfs(repository, manifests, Some(&changed_inputs), true)
}

fn handle_documents(
    repository: &std::path::Path,
    dataset_id: Option<&str>,
    all: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if dataset_id.is_none() && !all {
        return Err("provide a dataset ID or explicitly use --all".into());
    }
    let catalog = scan_repository(repository)?;
    let dataset_ids = select_dataset_ids(&catalog, dataset_id)?;
    let manifests = dataset_ids
        .iter()
        .map(|id| {
            read_manifest(
                &repository
                    .join(".davis/datasets")
                    .join(format!("{id}.yaml")),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    prepare_pdfs(repository, &manifests, None, true)
}

async fn ensure_operator_session(
    stored: session::Session,
) -> Result<session::Session, Box<dyn std::error::Error>> {
    let service = DavisService::new(&stored.service_url, Some(stored.token.clone()))?;
    match service.operator_session_status().await {
        Ok(_) => Ok(stored),
        Err(RemoteError::Api { status, .. }) if status == reqwest::StatusCode::UNAUTHORIZED => {
            if !std::io::stdin().is_terminal() {
                return Err(format!(
                    "operator session expired; run `davis operator login {}`",
                    stored.service_url
                )
                .into());
            }
            println!("Operator session expired; login is required");
            let operator_code = rpassword::prompt_password("Operator code: ")?;
            if operator_code.is_empty() {
                return Err("operator code must not be empty".into());
            }
            let service = DavisService::new(&stored.service_url, None)?;
            let login = service.exchange_operator_code(&operator_code).await?;
            let refreshed =
                session::Session::new(service.base_url().to_owned(), login.token, login.expires_at);
            session::save_operator(&refreshed)?;
            println!("Operator session renewed");
            Ok(refreshed)
        }
        Err(error) => Err(error.into()),
    }
}

async fn handle_operator_push(
    request: &PushRequest,
    manifests: &[davis_core::DatasetManifest],
    local_store: &LocalObjectStore,
    operator_session: session::Session,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = DavisService::new(
        &operator_session.service_url,
        Some(operator_session.token.clone()),
    )?;
    let objects = unique_manifest_objects(manifests)?;
    if request.all || request.dataset_id.is_none() {
        println!("Datasets: {}", manifests.len());
    } else if let Some(dataset_id) = &request.dataset_id {
        println!("Dataset: {dataset_id}");
    }
    println!("Remote: {} (operator session)", service.base_url());
    let upload_bar = transfer_progress_bar("Push");
    let result = service
        .upload_operator_objects(
            local_store,
            &objects,
            request.dry_run,
            |completed_bytes, total_bytes, completed_objects, total_objects| {
                upload_bar.set_length(total_bytes);
                upload_bar.set_position(completed_bytes.min(total_bytes));
                upload_bar.set_message(format!("Push {completed_objects}/{total_objects}"));
            },
        )
        .await;
    let report = match result {
        Ok(report) => {
            upload_bar.finish_with_message(if request.dry_run {
                "Check complete"
            } else {
                "Push complete"
            });
            report
        }
        Err(error) => {
            upload_bar.abandon_with_message("Push failed");
            return Err(error.into());
        }
    };
    println!("Missing objects: {}", report.missing);
    println!("Existing objects: {}", report.existing);
    println!("Upload size: {}", human_size(report.missing_bytes));
    if request.dry_run {
        println!("Dry run: no objects were uploaded");
    } else {
        println!("Uploaded objects: {}", report.uploaded);
    }
    if !request.dry_run {
        println!("Objects synchronized: yes");
        println!("Catalog published: no (run `davis publish` from the latest main branch)");
    }
    Ok(())
}

async fn handle_publish(
    repository: &std::path::Path,
    config: &std::path::Path,
    remote: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    verify_publish_git_state(repository)?;
    let catalog = scan_repository(repository)?;
    let revision = if let Some(operator_session) = session::load_operator()? {
        let operator_session = ensure_operator_session(operator_session).await?;
        let service =
            DavisService::new(&operator_session.service_url, Some(operator_session.token))?;
        let (revision, documents) = build_catalog_publication(repository, &catalog)?;
        service
            .publish_operator_catalog(&revision, &documents)
            .await?;
        revision
    } else {
        let remote_store = open_remote(repository, config, remote)?;
        publish_catalog(repository, &catalog, &remote_store, false).await?
    };
    println!("Catalog revision: {revision}");
    println!("Catalog published: yes");
    Ok(())
}

fn unique_manifest_objects(
    manifests: &[davis_core::DatasetManifest],
) -> Result<Vec<ObjectRef>, Box<dyn std::error::Error>> {
    let mut objects = HashMap::<String, ObjectRef>::new();
    for manifest in manifests {
        manifest.validate()?;
        for file in &manifest.files {
            let key = file.object.oid.to_string();
            if let Some(previous) = objects.get(&key) {
                if previous.size != file.object.size {
                    return Err(format!("conflicting sizes for object {}", file.object.oid).into());
                }
            } else {
                objects.insert(key, file.object.clone());
            }
        }
    }
    let mut objects = objects.into_values().collect::<Vec<_>>();
    objects.sort_by_key(|object| object.oid.to_string());
    Ok(objects)
}

fn prepare_push_manifests(
    repository: &std::path::Path,
    catalog: &davis_core::Catalog,
    dataset_ids: &[String],
    manifest_directory: &std::path::Path,
    local_store: &LocalObjectStore,
    rehash: bool,
    write_changes: bool,
) -> Result<Vec<davis_core::DatasetManifest>, Box<dyn std::error::Error>> {
    let mut manifests = Vec::with_capacity(dataset_ids.len());
    let updated_on = write_changes.then(current_local_date);
    for dataset_id in dataset_ids {
        let manifest_path = manifest_directory.join(format!("{dataset_id}.yaml"));
        let previous = manifest_path
            .is_file()
            .then(|| read_manifest(&manifest_path))
            .transpose()?;
        let dataset = catalog
            .dataset(dataset_id)
            .ok_or_else(|| format!("dataset was not found: {dataset_id}"))?;
        let report = refresh_dataset(
            repository,
            &dataset.id,
            &dataset.root,
            local_store,
            RefreshOptions {
                previous: previous.as_ref(),
                rehash,
                write_objects: write_changes,
                updated_on: updated_on.as_deref(),
            },
        )?;
        if write_changes && previous.as_ref() != Some(&report.manifest) {
            write_manifest(&manifest_path, &report.manifest)?;
        }
        println!(
            "Prepared {dataset_id}: {} hashed, {} unchanged",
            report.added_objects + report.existing_objects,
            report.reused_files
        );
        manifests.push(report.manifest);
    }
    Ok(manifests)
}

async fn publish_catalog(
    repository: &std::path::Path,
    catalog: &davis_core::Catalog,
    remote: &ObjectStorage,
    dry_run: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let (revision, documents) = build_catalog_publication(repository, catalog)?;
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
        let contents = documents
            .get(name)
            .ok_or_else(|| format!("catalog document was not generated: {name}"))?
            .as_bytes()
            .to_vec();
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

fn build_catalog_publication(
    repository: &std::path::Path,
    catalog: &davis_core::Catalog,
) -> Result<(String, HashMap<String, String>), Box<dyn std::error::Error>> {
    let index = build_catalog_index(repository, catalog)?;
    let temporary = tempfile::tempdir()?;
    write_catalog_index(temporary.path(), &index)?;
    let index_bytes = std::fs::read(temporary.path().join("index.json"))?;
    let revision = blake3::hash(&index_bytes).to_hex().to_string();
    let documents = CATALOG_DOCUMENTS
        .iter()
        .map(|name| {
            std::fs::read_to_string(temporary.path().join(name))
                .map(|contents| ((*name).to_owned(), contents))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    Ok((revision, documents))
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

#[cfg(test)]
mod tests {
    use super::{Cli, Command, ComponentCommand, InstallCommand};
    use clap::Parser;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    #[test]
    fn pull_accepts_first_retrieval_and_companion_options() {
        let cli = Cli::try_parse_from([
            "davis",
            "pull",
            "routes/Matsuyama",
            "--pdf-ja",
            "--out",
            "downloads",
        ])
        .expect("pull command should parse");

        match cli.command {
            Command::Pull { args } => {
                assert_eq!(args.dataset_id.as_deref(), Some("routes/Matsuyama"));
                assert!(args.pdf_ja);
                assert!(!args.pdf_en);
                assert!(!args.no_schema);
                assert_eq!(args.out, PathBuf::from("downloads"));
            }
            _ => panic!("expected pull command"),
        }
    }

    #[test]
    fn pull_and_push_select_all_datasets_when_the_id_is_omitted() {
        let pull = Cli::try_parse_from(["davis", "pull"]).expect("pull should allow omission");
        assert!(matches!(
            pull.command,
            Command::Pull { args } if args.dataset_id.is_none()
        ));

        let push = Cli::try_parse_from(["davis", "push"]).expect("push should allow omission");
        assert!(matches!(
            push.command,
            Command::Push {
                dataset_id: None,
                all: false,
                ..
            }
        ));

        let catalog = davis_core::Catalog {
            datasets: vec![
                davis_core::Dataset {
                    id: "first/data".into(),
                    root: "data/first/data".into(),
                    files: Vec::new(),
                },
                davis_core::Dataset {
                    id: "second/data".into(),
                    root: "data/second/data".into(),
                    files: Vec::new(),
                },
            ],
        };
        assert_eq!(
            super::select_dataset_ids(&catalog, None).expect("all datasets should be selected"),
            ["first/data", "second/data"]
        );
    }

    #[test]
    fn get_keeps_file_selection_for_one_time_retrieval() {
        let cli = Cli::try_parse_from(["davis", "get", "routes/Matsuyama", "--file", "path.csv"])
            .expect("get command should parse");

        match cli.command {
            Command::Get { files, force, .. } => {
                assert_eq!(files, ["path.csv"]);
                assert!(!force);
            }
            _ => panic!("expected get command"),
        }
    }

    #[test]
    fn update_accepts_non_interactive_confirmation() {
        let cli =
            Cli::try_parse_from(["davis", "update", "--yes"]).expect("update command should parse");

        assert!(matches!(cli.command, Command::Update { yes: true }));
    }

    #[test]
    fn component_management_commands_parse() {
        let install = Cli::try_parse_from(["davis", "install", "component", "components/example"])
            .expect("component install should parse");
        assert!(matches!(
            install.command,
            Command::Install {
                command: InstallCommand::Component {
                    source,
                    version: None,
                    registry: None,
                    json: false
                }
            } if source == "components/example"
        ));

        let inspect = Cli::try_parse_from([
            "davis",
            "component",
            "inspect",
            "davis/mnl",
            "--version",
            "0.1.0",
        ])
        .expect("component inspect should parse");
        assert!(matches!(
            inspect.command,
            Command::Component {
                command: ComponentCommand::Inspect { id, version, json: false }
            } if id == "davis/mnl" && version.as_deref() == Some("0.1.0")
        ));

        let pack = Cli::try_parse_from([
            "davis",
            "component",
            "pack",
            "components/davis-mnl",
            "--out",
            "dist",
            "--name",
            "mnl",
            "--requires-davis",
            ">=0.3.5",
        ])
        .expect("component pack should parse");
        assert!(matches!(
            pack.command,
            Command::Component {
                command: ComponentCommand::Pack {
                    path,
                    out,
                    name,
                    requires_davis,
                    json: false,
                }
            } if path == Path::new("components/davis-mnl")
                && out == Path::new("dist")
                && name.as_deref() == Some("mnl")
                && requires_davis.as_deref() == Some(">=0.3.5")
        ));

        let registry = Cli::try_parse_from([
            "davis",
            "component",
            "registry",
            "dist/mnl.entry.json",
            "--out",
            "dist/component-registry.json",
        ])
        .expect("component registry should parse");
        assert!(matches!(
            registry.command,
            Command::Component {
                command: ComponentCommand::Registry { entries, out, json: false }
            } if entries == [PathBuf::from("dist/mnl.entry.json")]
                && out == Path::new("dist/component-registry.json")
        ));
    }

    #[test]
    fn component_authoring_commands_parse() {
        let scaffold = Cli::try_parse_from([
            "davis",
            "component",
            "scaffold",
            "my-component",
            "--id",
            "example/my-component",
            "--kind",
            "transform",
            "--command",
            "python",
            "--command",
            "-m",
            "--command",
            "my_component",
        ])
        .expect("component scaffold should parse");
        assert!(matches!(
            scaffold.command,
            Command::Component {
                command: ComponentCommand::Scaffold {
                    path,
                    id,
                    runtime_command,
                    ..
                }
            } if path == Path::new("my-component")
                && id == "example/my-component"
                && runtime_command == ["python", "-m", "my_component"]
        ));

        let validate =
            Cli::try_parse_from(["davis", "component", "validate", "my-component", "--json"])
                .expect("component validate should parse");
        assert!(matches!(
            validate.command,
            Command::Component {
                command: ComponentCommand::Validate { path, json: true }
            } if path == Path::new("my-component")
        ));
    }

    #[test]
    fn desktop_bootstrap_commands_parse() {
        let install = Cli::try_parse_from([
            "davis",
            "install",
            "desktop",
            "--version",
            "0.5.0",
            "--registry",
            "https://example.com/software-registry.json",
        ])
        .expect("desktop install should parse");
        assert!(matches!(
            install.command,
            Command::Install {
                command: InstallCommand::Desktop {
                    version,
                    registry,
                    json: false
                }
            } if version.as_deref() == Some("0.5.0")
                && registry.as_deref() == Some("https://example.com/software-registry.json")
        ));

        let alias = Cli::try_parse_from(["davis", "install", "app"])
            .expect("legacy app spelling should parse");
        assert!(matches!(
            alias.command,
            Command::Install {
                command: InstallCommand::Desktop { .. }
            }
        ));

        let launch = Cli::try_parse_from(["davis", "desktop", "--version", "0.5.0"])
            .expect("desktop launch should parse");
        assert!(matches!(
            launch.command,
            Command::Desktop { version } if version.as_deref() == Some("0.5.0")
        ));

        let installed =
            Cli::try_parse_from(["davis", "installed", "--json"]).expect("list should parse");
        assert!(matches!(
            installed.command,
            Command::Installed { json: true }
        ));
    }

    #[test]
    fn overwrite_prompt_accepts_yes() {
        let mut input = Cursor::new(b"yes\n");
        let mut output = Vec::new();

        assert!(super::prompt_to_replace(&mut input, &mut output, 2)
            .expect("prompt should accept input"));
        assert_eq!(
            String::from_utf8(output).expect("prompt should be UTF-8"),
            "Replace all 2 existing file(s)? [y/N] "
        );
    }

    #[test]
    fn overwrite_prompt_defaults_to_no() {
        let mut input = Cursor::new(b"\n");
        let mut output = Vec::new();

        assert!(!super::prompt_to_replace(&mut input, &mut output, 1)
            .expect("prompt should accept input"));
    }

    #[test]
    fn overwrite_prompt_retries_invalid_input() {
        let mut input = Cursor::new(b"maybe\ny\n");
        let mut output = Vec::new();

        assert!(super::prompt_to_replace(&mut input, &mut output, 1)
            .expect("prompt should accept input"));
        assert!(String::from_utf8(output)
            .expect("prompt should be UTF-8")
            .contains("Please answer y or n."));
    }

    #[test]
    fn download_selection_accepts_directory_prefixes() {
        use davis_core::{DatasetManifest, ManifestDataset, ManifestFile, ObjectId, ObjectRef};

        let object = ObjectRef {
            oid: "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .parse::<ObjectId>()
                .expect("object ID should parse"),
            size: 1,
        };
        let manifest = DatasetManifest {
            version: 1,
            dataset: ManifestDataset {
                id: "sample/data".into(),
                root: "data/sample/data".into(),
            },
            files: vec![
                ManifestFile {
                    id: "raw/first.csv".into(),
                    path: "raw/first.csv".into(),
                    object: object.clone(),
                    updated_at: None,
                    schema_path: None,
                },
                ManifestFile {
                    id: "raw/second.csv".into(),
                    path: "raw/second.csv".into(),
                    object,
                    updated_at: None,
                    schema_path: None,
                },
            ],
        };

        let selected = super::select_download_files(&manifest, &["raw".into()])
            .expect("directory prefix should select both files");
        assert_eq!(selected.files.len(), 2);
    }
}
