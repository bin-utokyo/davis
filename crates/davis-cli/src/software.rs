use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use reqwest::{Client, Url};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use tempfile::Builder;
use thiserror::Error;
use walkdir::WalkDir;

const DEFAULT_REGISTRY_URL: &str =
    "https://github.com/bin-utokyo/davis/releases/latest/download/software-registry.json";
const INSTALL_RECORD: &str = ".davis-software-install.json";
const MAX_REGISTRY_BYTES: u64 = 2 * 1024 * 1024;
const MAX_BUNDLE_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub(crate) enum SoftwareError {
    #[error("invalid software registry URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("insecure software URL is not allowed: {0}")]
    InsecureUrl(Url),
    #[error("failed to retrieve software metadata or package: {0}")]
    Request(#[from] reqwest::Error),
    #[error("software registry returned HTTP {0}")]
    Http(reqwest::StatusCode),
    #[error("software registry is larger than the {MAX_REGISTRY_BYTES} byte limit")]
    RegistryTooLarge,
    #[error("unsupported software registry schema version: {0}")]
    UnsupportedSchema(u32),
    #[error("unsupported software install record schema version: {0}")]
    UnsupportedInstallSchema(u32),
    #[error("invalid software registry JSON: {0}")]
    InvalidRegistry(#[from] serde_json::Error),
    #[error("invalid semantic version `{value}` in software registry: {source}")]
    InvalidVersion {
        value: String,
        source: semver::Error,
    },
    #[error("software package `{0}` was not found in the registry")]
    NotFound(String),
    #[error("no version of `{package}` supports Davis v{davis_version} on target `{target}`")]
    NoCompatibleVersion {
        package: String,
        davis_version: String,
        target: String,
    },
    #[error("software bundle is larger than the {MAX_BUNDLE_BYTES} byte limit")]
    BundleTooLarge,
    #[error("software bundle size mismatch: expected {expected}, received {actual}")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("software bundle digest mismatch: expected {expected}, received {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error("software bundle digest must use a valid blake3 value: {0}")]
    InvalidDigest(String),
    #[error("software entrypoint must be a safe relative path: {0}")]
    UnsafeEntrypoint(PathBuf),
    #[error("software entrypoint does not exist: {0}")]
    MissingEntrypoint(PathBuf),
    #[error("software package `{id}` version `{version}` is already installed at {path}")]
    AlreadyInstalled {
        id: String,
        version: String,
        path: PathBuf,
    },
    #[error("software package `{0}` is not installed")]
    NotInstalled(String),
    #[error("installed software record identity does not match its path: {0}")]
    InstallRecordPath(PathBuf),
    #[error("software package contains an unsupported symbolic link: {0}")]
    Symlink(PathBuf),
    #[error("failed to walk software package at {path}: {source}")]
    Walk {
        path: PathBuf,
        source: walkdir::Error,
    },
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to launch Davis Desktop from {path}: {source}")]
    Launch {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Archive(#[from] crate::component_registry::RegistryError),
    #[error(transparent)]
    DataDirectory(#[from] davis_runtime::ComponentStoreError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SoftwareRegistry {
    schema_version: u32,
    packages: Vec<SoftwarePackage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SoftwarePackage {
    id: String,
    name: String,
    version: String,
    requires_davis: String,
    artifacts: Vec<SoftwareArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SoftwareArtifact {
    target: String,
    url: String,
    size: u64,
    blake3: String,
    entrypoint: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct InstalledSoftware {
    install_schema_version: u32,
    id: String,
    name: String,
    version: String,
    target: String,
    path: PathBuf,
    entrypoint: PathBuf,
    source: String,
    source_digest: String,
    installed_at_unix_seconds: u64,
}

#[derive(Debug, Serialize)]
struct InstalledInventory {
    software: Vec<InstalledSoftware>,
    components: Vec<davis_runtime::InstalledComponent>,
}

#[derive(Debug, Clone)]
struct SoftwareStore {
    root: PathBuf,
}

impl SoftwareStore {
    fn for_user() -> Result<Self, SoftwareError> {
        Ok(Self {
            root: davis_runtime::user_data_directory()?.join("software"),
        })
    }

    #[cfg(test)]
    fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn install(
        &self,
        source: &Path,
        package: &SoftwarePackage,
        artifact: &SoftwareArtifact,
        origin: String,
    ) -> Result<InstalledSoftware, SoftwareError> {
        validate_segment(&package.id)?;
        validate_segment(&package.version)?;
        validate_entrypoint(&artifact.entrypoint)?;
        let source_entrypoint = source.join(&artifact.entrypoint);
        if !source_entrypoint.is_file() && !source_entrypoint.is_dir() {
            return Err(SoftwareError::MissingEntrypoint(source_entrypoint));
        }
        let destination = self.root.join(&package.id).join(&package.version);
        if destination.exists() {
            return Err(SoftwareError::AlreadyInstalled {
                id: package.id.clone(),
                version: package.version.clone(),
                path: destination,
            });
        }
        let parent = destination
            .parent()
            .expect("version destination has a parent");
        fs::create_dir_all(parent).map_err(|source| SoftwareError::Io {
            path: parent.to_owned(),
            source,
        })?;
        let temporary = Builder::new()
            .prefix(".install-")
            .tempdir_in(parent)
            .map_err(|source| SoftwareError::Io {
                path: parent.to_owned(),
                source,
            })?;
        copy_tree(source, temporary.path())?;
        let installed = InstalledSoftware {
            install_schema_version: 1,
            id: package.id.clone(),
            name: package.name.clone(),
            version: package.version.clone(),
            target: artifact.target.clone(),
            path: destination.clone(),
            entrypoint: artifact.entrypoint.clone(),
            source: origin,
            source_digest: artifact.blake3.clone(),
            installed_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let record_path = temporary.path().join(INSTALL_RECORD);
        fs::write(&record_path, serde_json::to_vec_pretty(&installed)?).map_err(|source| {
            SoftwareError::Io {
                path: record_path,
                source,
            }
        })?;
        fs::rename(temporary.path(), &destination).map_err(|source| SoftwareError::Io {
            path: destination,
            source,
        })?;
        Ok(installed)
    }

    fn list(&self) -> Result<Vec<InstalledSoftware>, SoftwareError> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut installed = Vec::new();
        for entry in WalkDir::new(&self.root).into_iter().filter_entry(|entry| {
            entry
                .file_name()
                .to_str()
                .is_none_or(|name| !name.starts_with(".install-"))
        }) {
            let entry = entry.map_err(|source| SoftwareError::Walk {
                path: self.root.clone(),
                source,
            })?;
            if entry.file_name() != INSTALL_RECORD {
                continue;
            }
            let record_path = entry.path();
            let bytes = fs::read(record_path).map_err(|source| SoftwareError::Io {
                path: record_path.to_owned(),
                source,
            })?;
            let mut item: InstalledSoftware = serde_json::from_slice(&bytes)?;
            if item.install_schema_version != 1 {
                return Err(SoftwareError::UnsupportedInstallSchema(
                    item.install_schema_version,
                ));
            }
            let path = record_path.parent().expect("install record has a parent");
            validate_segment(&item.id)?;
            validate_segment(&item.version)?;
            validate_entrypoint(&item.entrypoint)?;
            Version::parse(&item.version).map_err(|source| SoftwareError::InvalidVersion {
                value: item.version.clone(),
                source,
            })?;
            if path != self.root.join(&item.id).join(&item.version) {
                return Err(SoftwareError::InstallRecordPath(path.to_owned()));
            }
            path.clone_into(&mut item.path);
            let entrypoint = path.join(&item.entrypoint);
            if !entrypoint.is_file() && !entrypoint.is_dir() {
                return Err(SoftwareError::MissingEntrypoint(entrypoint));
            }
            installed.push(item);
        }
        installed.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.version.cmp(&right.version))
        });
        Ok(installed)
    }

    fn inspect(&self, id: &str, version: Option<&str>) -> Result<InstalledSoftware, SoftwareError> {
        let mut matching: Vec<_> = self
            .list()?
            .into_iter()
            .filter(|item| item.id == id && version.is_none_or(|value| item.version == value))
            .collect();
        if matching.is_empty() {
            return Err(SoftwareError::NotInstalled(id.to_owned()));
        }
        matching.sort_by(|left, right| {
            let left = Version::parse(&left.version).expect("installed version was validated");
            let right = Version::parse(&right.version).expect("installed version was validated");
            left.cmp(&right)
        });
        Ok(matching.pop().expect("matching is not empty"))
    }
}

pub(crate) async fn install_desktop(
    version: Option<&str>,
    registry_override: Option<&str>,
    json: bool,
) -> Result<(), SoftwareError> {
    let registry_url = registry_url(registry_override)?;
    require_secure_url(&registry_url)?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_mins(5))
        .build()?;
    let registry = download_registry(&client, &registry_url).await?;
    let (package, artifact) = select_package(&registry, "desktop", version, current_target())?;
    let bundle_url = registry_url.join(&artifact.url)?;
    require_secure_url(&bundle_url)?;
    let temporary = tempfile::tempdir().map_err(|source| SoftwareError::Io {
        path: std::env::temp_dir(),
        source,
    })?;
    let archive_path = temporary.path().join("software.tar.gz");
    download_bundle(&client, &bundle_url, artifact, &archive_path).await?;
    let extracted = temporary.path().join("software");
    fs::create_dir(&extracted).map_err(|source| SoftwareError::Io {
        path: extracted.clone(),
        source,
    })?;
    crate::component_registry::extract_bundle(&archive_path, &extracted)?;
    let origin = format!(
        "registry:{}@{}#{}",
        package.id, package.version, artifact.target
    );
    let installed = SoftwareStore::for_user()?.install(&extracted, package, artifact, origin)?;
    print_software(&installed, json)?;
    Ok(())
}

pub(crate) fn launch_desktop(version: Option<&str>) -> Result<(), SoftwareError> {
    let installed = SoftwareStore::for_user()?.inspect("desktop", version)?;
    let entrypoint = installed.path.join(&installed.entrypoint);
    #[cfg(target_os = "macos")]
    let result = if entrypoint.extension().is_some_and(|value| value == "app") {
        Command::new("open").arg(&entrypoint).spawn()
    } else {
        Command::new(&entrypoint).spawn()
    };
    #[cfg(not(target_os = "macos"))]
    let result = Command::new(&entrypoint).spawn();
    result.map_err(|source| SoftwareError::Launch {
        path: entrypoint.clone(),
        source,
    })?;
    println!("Launched: {} {}", installed.name, installed.version);
    println!("Path: {}", entrypoint.display());
    Ok(())
}

pub(crate) fn print_installed(json: bool) -> Result<(), SoftwareError> {
    let inventory = InstalledInventory {
        software: SoftwareStore::for_user()?.list()?,
        components: davis_runtime::ComponentStore::for_user()?.list()?,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&inventory)?);
        return Ok(());
    }
    if inventory.software.is_empty() && inventory.components.is_empty() {
        println!("No optional Davis software or components are installed.");
        return Ok(());
    }
    for item in inventory.software {
        println!(
            "software\t{}\t{}\t{}",
            item.id,
            item.version,
            item.path.display()
        );
    }
    for item in inventory.components {
        println!(
            "component\t{}\t{}\t{}",
            item.id,
            item.version,
            item.path.display()
        );
    }
    Ok(())
}

async fn download_registry(client: &Client, url: &Url) -> Result<SoftwareRegistry, SoftwareError> {
    let response = client.get(url.clone()).send().await?;
    require_secure_url(response.url())?;
    if !response.status().is_success() {
        return Err(SoftwareError::Http(response.status()));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_REGISTRY_BYTES)
    {
        return Err(SoftwareError::RegistryTooLarge);
    }
    let bytes = response.bytes().await?;
    if bytes.len() as u64 > MAX_REGISTRY_BYTES {
        return Err(SoftwareError::RegistryTooLarge);
    }
    let registry: SoftwareRegistry = serde_json::from_slice(&bytes)?;
    if registry.schema_version != 1 {
        return Err(SoftwareError::UnsupportedSchema(registry.schema_version));
    }
    Ok(registry)
}

fn select_package<'a>(
    registry: &'a SoftwareRegistry,
    id: &str,
    requested_version: Option<&str>,
    target: &str,
) -> Result<(&'a SoftwarePackage, &'a SoftwareArtifact), SoftwareError> {
    let current = Version::parse(env!("CARGO_PKG_VERSION")).map_err(|source| {
        SoftwareError::InvalidVersion {
            value: env!("CARGO_PKG_VERSION").to_owned(),
            source,
        }
    })?;
    let mut compatible = Vec::new();
    let mut found = false;
    for package in registry.packages.iter().filter(|package| package.id == id) {
        if requested_version.is_some_and(|version| package.version != version) {
            continue;
        }
        found = true;
        let version =
            Version::parse(&package.version).map_err(|source| SoftwareError::InvalidVersion {
                value: package.version.clone(),
                source,
            })?;
        let requirement = VersionReq::parse(&package.requires_davis).map_err(|source| {
            SoftwareError::InvalidVersion {
                value: package.requires_davis.clone(),
                source,
            }
        })?;
        if requirement.matches(&current) {
            if let Some(artifact) = package
                .artifacts
                .iter()
                .find(|artifact| artifact.target == target)
            {
                compatible.push((version, package, artifact));
            }
        }
    }
    if !found {
        return Err(SoftwareError::NotFound(id.to_owned()));
    }
    compatible.sort_by(|left, right| left.0.cmp(&right.0));
    compatible
        .pop()
        .map(|(_, package, artifact)| (package, artifact))
        .ok_or_else(|| SoftwareError::NoCompatibleVersion {
            package: id.to_owned(),
            davis_version: current.to_string(),
            target: target.to_owned(),
        })
}

async fn download_bundle(
    client: &Client,
    url: &Url,
    artifact: &SoftwareArtifact,
    destination: &Path,
) -> Result<(), SoftwareError> {
    if artifact.size > MAX_BUNDLE_BYTES {
        return Err(SoftwareError::BundleTooLarge);
    }
    let expected = parse_digest(&artifact.blake3)?;
    let response = client.get(url.clone()).send().await?;
    require_secure_url(response.url())?;
    if !response.status().is_success() {
        return Err(SoftwareError::Http(response.status()));
    }
    let mut file = File::create(destination).map_err(|source| SoftwareError::Io {
        path: destination.to_owned(),
        source,
    })?;
    let mut size = 0_u64;
    let mut hasher = blake3::Hasher::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        size = size.saturating_add(chunk.len() as u64);
        if size > artifact.size || size > MAX_BUNDLE_BYTES {
            return Err(SoftwareError::BundleTooLarge);
        }
        hasher.update(&chunk);
        file.write_all(&chunk).map_err(|source| SoftwareError::Io {
            path: destination.to_owned(),
            source,
        })?;
    }
    if size != artifact.size {
        return Err(SoftwareError::SizeMismatch {
            expected: artifact.size,
            actual: size,
        });
    }
    let actual = hasher.finalize();
    if actual != expected {
        return Err(SoftwareError::DigestMismatch {
            expected: artifact.blake3.clone(),
            actual: format!("blake3:{actual}"),
        });
    }
    Ok(())
}

fn registry_url(override_url: Option<&str>) -> Result<Url, SoftwareError> {
    let value = override_url
        .map(str::to_owned)
        .or_else(|| std::env::var("DAVIS_SOFTWARE_REGISTRY_URL").ok())
        .unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_owned());
    Ok(Url::parse(&value)?)
}

fn require_secure_url(url: &Url) -> Result<(), SoftwareError> {
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    if url.scheme() == "https" || url.scheme() == "http" && local {
        Ok(())
    } else {
        Err(SoftwareError::InsecureUrl(url.clone()))
    }
}

fn parse_digest(value: &str) -> Result<blake3::Hash, SoftwareError> {
    value
        .strip_prefix("blake3:")
        .ok_or_else(|| SoftwareError::InvalidDigest(value.to_owned()))?
        .parse()
        .map_err(|_| SoftwareError::InvalidDigest(value.to_owned()))
}

fn current_target() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("windows", "aarch64") => "aarch64-pc-windows-msvc",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        _ => "unsupported",
    }
}

fn validate_entrypoint(path: &Path) -> Result<(), SoftwareError> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(SoftwareError::UnsafeEntrypoint(path.to_owned()));
    }
    Ok(())
}

fn validate_segment(value: &str) -> Result<(), SoftwareError> {
    if value.is_empty()
        || !value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
    {
        return Err(SoftwareError::UnsafeEntrypoint(PathBuf::from(value)));
    }
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), SoftwareError> {
    for entry in WalkDir::new(source) {
        let entry = entry.map_err(|source_error| SoftwareError::Walk {
            path: source.to_owned(),
            source: source_error,
        })?;
        let relative = entry
            .path()
            .strip_prefix(source)
            .expect("walk entry is below root");
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = destination.join(relative);
        if entry.file_type().is_symlink() {
            return Err(SoftwareError::Symlink(relative.to_owned()));
        }
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target).map_err(|source| SoftwareError::Io {
                path: target,
                source,
            })?;
        } else if entry.file_type().is_file() {
            fs::copy(entry.path(), &target).map_err(|source| SoftwareError::Io {
                path: target,
                source,
            })?;
        }
    }
    Ok(())
}

fn print_software(installed: &InstalledSoftware, json: bool) -> Result<(), SoftwareError> {
    if json {
        println!("{}", serde_json::to_string_pretty(installed)?);
    } else {
        println!("Installed: {} {}", installed.name, installed.version);
        println!("Target: {}", installed.target);
        println!("Path: {}", installed.path.display());
        println!("Launch with: davis desktop");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::{
        select_package, SoftwareArtifact, SoftwarePackage, SoftwareRegistry, SoftwareStore,
    };

    fn package(version: &str, requirement: &str, target: &str) -> SoftwarePackage {
        SoftwarePackage {
            id: "desktop".to_owned(),
            name: "Davis Desktop".to_owned(),
            version: version.to_owned(),
            requires_davis: requirement.to_owned(),
            artifacts: vec![SoftwareArtifact {
                target: target.to_owned(),
                url: "desktop.tar.gz".to_owned(),
                size: 1,
                blake3: format!("blake3:{}", "a".repeat(64)),
                entrypoint: PathBuf::from("davis-app"),
            }],
        }
    }

    #[test]
    fn selects_latest_compatible_package_for_target() {
        let target = super::current_target();
        let registry = SoftwareRegistry {
            schema_version: 1,
            packages: vec![
                package("0.5.0", ">=0.5.0", target),
                package("0.6.0", ">=0.6.0", target),
                package("0.5.1", ">=0.5.0", "some-other-target"),
            ],
        };
        let (selected, _) = select_package(&registry, "desktop", None, target).unwrap();
        assert_eq!(selected.version, "0.5.0");
    }

    #[test]
    fn installs_and_discovers_a_package_without_using_cwd() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("davis-app"), "binary").unwrap();
        let package = package("0.5.0", ">=0.5.0", super::current_target());
        let artifact = &package.artifacts[0];
        let store = SoftwareStore::new(temporary.path().join("managed"));
        let installed = store
            .install(&source, &package, artifact, "test".to_owned())
            .unwrap();

        assert_eq!(
            installed.path,
            temporary.path().join("managed/desktop/0.5.0")
        );
        assert_eq!(store.inspect("desktop", None).unwrap(), installed);
        assert!(installed.path.join("davis-app").is_file());
    }

    #[test]
    fn selects_latest_installed_desktop_by_default() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("davis-app"), "binary").unwrap();
        let store = SoftwareStore::new(temporary.path().join("managed"));
        for version in ["0.5.0", "0.5.1"] {
            let package = package(version, ">=0.5.0", super::current_target());
            store
                .install(&source, &package, &package.artifacts[0], "test".to_owned())
                .unwrap();
        }

        assert_eq!(store.inspect("desktop", None).unwrap().version, "0.5.1");
        assert_eq!(
            store.inspect("desktop", Some("0.5.0")).unwrap().version,
            "0.5.0"
        );
    }
}
