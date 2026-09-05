use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use davis_model_api::{ComponentKind, ComponentManifest};
use serde::{Deserialize, Serialize};
use tempfile::Builder;
use thiserror::Error;
use walkdir::WalkDir;

use crate::hash_component;

const INSTALL_RECORD: &str = ".davis-install.json";
const INSTALL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ComponentStoreError {
    #[error("Davis data directory could not be determined")]
    DirectoryUnavailable,
    #[error(transparent)]
    Contract(#[from] davis_model_api::ContractError),
    #[error("component source is not a directory: {0}")]
    InvalidSource(PathBuf),
    #[error("component package is invalid: {0}")]
    InvalidPackage(String),
    #[error("component identity `{0}` cannot be used as a portable install path")]
    InvalidIdentity(String),
    #[error("component `{id}` version `{version}` is already installed at {path}")]
    AlreadyInstalled {
        id: String,
        version: String,
        path: PathBuf,
    },
    #[error("component `{id}` is not installed")]
    NotInstalled { id: String },
    #[error("component `{id}` has multiple installed versions; specify --version")]
    VersionRequired { id: String },
    #[error("component package contains a symlink, which is not portable: {0}")]
    Symlink(PathBuf),
    #[error("failed to walk component package at {path}: {source}")]
    Walk {
        path: PathBuf,
        source: walkdir::Error,
    },
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize component install record: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Runtime(#[from] crate::RuntimeError),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledComponent {
    pub install_schema_version: u32,
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub kind: ComponentKind,
    pub path: PathBuf,
    pub source: String,
    pub source_digest: String,
    pub installed_at_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedComponentPackage {
    pub source: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: ComponentManifest,
    pub configuration_source: PathBuf,
    pub presentation_source: Option<PathBuf>,
}

/// Validates a component package without installing or modifying it.
///
/// # Errors
///
/// Returns an error when the source, manifest, referenced documents, runtime
/// lockfile, or package paths are invalid.
pub fn validate_component_package(
    source: &Path,
) -> Result<ValidatedComponentPackage, ComponentStoreError> {
    let source = fs::canonicalize(source).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ComponentStoreError::InvalidSource(source.to_owned())
        } else {
            ComponentStoreError::Io {
                path: source.to_owned(),
                source: error,
            }
        }
    })?;
    if !source.is_dir() {
        return Err(ComponentStoreError::InvalidSource(source));
    }
    let (manifest_path, manifest) = ComponentManifest::read_from_directory(&source)?;
    validate_package(&source, &manifest_path, &manifest)?;
    let configuration = manifest.resolve_configuration(&manifest_path)?;
    let presentation = manifest.resolve_presentation(&manifest_path)?;
    Ok(ValidatedComponentPackage {
        source,
        manifest_path,
        manifest,
        configuration_source: configuration.source_path,
        presentation_source: presentation.map(|document| document.source_path),
    })
}

#[derive(Debug, Clone)]
pub struct ComponentStore {
    root: PathBuf,
}

impl ComponentStore {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Opens the per-user component store for the current operating system.
    ///
    /// # Errors
    ///
    /// Returns an error when no suitable user data directory can be found.
    pub fn for_user() -> Result<Self, ComponentStoreError> {
        Ok(Self::new(user_data_directory()?.join("components")))
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Validates and atomically installs a local component package.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid package, unsafe identity, duplicate
    /// version, symlink, or filesystem failure.
    pub fn install(&self, source: &Path) -> Result<InstalledComponent, ComponentStoreError> {
        self.install_with_origin(source, None)
    }

    /// Installs a package while recording a registry URL or other stable origin.
    ///
    /// # Errors
    ///
    /// Returns the same validation and filesystem errors as [`Self::install`].
    pub fn install_with_origin(
        &self,
        source: &Path,
        origin: Option<String>,
    ) -> Result<InstalledComponent, ComponentStoreError> {
        let validated = validate_component_package(source)?;
        let source = validated.source;
        let manifest = validated.manifest;
        let destination = self.destination(&manifest.id, &manifest.version)?;
        if destination.exists() {
            return Err(ComponentStoreError::AlreadyInstalled {
                id: manifest.id,
                version: manifest.version,
                path: destination,
            });
        }
        let parent = destination
            .parent()
            .ok_or_else(|| ComponentStoreError::InvalidIdentity(manifest.id.clone()))?;
        fs::create_dir_all(parent).map_err(|source| ComponentStoreError::Io {
            path: parent.to_owned(),
            source,
        })?;
        let temporary = Builder::new()
            .prefix(".install-")
            .tempdir_in(parent)
            .map_err(|source| ComponentStoreError::Io {
                path: parent.to_owned(),
                source,
            })?;
        copy_package(&source, temporary.path())?;
        let digest = format!("blake3:{}", hash_component(&source)?);
        let installed = InstalledComponent {
            install_schema_version: INSTALL_SCHEMA_VERSION,
            id: manifest.id,
            name: manifest.name,
            version: manifest.version,
            kind: manifest.kind,
            path: destination.clone(),
            source: origin.unwrap_or_else(|| source.to_string_lossy().into_owned()),
            source_digest: digest,
            installed_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let record_path = temporary.path().join(INSTALL_RECORD);
        let record = serde_json::to_vec_pretty(&installed)?;
        fs::write(&record_path, record).map_err(|source| ComponentStoreError::Io {
            path: record_path,
            source,
        })?;
        fs::rename(temporary.path(), &destination).map_err(|source| ComponentStoreError::Io {
            path: destination.clone(),
            source,
        })?;
        Ok(installed)
    }

    /// Lists valid components installed in this store.
    ///
    /// # Errors
    ///
    /// Returns an error when an install record cannot be read or decoded.
    pub fn list(&self) -> Result<Vec<InstalledComponent>, ComponentStoreError> {
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
            let entry = entry.map_err(|source| ComponentStoreError::Walk {
                path: self.root.clone(),
                source,
            })?;
            if entry.file_name() != INSTALL_RECORD {
                continue;
            }
            let bytes = fs::read(entry.path()).map_err(|source| ComponentStoreError::Io {
                path: entry.path().to_owned(),
                source,
            })?;
            let mut item: InstalledComponent = serde_json::from_slice(&bytes)?;
            if item.install_schema_version != INSTALL_SCHEMA_VERSION {
                return Err(ComponentStoreError::InvalidPackage(format!(
                    "unsupported install record version {} at {}",
                    item.install_schema_version,
                    entry.path().display()
                )));
            }
            let package_path = entry.path().parent().ok_or_else(|| {
                ComponentStoreError::InvalidPackage(format!(
                    "install record has no component directory: {}",
                    entry.path().display()
                ))
            })?;
            let (_, manifest) = ComponentManifest::read_from_directory(package_path)?;
            if manifest.id != item.id || manifest.version != item.version {
                return Err(ComponentStoreError::InvalidPackage(format!(
                    "install record identity does not match manifest at {}",
                    package_path.display()
                )));
            }
            package_path.clone_into(&mut item.path);
            installed.push(item);
        }
        installed.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.version.cmp(&right.version))
        });
        Ok(installed)
    }

    /// Returns one installed component, requiring a version when ambiguous.
    ///
    /// # Errors
    ///
    /// Returns an error when the component is missing or has multiple versions.
    pub fn inspect(
        &self,
        id: &str,
        version: Option<&str>,
    ) -> Result<InstalledComponent, ComponentStoreError> {
        let mut matches: Vec<InstalledComponent> = self
            .list()?
            .into_iter()
            .filter(|component| {
                component.id == id && version.is_none_or(|value| component.version == value)
            })
            .collect();
        match matches.len() {
            0 => Err(ComponentStoreError::NotInstalled { id: id.to_owned() }),
            1 => Ok(matches.remove(0)),
            _ => Err(ComponentStoreError::VersionRequired { id: id.to_owned() }),
        }
    }

    /// Removes exactly one installed component version.
    ///
    /// # Errors
    ///
    /// Returns an error when selection is ambiguous or removal fails.
    pub fn remove(
        &self,
        id: &str,
        version: Option<&str>,
    ) -> Result<InstalledComponent, ComponentStoreError> {
        let installed = self.inspect(id, version)?;
        if installed.path == self.root || !installed.path.starts_with(&self.root) {
            return Err(ComponentStoreError::InvalidPackage(format!(
                "installed component path escapes the store: {}",
                installed.path.display()
            )));
        }
        fs::remove_dir_all(&installed.path).map_err(|source| ComponentStoreError::Io {
            path: installed.path.clone(),
            source,
        })?;
        prune_empty_parents(installed.path.parent(), &self.root);
        Ok(installed)
    }

    fn destination(&self, id: &str, version: &str) -> Result<PathBuf, ComponentStoreError> {
        let mut destination = self.root.clone();
        for segment in id.split('/') {
            validate_identity_segment(segment, id)?;
            destination.push(segment);
        }
        validate_identity_segment(version, version)?;
        destination.push(version);
        Ok(destination)
    }
}

/// Returns the per-user Davis data directory.
///
/// `DAVIS_DATA_HOME` overrides the platform default, primarily for isolated
/// development and tests.
///
/// # Errors
///
/// Returns an error when neither an override nor a home directory is present.
pub fn user_data_directory() -> Result<PathBuf, ComponentStoreError> {
    if let Some(directory) = std::env::var_os("DAVIS_DATA_HOME") {
        return Ok(PathBuf::from(directory));
    }
    #[cfg(target_os = "windows")]
    if let Some(directory) =
        std::env::var_os("LOCALAPPDATA").or_else(|| std::env::var_os("APPDATA"))
    {
        return Ok(PathBuf::from(directory).join("Davis"));
    }
    #[cfg(target_os = "macos")]
    if let Some(directory) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(directory).join("Library/Application Support/Davis"));
    }
    if let Some(directory) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(directory).join("davis"));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|directory| directory.join(".local/share/davis"))
        .ok_or(ComponentStoreError::DirectoryUnavailable)
}

fn validate_package(
    root: &Path,
    manifest_path: &Path,
    manifest: &ComponentManifest,
) -> Result<(), ComponentStoreError> {
    for segment in manifest.id.split('/') {
        validate_identity_segment(segment, &manifest.id)?;
    }
    validate_identity_segment(&manifest.version, &manifest.version)?;
    if manifest.runtime.command.is_empty() {
        return Err(ComponentStoreError::InvalidPackage(
            "runtime.command must not be empty".to_owned(),
        ));
    }
    manifest.resolve_configuration(manifest_path)?;
    manifest.resolve_presentation(manifest_path)?;
    if let Some(path) = &manifest.runtime.lockfile {
        require_package_file(root, path, "runtime.lockfile")?;
    }
    Ok(())
}

fn require_package_file(
    root: &Path,
    relative: &Path,
    field: &str,
) -> Result<(), ComponentStoreError> {
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ComponentStoreError::InvalidPackage(format!(
            "{field} must be a safe relative path"
        )));
    }
    let path = root.join(relative);
    if !path.is_file() {
        return Err(ComponentStoreError::InvalidPackage(format!(
            "{field} does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

fn validate_identity_segment(segment: &str, identity: &str) -> Result<(), ComponentStoreError> {
    if segment.is_empty()
        || segment == "."
        || segment == ".."
        || !segment.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
    {
        return Err(ComponentStoreError::InvalidIdentity(identity.to_owned()));
    }
    Ok(())
}

fn copy_package(source: &Path, destination: &Path) -> Result<(), ComponentStoreError> {
    let entries = WalkDir::new(source).into_iter().filter_entry(|entry| {
        entry
            .path()
            .strip_prefix(source)
            .map_or(true, |relative| !should_exclude(relative))
    });
    for entry in entries {
        let entry = entry.map_err(|source_error| ComponentStoreError::Walk {
            path: source.to_owned(),
            source: source_error,
        })?;
        let relative = entry
            .path()
            .strip_prefix(source)
            .map_err(|_| ComponentStoreError::InvalidSource(source.to_owned()))?;
        if relative.as_os_str().is_empty() {
            continue;
        }
        if entry.file_type().is_symlink() {
            return Err(ComponentStoreError::Symlink(entry.path().to_owned()));
        }
        let target = destination.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target).map_err(|source| ComponentStoreError::Io {
                path: target,
                source,
            })?;
        } else if entry.file_type().is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|source| ComponentStoreError::Io {
                    path: parent.to_owned(),
                    source,
                })?;
            }
            fs::copy(entry.path(), &target).map_err(|source| ComponentStoreError::Io {
                path: target,
                source,
            })?;
        }
    }
    Ok(())
}

fn should_exclude(relative: &Path) -> bool {
    relative.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".venv" | "__pycache__" | ".pytest_cache" | ".git" | "target")
        )
    }) || relative
        .file_name()
        .is_some_and(|name| matches!(name.to_str(), Some(".DS_Store" | INSTALL_RECORD)))
}

fn prune_empty_parents(mut directory: Option<&Path>, root: &Path) {
    while let Some(path) = directory {
        if path == root || !path.starts_with(root) || fs::remove_dir(path).is_err() {
            break;
        }
        directory = path.parent();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{validate_component_package, ComponentStore, ComponentStoreError};

    #[test]
    fn installs_lists_inspects_and_removes_a_component() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir_all(source.join("schemas")).unwrap();
        fs::create_dir_all(source.join(".venv/bin")).unwrap();
        fs::write(
            source.join("component-manifest.yaml"),
            r"api_version: davis.component/v1alpha1
id: example/native
name: Example Native
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs:
  - name: data
    media_types: [text/csv]
config_schema: schemas/config.json
outputs: {}
",
        )
        .unwrap();
        fs::write(source.join("schemas/config.json"), "{}").unwrap();
        fs::write(source.join(".venv/bin/generated"), "ignored").unwrap();
        let store = ComponentStore::new(temporary.path().join("store"));

        let installed = store.install(&source).unwrap();
        assert_eq!(installed.id, "example/native");
        assert!(installed.path.join("component-manifest.yaml").is_file());
        assert!(!installed.path.join(".venv").exists());
        assert!(matches!(
            store.install(&source),
            Err(ComponentStoreError::AlreadyInstalled { .. })
        ));
        assert_eq!(store.list().unwrap(), vec![installed.clone()]);
        assert_eq!(store.inspect("example/native", None).unwrap(), installed);
        let removed = store.remove("example/native", Some("1.0.0")).unwrap();
        assert_eq!(removed.id, "example/native");
        assert!(store.list().unwrap().is_empty());

        let unsafe_manifest = fs::read_to_string(source.join("component-manifest.yaml"))
            .unwrap()
            .replace("example/native", "../escape");
        fs::write(source.join("component-manifest.yaml"), unsafe_manifest).unwrap();
        assert!(matches!(
            store.install(&source),
            Err(ComponentStoreError::InvalidIdentity(_))
        ));
        assert!(!temporary.path().join("escape").exists());
    }

    #[test]
    fn installs_a_self_contained_component_yaml() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        fs::write(
            source.join("component.yaml"),
            r"api_version: davis.component/v1
id: example/inline
name: Inline component
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs: []
configuration:
  schema:
    type: object
presentation:
  ui:
    ui:editor: generic
outputs: {}
",
        )
        .unwrap();
        let store = ComponentStore::new(temporary.path().join("store"));

        let installed = store.install(&source).unwrap();

        assert!(installed.path.join("component.yaml").is_file());
        assert_eq!(store.inspect("example/inline", None).unwrap(), installed);
    }

    #[test]
    fn validates_without_installing_and_rejects_unsafe_identity() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        let manifest_path = source.join("component.yaml");
        fs::write(
            &manifest_path,
            r"api_version: davis.component/v1
id: example/validation
name: Validation example
version: 1.0.0
runtime:
  executor: process
  command: [example]
operations: [estimate]
inputs: []
configuration:
  schema:
    type: object
outputs: {}
",
        )
        .unwrap();

        let validated = validate_component_package(&source).unwrap();
        assert_eq!(validated.manifest.id, "example/validation");
        assert_eq!(fs::read_dir(&source).unwrap().count(), 1);

        let unsafe_manifest = fs::read_to_string(&manifest_path)
            .unwrap()
            .replace("example/validation", "../escape");
        fs::write(&manifest_path, unsafe_manifest).unwrap();
        assert!(matches!(
            validate_component_package(&source),
            Err(ComponentStoreError::InvalidIdentity(_))
        ));
    }
}
