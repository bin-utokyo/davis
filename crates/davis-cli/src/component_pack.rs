use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use davis_model_api::ComponentManifest;
use davis_runtime::ComponentStore;
use flate2::{Compression, GzBuilder};
use semver::{Version, VersionReq};
use serde::Serialize;
use tempfile::NamedTempFile;
use thiserror::Error;
use walkdir::WalkDir;

use crate::component_registry::{Bundle, Registry, RegistryComponent};

const INSTALL_RECORD: &str = ".davis-install.json";

#[derive(Debug, Error)]
pub(crate) enum PackError {
    #[error(transparent)]
    Component(#[from] davis_runtime::ComponentStoreError),
    #[error(transparent)]
    Contract(#[from] davis_model_api::ContractError),
    #[error(transparent)]
    Registry(#[from] crate::component_registry::RegistryError),
    #[error("component package has no portable files")]
    EmptyPackage,
    #[error("component package contains a non-portable path: {0}")]
    NonPortablePath(PathBuf),
    #[error("invalid official component name `{0}`")]
    InvalidName(String),
    #[error("component manifest does not declare requires_davis; add it or pass --requires-davis")]
    MissingRequirement,
    #[error("invalid semantic version `{value}`: {source}")]
    InvalidVersion {
        value: String,
        source: semver::Error,
    },
    #[error("component registry requires at least one entry file")]
    EmptyRegistry,
    #[error("duplicate component registry entry: {0}")]
    DuplicateEntry(String),
    #[error("registry bundle URL must be one portable relative file name: {0}")]
    InvalidBundleUrl(String),
    #[error("component bundle digest must use a valid blake3 value: {0}")]
    InvalidDigest(String),
    #[error("component bundle size mismatch for {path}: expected {expected}, found {actual}")]
    BundleSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("component bundle digest mismatch for {path}: expected {expected}, found {actual}")]
    BundleDigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error(
        "packed component identity `{actual_id}` `{actual_version}` does not match source `{expected_id}` `{expected_version}`"
    )]
    IdentityMismatch {
        expected_id: String,
        expected_version: String,
        actual_id: String,
        actual_version: String,
    },
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
    #[error("failed to serialize package metadata: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PackedComponent {
    pub(crate) bundle_path: PathBuf,
    pub(crate) entry_path: PathBuf,
    pub(crate) entry: RegistryComponent,
}

pub(crate) fn pack(
    source: &Path,
    output_directory: &Path,
    name: Option<&str>,
    requires_davis: Option<&str>,
) -> Result<PackedComponent, PackError> {
    let staging = tempfile::tempdir().map_err(|source| PackError::Io {
        path: std::env::temp_dir(),
        source,
    })?;
    let installed = ComponentStore::new(staging.path().join("store")).install(source)?;
    let short_name = name.map_or_else(
        || {
            installed
                .id
                .rsplit('/')
                .next()
                .unwrap_or(&installed.id)
                .to_owned()
        },
        str::to_owned,
    );
    validate_name(&short_name)?;
    let (_, manifest) = ComponentManifest::read_from_directory(&installed.path)?;
    let requirement = requires_davis
        .map(str::to_owned)
        .or(manifest.requires_davis)
        .ok_or(PackError::MissingRequirement)?;
    validate_requirement(&requirement)?;
    fs::create_dir_all(output_directory).map_err(|source| PackError::Io {
        path: output_directory.to_owned(),
        source,
    })?;

    let stem = format!("{}-{}", installed.id.replace('/', "-"), installed.version);
    let bundle_name = format!("{stem}.tar.gz");
    let bundle_path = output_directory.join(&bundle_name);
    write_deterministic_bundle(&installed.path, output_directory, &bundle_path)?;
    verify_bundle(&bundle_path, &installed.id, &installed.version)?;
    let metadata = fs::metadata(&bundle_path).map_err(|source| PackError::Io {
        path: bundle_path.clone(),
        source,
    })?;
    let digest = hash_file(&bundle_path)?;
    let entry = RegistryComponent {
        name: short_name,
        id: installed.id,
        version: installed.version,
        requires_davis: requirement,
        bundle: Bundle {
            url: bundle_name,
            size: metadata.len(),
            blake3: format!("blake3:{digest}"),
        },
    };
    let entry_path = output_directory.join(format!("{stem}.entry.json"));
    write_json_atomic(&entry_path, &entry)?;
    Ok(PackedComponent {
        bundle_path,
        entry_path,
        entry,
    })
}

fn verify_bundle(
    bundle_path: &Path,
    expected_id: &str,
    expected_version: &str,
) -> Result<(), PackError> {
    let verification = tempfile::tempdir().map_err(|source| PackError::Io {
        path: std::env::temp_dir(),
        source,
    })?;
    let extracted = verification.path().join("extracted");
    fs::create_dir(&extracted).map_err(|source| PackError::Io {
        path: extracted.clone(),
        source,
    })?;
    crate::component_registry::extract_bundle(bundle_path, &extracted)?;
    let installed = ComponentStore::new(verification.path().join("store")).install(&extracted)?;
    if installed.id != expected_id || installed.version != expected_version {
        return Err(PackError::IdentityMismatch {
            expected_id: expected_id.to_owned(),
            expected_version: expected_version.to_owned(),
            actual_id: installed.id,
            actual_version: installed.version,
        });
    }
    Ok(())
}

pub(crate) fn registry(entry_paths: &[PathBuf], output_path: &Path) -> Result<Registry, PackError> {
    if entry_paths.is_empty() {
        return Err(PackError::EmptyRegistry);
    }
    let mut entries = Vec::with_capacity(entry_paths.len());
    let mut identities = HashSet::new();
    let bundle_directory = output_path.parent().unwrap_or_else(|| Path::new("."));
    for path in entry_paths {
        let bytes = fs::read(path).map_err(|source| PackError::Io {
            path: path.clone(),
            source,
        })?;
        let entry: RegistryComponent = serde_json::from_slice(&bytes)?;
        validate_registry_entry(&entry)?;
        validate_registry_bundle(&entry, bundle_directory)?;
        let identity = format!("{}@{}", entry.id, entry.version);
        if !identities.insert(identity.clone()) {
            return Err(PackError::DuplicateEntry(identity));
        }
        entries.push(entry);
    }
    entries.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| version_for_sort(&left.version).cmp(&version_for_sort(&right.version)))
    });
    let registry = Registry {
        schema_version: 1,
        components: entries,
    };
    write_json_atomic(output_path, &registry)?;
    Ok(registry)
}

fn validate_registry_bundle(
    entry: &RegistryComponent,
    bundle_directory: &Path,
) -> Result<(), PackError> {
    let path = bundle_directory.join(&entry.bundle.url);
    let actual_size = fs::metadata(&path)
        .map_err(|source| PackError::Io {
            path: path.clone(),
            source,
        })?
        .len();
    if actual_size != entry.bundle.size {
        return Err(PackError::BundleSizeMismatch {
            path,
            expected: entry.bundle.size,
            actual: actual_size,
        });
    }
    let actual_digest = format!("blake3:{}", hash_file(&path)?);
    if actual_digest != entry.bundle.blake3 {
        return Err(PackError::BundleDigestMismatch {
            path,
            expected: entry.bundle.blake3.clone(),
            actual: actual_digest,
        });
    }
    Ok(())
}

fn write_deterministic_bundle(
    package_root: &Path,
    temporary_directory: &Path,
    destination: &Path,
) -> Result<(), PackError> {
    let mut files = Vec::new();
    for entry in WalkDir::new(package_root) {
        let entry = entry.map_err(|source| PackError::Walk {
            path: package_root.to_owned(),
            source,
        })?;
        if !entry.file_type().is_file() || entry.file_name() == INSTALL_RECORD {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(package_root)
            .map_err(|_| PackError::NonPortablePath(entry.path().to_owned()))?
            .to_owned();
        if !is_portable_relative_path(&relative) {
            return Err(PackError::NonPortablePath(relative));
        }
        files.push((relative, entry.path().to_owned()));
    }
    if files.is_empty() {
        return Err(PackError::EmptyPackage);
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));

    let mut temporary =
        NamedTempFile::new_in(temporary_directory).map_err(|source| PackError::Io {
            path: temporary_directory.to_owned(),
            source,
        })?;
    {
        let encoder = GzBuilder::new()
            .mtime(0)
            .write(temporary.as_file_mut(), Compression::best());
        let mut archive = tar::Builder::new(encoder);
        for (relative, path) in files {
            let mut file = File::open(&path).map_err(|source| PackError::Io {
                path: path.clone(),
                source,
            })?;
            let size = file
                .metadata()
                .map_err(|source| PackError::Io {
                    path: path.clone(),
                    source,
                })?
                .len();
            let mut header = tar::Header::new_gnu();
            header.set_size(size);
            header.set_mode(portable_mode(&path));
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header.set_entry_type(tar::EntryType::Regular);
            header.set_cksum();
            archive
                .append_data(&mut header, &relative, &mut file)
                .map_err(|source| PackError::Io {
                    path: destination.to_owned(),
                    source,
                })?;
        }
        let encoder = archive.into_inner().map_err(|source| PackError::Io {
            path: destination.to_owned(),
            source,
        })?;
        encoder.finish().map_err(|source| PackError::Io {
            path: destination.to_owned(),
            source,
        })?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| PackError::Io {
            path: destination.to_owned(),
            source,
        })?;
    temporary
        .persist(destination)
        .map_err(|error| PackError::Io {
            path: destination.to_owned(),
            source: error.error,
        })?;
    Ok(())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), PackError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| PackError::Io {
        path: parent.to_owned(),
        source,
    })?;
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut temporary = NamedTempFile::new_in(parent).map_err(|source| PackError::Io {
        path: parent.to_owned(),
        source,
    })?;
    temporary
        .write_all(&bytes)
        .map_err(|source| PackError::Io {
            path: path.to_owned(),
            source,
        })?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| PackError::Io {
            path: path.to_owned(),
            source,
        })?;
    temporary.persist(path).map_err(|error| PackError::Io {
        path: path.to_owned(),
        source: error.error,
    })?;
    Ok(())
}

fn validate_registry_entry(entry: &RegistryComponent) -> Result<(), PackError> {
    validate_name(&entry.name)?;
    Version::parse(&entry.version).map_err(|source| PackError::InvalidVersion {
        value: entry.version.clone(),
        source,
    })?;
    VersionReq::parse(&entry.requires_davis).map_err(|source| PackError::InvalidVersion {
        value: entry.requires_davis.clone(),
        source,
    })?;
    let path = Path::new(&entry.bundle.url);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(PackError::InvalidBundleUrl(entry.bundle.url.clone()));
    }
    let digest = entry.bundle.blake3.strip_prefix("blake3:");
    if digest.is_none_or(|value| value.parse::<blake3::Hash>().is_err()) {
        return Err(PackError::InvalidDigest(entry.bundle.blake3.clone()));
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), PackError> {
    if name.is_empty()
        || !name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
    {
        return Err(PackError::InvalidName(name.to_owned()));
    }
    Ok(())
}

fn validate_requirement(requirement: &str) -> Result<(), PackError> {
    VersionReq::parse(requirement).map_err(|source| PackError::InvalidVersion {
        value: requirement.to_owned(),
        source,
    })?;
    Ok(())
}

fn version_for_sort(value: &str) -> Version {
    Version::parse(value).expect("registry entry version was validated")
}

fn is_portable_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && path.to_str().is_some()
}

fn hash_file(path: &Path) -> Result<blake3::Hash, PackError> {
    let mut file = File::open(path).map_err(|source| PackError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| PackError::Io {
            path: path.to_owned(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize())
}

#[cfg(unix)]
fn portable_mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).map_or(0o644, |metadata| {
        if metadata.permissions().mode() & 0o111 == 0 {
            0o644
        } else {
            0o755
        }
    })
}

#[cfg(not(unix))]
fn portable_mode(path: &Path) -> u32 {
    if path.extension().is_some_and(|extension| extension == "exe") {
        0o755
    } else {
        0o644
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use super::{pack, registry, PackError};

    #[test]
    fn creates_deterministic_bundle_and_registry() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir_all(source.join("schemas")).unwrap();
        fs::write(
            source.join("component-manifest.yaml"),
            r"api_version: davis.component/v1alpha1
id: example/native
name: Example Native
version: 1.2.3
requires_davis: '>=0.3.0'
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
        let first = temporary.path().join("first");
        let second = temporary.path().join("second");
        let packed_first = pack(&source, &first, None, None).unwrap();
        let packed_second = pack(&source, &second, None, None).unwrap();

        assert_eq!(
            fs::read(&packed_first.bundle_path).unwrap(),
            fs::read(&packed_second.bundle_path).unwrap()
        );
        let registry_path = first.join("component-registry.json");
        let entry_path = packed_first.entry_path.clone();
        let generated = registry(std::slice::from_ref(&entry_path), &registry_path).unwrap();
        assert_eq!(generated.schema_version, 1);
        assert_eq!(generated.components[0].name, "native");
        assert!(registry_path.is_file());

        fs::OpenOptions::new()
            .append(true)
            .open(&packed_first.bundle_path)
            .unwrap()
            .write_all(b"corrupt")
            .unwrap();
        assert!(matches!(
            registry(&[entry_path], &first.join("corrupt-registry.json")),
            Err(PackError::BundleSizeMismatch { .. })
        ));
    }
}
