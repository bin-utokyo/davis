use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use davis_model_api::ModelManifest;
use flate2::read::GzDecoder;
use futures::StreamExt;
use reqwest::{Client, Url};
use semver::{Version, VersionReq};
use serde::Deserialize;
use tempfile::TempDir;
use thiserror::Error;

const DEFAULT_REGISTRY_URL: &str =
    "https://github.com/bin-utokyo/davis/releases/latest/download/component-registry.json";
const MAX_REGISTRY_BYTES: u64 = 2 * 1024 * 1024;
const MAX_BUNDLE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_EXPANDED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("invalid component registry URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("insecure component URL is not allowed: {0}")]
    InsecureUrl(Url),
    #[error("failed to retrieve component metadata or package: {0}")]
    Request(#[from] reqwest::Error),
    #[error("component registry returned HTTP {0}")]
    Http(reqwest::StatusCode),
    #[error("component registry is larger than the {MAX_REGISTRY_BYTES} byte limit")]
    RegistryTooLarge,
    #[error("unsupported component registry schema version: {0}")]
    UnsupportedSchema(u32),
    #[error("invalid component registry JSON: {0}")]
    InvalidRegistry(#[from] serde_json::Error),
    #[error("invalid semantic version `{value}` in component registry: {source}")]
    InvalidVersion {
        value: String,
        source: semver::Error,
    },
    #[error("official component `{reference}` was not found in the registry")]
    NotFound { reference: String },
    #[error("no version of `{reference}` is compatible with Davis v{davis_version}")]
    NoCompatibleVersion {
        reference: String,
        davis_version: String,
    },
    #[error("component bundle is larger than the {MAX_BUNDLE_BYTES} byte limit")]
    BundleTooLarge,
    #[error("component bundle size mismatch: expected {expected}, received {actual}")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("component bundle digest mismatch: expected {expected}, received {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error("component bundle digest must use a valid blake3 value: {0}")]
    InvalidDigest(String),
    #[error("component bundle contains an unsafe path: {0}")]
    UnsafeArchivePath(PathBuf),
    #[error("component bundle contains unsupported entry type at {0}")]
    UnsupportedArchiveEntry(PathBuf),
    #[error("expanded component exceeds the {MAX_EXPANDED_BYTES} byte limit")]
    ExpandedTooLarge,
    #[error("component bundle must contain model-manifest.yaml at its root")]
    MissingManifest,
    #[error(
        "downloaded component identity `{actual_id}` `{actual_version}` does not match registry `{expected_id}` `{expected_version}`"
    )]
    IdentityMismatch {
        expected_id: String,
        expected_version: String,
        actual_id: String,
        actual_version: String,
    },
    #[error(transparent)]
    Contract(#[from] davis_model_api::ContractError),
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid tar archive: {0}")]
    Archive(std::io::Error),
}

#[derive(Debug, Clone, Deserialize)]
struct Registry {
    schema_version: u32,
    components: Vec<RegistryComponent>,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryComponent {
    name: String,
    id: String,
    version: String,
    requires_davis: String,
    bundle: Bundle,
}

#[derive(Debug, Clone, Deserialize)]
struct Bundle {
    url: String,
    size: u64,
    blake3: String,
}

pub struct DownloadedComponent {
    directory: PathBuf,
    selected: RegistryComponent,
    _temporary: TempDir,
}

impl DownloadedComponent {
    pub fn path(&self) -> &Path {
        &self.directory
    }

    pub fn id(&self) -> &str {
        &self.selected.id
    }

    pub fn version(&self) -> &str {
        &self.selected.version
    }
}

pub async fn download(
    reference: &str,
    requested_version: Option<&str>,
    registry_override: Option<&str>,
) -> Result<DownloadedComponent, RegistryError> {
    let registry_url = registry_url(registry_override)?;
    require_secure_url(&registry_url)?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_mins(2))
        .build()?;
    let response = client.get(registry_url.clone()).send().await?;
    require_secure_url(response.url())?;
    if !response.status().is_success() {
        return Err(RegistryError::Http(response.status()));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_REGISTRY_BYTES)
    {
        return Err(RegistryError::RegistryTooLarge);
    }
    let bytes = response.bytes().await?;
    if bytes.len() as u64 > MAX_REGISTRY_BYTES {
        return Err(RegistryError::RegistryTooLarge);
    }
    let registry: Registry = serde_json::from_slice(&bytes)?;
    if registry.schema_version != 1 {
        return Err(RegistryError::UnsupportedSchema(registry.schema_version));
    }
    let selected = select_component(&registry, reference, requested_version)?;
    let bundle_url = registry_url.join(&selected.bundle.url)?;
    require_secure_url(&bundle_url)?;
    let temporary = tempfile::tempdir().map_err(|source| RegistryError::Io {
        path: std::env::temp_dir(),
        source,
    })?;
    let archive_path = temporary.path().join("component.tar.gz");
    download_bundle(&client, &bundle_url, &selected.bundle, &archive_path).await?;
    let directory = temporary.path().join("component");
    fs::create_dir(&directory).map_err(|source| RegistryError::Io {
        path: directory.clone(),
        source,
    })?;
    extract_bundle(&archive_path, &directory)?;
    let manifest_path = directory.join("model-manifest.yaml");
    if !manifest_path.is_file() {
        return Err(RegistryError::MissingManifest);
    }
    let manifest = ModelManifest::read(&manifest_path)?;
    if manifest.id != selected.id || manifest.version != selected.version {
        return Err(RegistryError::IdentityMismatch {
            expected_id: selected.id,
            expected_version: selected.version,
            actual_id: manifest.id,
            actual_version: manifest.version,
        });
    }
    Ok(DownloadedComponent {
        directory,
        selected,
        _temporary: temporary,
    })
}

fn registry_url(override_url: Option<&str>) -> Result<Url, RegistryError> {
    let value = override_url
        .map(str::to_owned)
        .or_else(|| std::env::var("DAVIS_COMPONENT_REGISTRY_URL").ok())
        .unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_owned());
    Ok(Url::parse(&value)?)
}

fn require_secure_url(url: &Url) -> Result<(), RegistryError> {
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    if url.scheme() == "https" || url.scheme() == "http" && local {
        Ok(())
    } else {
        Err(RegistryError::InsecureUrl(url.clone()))
    }
}

fn select_component(
    registry: &Registry,
    reference: &str,
    requested_version: Option<&str>,
) -> Result<RegistryComponent, RegistryError> {
    let current = Version::parse(env!("CARGO_PKG_VERSION")).map_err(|source| {
        RegistryError::InvalidVersion {
            value: env!("CARGO_PKG_VERSION").to_owned(),
            source,
        }
    })?;
    let matching: Vec<&RegistryComponent> = registry
        .components
        .iter()
        .filter(|component| component.name == reference || component.id == reference)
        .filter(|component| requested_version.is_none_or(|version| component.version == version))
        .collect();
    if matching.is_empty() {
        return Err(RegistryError::NotFound {
            reference: reference.to_owned(),
        });
    }
    let mut compatible = Vec::new();
    for component in matching {
        let version =
            Version::parse(&component.version).map_err(|source| RegistryError::InvalidVersion {
                value: component.version.clone(),
                source,
            })?;
        let requirement = VersionReq::parse(&component.requires_davis).map_err(|source| {
            RegistryError::InvalidVersion {
                value: component.requires_davis.clone(),
                source,
            }
        })?;
        if requirement.matches(&current) {
            compatible.push((version, component));
        }
    }
    compatible.sort_by(|left, right| left.0.cmp(&right.0));
    compatible
        .pop()
        .map(|(_, component)| component.clone())
        .ok_or_else(|| RegistryError::NoCompatibleVersion {
            reference: reference.to_owned(),
            davis_version: current.to_string(),
        })
}

async fn download_bundle(
    client: &Client,
    url: &Url,
    bundle: &Bundle,
    destination: &Path,
) -> Result<(), RegistryError> {
    if bundle.size > MAX_BUNDLE_BYTES {
        return Err(RegistryError::BundleTooLarge);
    }
    let expected_digest = parse_digest(&bundle.blake3)?;
    let response = client.get(url.clone()).send().await?;
    require_secure_url(response.url())?;
    if !response.status().is_success() {
        return Err(RegistryError::Http(response.status()));
    }
    if response
        .content_length()
        .is_some_and(|length| length != bundle.size || length > MAX_BUNDLE_BYTES)
    {
        return Err(RegistryError::SizeMismatch {
            expected: bundle.size,
            actual: response.content_length().unwrap_or_default(),
        });
    }
    let mut file = File::create(destination).map_err(|source| RegistryError::Io {
        path: destination.to_owned(),
        source,
    })?;
    let mut stream = response.bytes_stream();
    let mut size = 0_u64;
    let mut hasher = blake3::Hasher::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        size = size.saturating_add(chunk.len() as u64);
        if size > bundle.size || size > MAX_BUNDLE_BYTES {
            return Err(RegistryError::BundleTooLarge);
        }
        hasher.update(&chunk);
        file.write_all(&chunk).map_err(|source| RegistryError::Io {
            path: destination.to_owned(),
            source,
        })?;
    }
    if size != bundle.size {
        return Err(RegistryError::SizeMismatch {
            expected: bundle.size,
            actual: size,
        });
    }
    let actual = hasher.finalize();
    if actual != expected_digest {
        return Err(RegistryError::DigestMismatch {
            expected: bundle.blake3.clone(),
            actual: format!("blake3:{actual}"),
        });
    }
    Ok(())
}

fn parse_digest(value: &str) -> Result<blake3::Hash, RegistryError> {
    value
        .strip_prefix("blake3:")
        .ok_or_else(|| RegistryError::InvalidDigest(value.to_owned()))?
        .parse()
        .map_err(|_| RegistryError::InvalidDigest(value.to_owned()))
}

fn extract_bundle(archive_path: &Path, destination: &Path) -> Result<(), RegistryError> {
    let file = File::open(archive_path).map_err(|source| RegistryError::Io {
        path: archive_path.to_owned(),
        source,
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let entries = archive.entries().map_err(RegistryError::Archive)?;
    let mut expanded = 0_u64;
    for entry in entries {
        let mut entry = entry.map_err(RegistryError::Archive)?;
        let path = entry.path().map_err(RegistryError::Archive)?.into_owned();
        let entry_type = entry.header().entry_type();
        if is_archive_root(&path) && entry_type.is_dir() {
            continue;
        }
        if !is_safe_archive_path(&path) {
            return Err(RegistryError::UnsafeArchivePath(path));
        }
        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(RegistryError::UnsupportedArchiveEntry(path));
        }
        expanded = expanded.saturating_add(entry.header().size().unwrap_or(0));
        if expanded > MAX_EXPANDED_BYTES {
            return Err(RegistryError::ExpandedTooLarge);
        }
        if !entry
            .unpack_in(destination)
            .map_err(RegistryError::Archive)?
        {
            return Err(RegistryError::UnsafeArchivePath(path));
        }
    }
    Ok(())
}

fn is_safe_archive_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
        && path
            .components()
            .any(|component| matches!(component, Component::Normal(_)))
}

fn is_archive_root(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| component == Component::CurDir)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::Path;

    use flate2::write::GzEncoder;
    use flate2::Compression;

    use super::{download, is_safe_archive_path, select_component, Registry};

    #[test]
    fn selects_latest_compatible_component() {
        let registry: Registry = serde_json::from_str(
            r#"{
              "schema_version": 1,
              "components": [
                {"name":"mnl","id":"davis/mnl","version":"0.1.0","requires_davis":">=0.3.0, <0.4.0","bundle":{"url":"mnl-0.1.0.tar.gz","size":1,"blake3":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
                {"name":"mnl","id":"davis/mnl","version":"0.2.0","requires_davis":">=0.3.0, <0.4.0","bundle":{"url":"mnl-0.2.0.tar.gz","size":1,"blake3":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
                {"name":"mnl","id":"davis/mnl","version":"1.0.0","requires_davis":">=1.0.0","bundle":{"url":"mnl-1.0.0.tar.gz","size":1,"blake3":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}
              ]
            }"#,
        )
        .unwrap();

        let selected = select_component(&registry, "mnl", None).unwrap();
        assert_eq!(selected.version, "0.2.0");
        let exact = select_component(&registry, "davis/mnl", Some("0.1.0")).unwrap();
        assert_eq!(exact.version, "0.1.0");
    }

    #[test]
    fn rejects_unsafe_archive_paths() {
        assert!(!is_safe_archive_path(Path::new("../escape")));
        assert!(!is_safe_archive_path(Path::new("nested/../../escape")));
        assert!(!is_safe_archive_path(Path::new("/absolute")));
        assert!(is_safe_archive_path(Path::new("schemas/config.json")));
        assert!(is_safe_archive_path(Path::new("./schemas/config.json")));
    }

    #[tokio::test]
    async fn downloads_and_verifies_a_registry_component() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir_all(source.join("schemas")).unwrap();
        fs::write(
            source.join("model-manifest.yaml"),
            r"api_version: davis.model/v1alpha1
id: davis/test
name: Davis Test
version: 0.1.0
runtime:
  kind: native
  command: [test]
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

        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(encoder);
        archive
            .append_path_with_name(source.join("model-manifest.yaml"), "model-manifest.yaml")
            .unwrap();
        archive
            .append_path_with_name(source.join("schemas/config.json"), "schemas/config.json")
            .unwrap();
        let bundle = archive.into_inner().unwrap().finish().unwrap();
        let digest = blake3::hash(&bundle);

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let registry = serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "components": [{
                "name": "test",
                "id": "davis/test",
                "version": "0.1.0",
                "requires_davis": ">=0.3.0, <0.4.0",
                "bundle": {
                    "url": format!("http://{address}/component.tar.gz"),
                    "size": bundle.len(),
                    "blake3": format!("blake3:{digest}")
                }
            }]
        }))
        .unwrap();
        let server = std::thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let mut stream = stream.unwrap();
                let mut request = [0_u8; 4096];
                let read = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                let body = if request.starts_with("GET /registry.json ") {
                    &registry
                } else {
                    &bundle
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });

        let registry_url = format!("http://{address}/registry.json");
        let downloaded = download("test", None, Some(&registry_url)).await.unwrap();
        assert_eq!(downloaded.id(), "davis/test");
        assert_eq!(downloaded.version(), "0.1.0");
        assert!(downloaded.path().join("model-manifest.yaml").is_file());
        server.join().unwrap();
    }
}
