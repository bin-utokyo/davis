//! Process runner for language-independent Davis model components.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use davis_model_api::{ComponentManifest, RunRequest, RunResult, RunStatus, RESULT_API_VERSION};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("component runtime command is empty")]
    EmptyCommand,
    #[error("failed to prepare component run file {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize component request: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("failed to start component process `{command}`: {source}")]
    Spawn {
        command: String,
        source: std::io::Error,
    },
    #[error("component process exited with code {code}; see {stderr}")]
    ProcessFailed { code: i32, stderr: PathBuf },
    #[error("component did not write {0}")]
    MissingResult(PathBuf),
    #[error("component result uses run ID `{actual}` instead of `{expected}`")]
    WrongRunId { actual: String, expected: String },
    #[error("component result uses unsupported API version `{0}`")]
    WrongResultVersion(String),
    #[error("component result declared success but returned status `{0:?}`")]
    UnsuccessfulResult(RunStatus),
    #[error("artifact `{name}` escapes the output directory: {path}")]
    UnsafeArtifact { name: String, path: PathBuf },
    #[error("artifact `{name}` does not exist: {path}")]
    MissingArtifact { name: String, path: PathBuf },
    #[error("component did not return required artifact `{0}`")]
    MissingRequiredArtifact(String),
    #[error("component returned undeclared artifact `{0}`")]
    UndeclaredArtifact(String),
    #[error("artifact `{name}` uses media type `{media_type}`; expected one of {expected:?}")]
    UnsupportedArtifactMediaType {
        name: String,
        media_type: String,
        expected: Vec<String>,
    },
}

pub struct RunOutput {
    pub result: RunResult,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

/// Executes one model process and validates its declared artifacts.
///
/// # Errors
///
/// Returns an error when the process cannot be started, exits unsuccessfully,
/// omits a result, or declares an unsafe or missing artifact.
pub fn run_component(
    manifest_path: &Path,
    manifest: &ComponentManifest,
    request: &RunRequest,
    run_directory: &Path,
) -> Result<RunOutput, RunnerError> {
    let (program, arguments) = manifest
        .runtime
        .command
        .split_first()
        .ok_or(RunnerError::EmptyCommand)?;
    let component_directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let request_path = run_directory.join("request.json");
    let logs_directory = run_directory.join("logs");
    fs::create_dir_all(&logs_directory).map_err(|source| RunnerError::Io {
        path: logs_directory.clone(),
        source,
    })?;
    let request_json = serde_json::to_vec_pretty(request)?;
    fs::write(&request_path, request_json).map_err(|source| RunnerError::Io {
        path: request_path.clone(),
        source,
    })?;

    let stdout_path = logs_directory.join("stdout.log");
    let stderr_path = logs_directory.join("stderr.log");
    let stdout = File::create(&stdout_path).map_err(|source| RunnerError::Io {
        path: stdout_path.clone(),
        source,
    })?;
    let stderr = File::create(&stderr_path).map_err(|source| RunnerError::Io {
        path: stderr_path.clone(),
        source,
    })?;

    let status = Command::new(program)
        .args(arguments)
        .arg(&manifest.runtime.request_argument)
        .arg(&request_path)
        .current_dir(component_directory)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", std::env::var_os("HOME").unwrap_or_default())
        .env("LANG", std::env::var_os("LANG").unwrap_or_default())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .status()
        .map_err(|source| RunnerError::Spawn {
            command: program.clone(),
            source,
        })?;
    if !status.success() {
        return Err(RunnerError::ProcessFailed {
            code: status.code().unwrap_or(-1),
            stderr: stderr_path,
        });
    }

    let result_path = request.output_directory.join("run-result.json");
    if !result_path.is_file() {
        return Err(RunnerError::MissingResult(result_path));
    }
    let result_bytes = fs::read(&result_path).map_err(|source| RunnerError::Io {
        path: result_path.clone(),
        source,
    })?;
    let mut result: RunResult = serde_json::from_slice(&result_bytes)?;
    if result.api_version != RESULT_API_VERSION {
        return Err(RunnerError::WrongResultVersion(result.api_version));
    }
    if result.run_id != request.run_id {
        return Err(RunnerError::WrongRunId {
            actual: result.run_id,
            expected: request.run_id.clone(),
        });
    }
    if result.status != RunStatus::Succeeded {
        return Err(RunnerError::UnsuccessfulResult(result.status));
    }
    validate_artifacts(&request.output_directory, manifest, &mut result)?;
    Ok(RunOutput {
        result,
        stdout_path,
        stderr_path,
    })
}

fn validate_artifacts(
    output_directory: &Path,
    manifest: &ComponentManifest,
    result: &mut RunResult,
) -> Result<(), RunnerError> {
    for (name, declaration) in &manifest.outputs.artifacts {
        if declaration.required && !result.artifacts.contains_key(name) {
            return Err(RunnerError::MissingRequiredArtifact(name.clone()));
        }
    }
    if !manifest.outputs.artifacts.is_empty() {
        for (name, artifact) in &result.artifacts {
            let Some(declaration) = manifest.outputs.artifacts.get(name) else {
                return Err(RunnerError::UndeclaredArtifact(name.clone()));
            };
            if !declaration.media_types.contains(&artifact.media_type) {
                return Err(RunnerError::UnsupportedArtifactMediaType {
                    name: name.clone(),
                    media_type: artifact.media_type.clone(),
                    expected: declaration.media_types.clone(),
                });
            }
        }
    }
    for (name, artifact) in result
        .artifacts
        .iter_mut()
        .chain(result.extensions.iter_mut())
    {
        if artifact.path.is_absolute()
            || artifact
                .path
                .components()
                .any(|component| component == Component::ParentDir)
        {
            return Err(RunnerError::UnsafeArtifact {
                name: name.clone(),
                path: artifact.path.clone(),
            });
        }
        let path = output_directory.join(&artifact.path);
        let metadata = fs::metadata(&path).map_err(|_| RunnerError::MissingArtifact {
            name: name.clone(),
            path: path.clone(),
        })?;
        let digest = hash_file(&path)?;
        artifact.size = Some(metadata.len());
        artifact.object_id = Some(format!("blake3:{digest}"));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<blake3::Hash, RunnerError> {
    let mut file = File::open(path).map_err(|source| RunnerError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| RunnerError::Io {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    use davis_model_api::{
        ArtifactDescriptor, ComponentManifest, RunResult, RunStatus, RESULT_API_VERSION,
    };

    use super::{validate_artifacts, RunnerError};

    fn manifest() -> ComponentManifest {
        serde_json::from_value(serde_json::json!({
            "api_version": "davis.component/v1alpha1",
            "id": "example/transform",
            "name": "Example transform",
            "version": "0.1.0",
            "kind": "transform",
            "runtime": {"kind": "native", "command": ["example"]},
            "operations": ["transform"],
            "inputs": [],
            "config_schema": "schemas/config.json",
            "outputs": {
                "artifacts": {
                    "table": {"media_types": ["text/csv"], "required": true}
                }
            }
        }))
        .unwrap()
    }

    fn result(media_type: &str) -> RunResult {
        RunResult {
            api_version: RESULT_API_VERSION.to_owned(),
            run_id: "run".to_owned(),
            status: RunStatus::Succeeded,
            artifacts: BTreeMap::from([(
                "table".to_owned(),
                ArtifactDescriptor {
                    path: PathBuf::from("table.csv"),
                    media_type: media_type.to_owned(),
                    size: None,
                    object_id: None,
                },
            )]),
            extensions: BTreeMap::new(),
            error: None,
        }
    }

    #[test]
    fn enforces_declared_artifact_names_media_types_and_integrity() {
        let temporary = tempfile::tempdir().unwrap();
        fs::write(temporary.path().join("table.csv"), "value\n1\n").unwrap();
        let manifest = manifest();
        let mut valid = result("text/csv");
        validate_artifacts(temporary.path(), &manifest, &mut valid).unwrap();
        assert_eq!(valid.artifacts["table"].size, Some(8));
        assert!(valid.artifacts["table"]
            .object_id
            .as_deref()
            .is_some_and(|value| value.starts_with("blake3:")));

        let mut missing = result("text/csv");
        missing.artifacts.clear();
        assert!(matches!(
            validate_artifacts(temporary.path(), &manifest, &mut missing),
            Err(RunnerError::MissingRequiredArtifact(name)) if name == "table"
        ));

        let mut wrong_media_type = result("application/json");
        assert!(matches!(
            validate_artifacts(temporary.path(), &manifest, &mut wrong_media_type),
            Err(RunnerError::UnsupportedArtifactMediaType { .. })
        ));

        let mut undeclared = result("text/csv");
        let descriptor = undeclared.artifacts["table"].clone();
        undeclared.artifacts.insert("other".to_owned(), descriptor);
        assert!(matches!(
            validate_artifacts(temporary.path(), &manifest, &mut undeclared),
            Err(RunnerError::UndeclaredArtifact(name)) if name == "other"
        ));
    }
}
