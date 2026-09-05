//! Local-first analysis-plan compiler and model runtime.

mod components;
mod inspect;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub use components::{
    user_data_directory, ComponentStore, ComponentStoreError, InstalledComponent,
};
use davis_model_api::{
    AnalysisPlan, ComponentManifest, InputSource, ResolvedComponent, ResolvedFile, ResolvedInput,
    RunRequest, RunResult, RunStatus, RESULT_API_VERSION, RUN_API_VERSION,
};
use davis_model_runner::{run_component, RunnerError};
pub use inspect::{inspect_csv, ColumnProfile, CsvProfile, InspectError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Contract(#[from] davis_model_api::ContractError),
    #[error(transparent)]
    Runner(#[from] RunnerError),
    #[error("failed to access {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("component `{id}` version `{version}` was not found; searched {roots:?}")]
    ModelNotFound {
        id: String,
        version: String,
        roots: Vec<PathBuf>,
    },
    #[error("component `{id}` does not support operation `{operation}`")]
    UnsupportedOperation { id: String, operation: String },
    #[error("required component input `{0}` was not provided")]
    MissingInput(String),
    #[error("analysis input `{0}` is not declared by the component manifest")]
    UnexpectedInput(String),
    #[error("component input `{name}` does not accept media type `{media_type}`")]
    UnsupportedMediaType { name: String, media_type: String },
    #[error("catalog inputs are not implemented in this prototype: {dataset_id}/{file_id}")]
    CatalogNotImplemented { dataset_id: String, file_id: String },
    #[error("invalid run ID `{0}`")]
    InvalidRunId(String),
    #[error("run result does not exist: {0}")]
    MissingRunResult(PathBuf),
    #[error("run `{run_id}` does not contain artifact `{artifact}`")]
    MissingRunArtifact { run_id: String, artifact: String },
    #[error("run artifact `{run_id}/{artifact}` has an unsafe path: {path}")]
    UnsafeRunArtifact {
        run_id: String,
        artifact: String,
        path: PathBuf,
    },
    #[error("run artifact `{run_id}/{artifact}` failed integrity verification: {message}")]
    RunArtifactIntegrity {
        run_id: String,
        artifact: String,
        message: String,
    },
    #[error("local input does not exist: {0}")]
    MissingLocalInput(PathBuf),
    #[error("failed to serialize run record: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("invalid JSON schema {path}: {message}")]
    InvalidConfigSchema { path: PathBuf, message: String },
    #[error("component configuration does not match {path}: {errors}")]
    InvalidModelConfig { path: PathBuf, errors: String },
    #[error("runtime generated an invalid output directory: {0}")]
    InvalidOutputDirectory(PathBuf),
}

#[derive(Debug, Clone)]
pub struct ValidatedPlan {
    pub plan: AnalysisPlan,
    pub plan_path: PathBuf,
    pub manifest: ComponentManifest,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedRun {
    pub request: RunRequest,
    pub plan_path: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedRun {
    pub run_directory: PathBuf,
    pub request: RunRequest,
    pub result: RunResult,
}

/// Validates an analysis plan and resolves its declared model component.
///
/// # Errors
///
/// Returns an error when a contract, component, input declaration, operation,
/// or component-specific JSON Schema is invalid.
pub fn validate_plan(repository: &Path, plan_path: &Path) -> Result<ValidatedPlan, RuntimeError> {
    let plan_path = absolute_path(plan_path)?;
    let plan = AnalysisPlan::read(&plan_path)?;
    let (manifest_path, manifest) = find_manifest(
        repository,
        &plan.component.component,
        &plan.component.version,
    )?;
    if !manifest.operations.contains(&plan.component.operation) {
        return Err(RuntimeError::UnsupportedOperation {
            id: manifest.id.clone(),
            operation: plan.component.operation.clone(),
        });
    }
    for input in &manifest.inputs {
        if input.required && !plan.inputs.contains_key(&input.name) {
            return Err(RuntimeError::MissingInput(input.name.clone()));
        }
    }
    for name in plan.inputs.keys() {
        if !manifest.inputs.iter().any(|input| &input.name == name) {
            return Err(RuntimeError::UnexpectedInput(name.clone()));
        }
    }
    let config_schema = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&manifest.config_schema);
    if !config_schema.is_file() {
        return Err(RuntimeError::MissingLocalInput(config_schema));
    }
    validate_component_config(&config_schema, &plan.config)?;
    Ok(ValidatedPlan {
        plan,
        plan_path,
        manifest,
        manifest_path,
    })
}

/// Resolves local inputs and compiles an analysis plan into a `RunRequest`.
///
/// # Errors
///
/// Returns an error when validation fails, an input cannot be resolved or
/// hashed, or its media type is not accepted by the component.
pub fn plan_run(
    repository: &Path,
    plan_path: &Path,
    run_root: &Path,
) -> Result<PlannedRun, RuntimeError> {
    let validated = validate_plan(repository, plan_path)?;
    let run_root = absolute_path(run_root)?;
    let plan_directory = validated
        .plan_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let mut inputs = BTreeMap::new();
    for (name, source) in &validated.plan.inputs {
        let resolved = match source {
            InputSource::Local { path, .. } => {
                let path = if path.is_absolute() {
                    path.clone()
                } else {
                    plan_directory.join(path)
                };
                let path = absolute_path(&path)?;
                if !path.is_file() {
                    return Err(RuntimeError::MissingLocalInput(path));
                }
                let digest = hash_file(&path)?;
                let metadata = fs::metadata(&path).map_err(|source| RuntimeError::Io {
                    path: path.clone(),
                    source,
                })?;
                ResolvedFile {
                    media_type: media_type(&path).to_owned(),
                    path,
                    object_id: format!("blake3:{digest}"),
                    size: metadata.len(),
                }
            }
            InputSource::Catalog {
                dataset_id,
                file_id,
                ..
            } => {
                return Err(RuntimeError::CatalogNotImplemented {
                    dataset_id: dataset_id.clone(),
                    file_id: file_id.clone(),
                });
            }
            InputSource::RunArtifact { run_id, artifact } => {
                resolve_run_artifact(&run_root, run_id, artifact)?
            }
        };
        let declaration = validated
            .manifest
            .inputs
            .iter()
            .find(|input| input.name == *name)
            .ok_or_else(|| RuntimeError::UnexpectedInput(name.clone()))?;
        if !declaration.media_types.contains(&resolved.media_type) {
            return Err(RuntimeError::UnsupportedMediaType {
                name: name.clone(),
                media_type: resolved.media_type,
            });
        }
        inputs.insert(
            name.clone(),
            ResolvedInput {
                source: source.clone(),
                resolved,
            },
        );
    }
    let run_id = new_run_id();
    let run_directory = run_root.join(&run_id);
    let output_directory = run_directory.join("artifacts");
    let component_directory = validated
        .manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let component_digest = hash_component(component_directory)?;
    Ok(PlannedRun {
        request: RunRequest {
            api_version: RUN_API_VERSION.to_owned(),
            run_id,
            operation: validated.plan.component.operation.clone(),
            component: ResolvedComponent {
                id: validated.manifest.id,
                version: validated.manifest.version,
                kind: validated.manifest.kind,
                manifest_path: validated.manifest_path.clone(),
                source_digest: format!("blake3:{component_digest}"),
            },
            inputs,
            config: validated.plan.config,
            output_directory,
        },
        plan_path: validated.plan_path,
        manifest_path: validated.manifest_path,
    })
}

fn resolve_run_artifact(
    run_root: &Path,
    run_id: &str,
    artifact_name: &str,
) -> Result<ResolvedFile, RuntimeError> {
    let run_id_path = Path::new(run_id);
    if run_id_path.components().count() != 1
        || !matches!(run_id_path.components().next(), Some(Component::Normal(_)))
    {
        return Err(RuntimeError::InvalidRunId(run_id.to_owned()));
    }
    let run_directory = run_root.join(run_id);
    let result_path = run_directory.join("result.json");
    let result = read_run_result(&result_path, run_id, artifact_name)?;
    let descriptor = result
        .artifacts
        .get(artifact_name)
        .or_else(|| result.extensions.get(artifact_name))
        .ok_or_else(|| RuntimeError::MissingRunArtifact {
            run_id: run_id.to_owned(),
            artifact: artifact_name.to_owned(),
        })?;
    if descriptor.path.is_absolute()
        || descriptor
            .path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(RuntimeError::UnsafeRunArtifact {
            run_id: run_id.to_owned(),
            artifact: artifact_name.to_owned(),
            path: descriptor.path.clone(),
        });
    }
    let output_directory = run_directory.join("artifacts");
    let path = output_directory.join(&descriptor.path);
    if !path.is_file() {
        return Err(RuntimeError::MissingLocalInput(path));
    }
    let canonical_output =
        fs::canonicalize(&output_directory).map_err(|source| RuntimeError::Io {
            path: output_directory.clone(),
            source,
        })?;
    let canonical_path = fs::canonicalize(&path).map_err(|source| RuntimeError::Io {
        path: path.clone(),
        source,
    })?;
    if !canonical_path.starts_with(&canonical_output) {
        return Err(RuntimeError::UnsafeRunArtifact {
            run_id: run_id.to_owned(),
            artifact: artifact_name.to_owned(),
            path,
        });
    }
    let metadata = fs::metadata(&canonical_path).map_err(|source| RuntimeError::Io {
        path: canonical_path.clone(),
        source,
    })?;
    let digest = format!("blake3:{}", hash_file(&canonical_path)?);
    verify_artifact_integrity(run_id, artifact_name, descriptor, metadata.len(), &digest)?;
    Ok(ResolvedFile {
        path: canonical_path,
        object_id: digest,
        size: metadata.len(),
        media_type: descriptor.media_type.clone(),
    })
}

fn read_run_result(
    result_path: &Path,
    run_id: &str,
    artifact_name: &str,
) -> Result<RunResult, RuntimeError> {
    if !result_path.is_file() {
        return Err(RuntimeError::MissingRunResult(result_path.to_owned()));
    }
    let result_bytes = fs::read(result_path).map_err(|source| RuntimeError::Io {
        path: result_path.to_owned(),
        source,
    })?;
    let result: RunResult = serde_json::from_slice(&result_bytes)?;
    if result.api_version != RESULT_API_VERSION || result.status != RunStatus::Succeeded {
        return Err(RuntimeError::RunArtifactIntegrity {
            run_id: run_id.to_owned(),
            artifact: artifact_name.to_owned(),
            message: format!(
                "result has API version `{}` and status `{:?}`",
                result.api_version, result.status
            ),
        });
    }
    if result.run_id != run_id {
        return Err(RuntimeError::RunArtifactIntegrity {
            run_id: run_id.to_owned(),
            artifact: artifact_name.to_owned(),
            message: format!("result records run ID `{}`", result.run_id),
        });
    }
    Ok(result)
}

fn verify_artifact_integrity(
    run_id: &str,
    artifact_name: &str,
    descriptor: &davis_model_api::ArtifactDescriptor,
    actual_size: u64,
    actual_digest: &str,
) -> Result<(), RuntimeError> {
    if let Some(expected) = descriptor.size {
        if expected != actual_size {
            return Err(RuntimeError::RunArtifactIntegrity {
                run_id: run_id.to_owned(),
                artifact: artifact_name.to_owned(),
                message: format!("recorded size {expected} differs from {actual_size}"),
            });
        }
    }
    if let Some(expected) = &descriptor.object_id {
        if expected != actual_digest {
            return Err(RuntimeError::RunArtifactIntegrity {
                run_id: run_id.to_owned(),
                artifact: artifact_name.to_owned(),
                message: format!("recorded digest does not match {actual_digest}"),
            });
        }
    }
    Ok(())
}

/// Compiles and executes one local analysis plan and persists its run record.
///
/// # Errors
///
/// Returns an error when planning, process execution, result validation, or
/// run-record persistence fails.
pub fn execute_plan(
    repository: &Path,
    plan_path: &Path,
    run_root: &Path,
) -> Result<CompletedRun, RuntimeError> {
    let planned = plan_run(repository, plan_path, run_root)?;
    let run_directory = planned
        .request
        .output_directory
        .parent()
        .ok_or_else(|| {
            RuntimeError::InvalidOutputDirectory(planned.request.output_directory.clone())
        })?
        .to_owned();
    fs::create_dir_all(&planned.request.output_directory).map_err(|source| RuntimeError::Io {
        path: planned.request.output_directory.clone(),
        source,
    })?;
    fs::copy(&planned.plan_path, run_directory.join("model.yaml")).map_err(|source| {
        RuntimeError::Io {
            path: run_directory.join("model.yaml"),
            source,
        }
    })?;
    let manifest = ComponentManifest::read(&planned.manifest_path)?;
    let output = run_component(
        &planned.manifest_path,
        &manifest,
        &planned.request,
        &run_directory,
    )?;
    let completed = CompletedRun {
        run_directory: run_directory.clone(),
        request: planned.request,
        result: output.result,
    };
    let run_json = serde_json::to_vec_pretty(&completed)?;
    fs::write(run_directory.join("run.json"), run_json).map_err(|source| RuntimeError::Io {
        path: run_directory.join("run.json"),
        source,
    })?;
    let result_json = serde_json::to_vec_pretty(&completed.result)?;
    fs::write(run_directory.join("result.json"), result_json).map_err(|source| {
        RuntimeError::Io {
            path: run_directory.join("result.json"),
            source,
        }
    })?;
    Ok(completed)
}

fn find_manifest(
    repository: &Path,
    id: &str,
    version: &str,
) -> Result<(PathBuf, ComponentManifest), RuntimeError> {
    let mut roots = vec![repository.join("components")];
    if let Ok(store) = ComponentStore::for_user() {
        if !roots.iter().any(|root| root == store.root()) {
            roots.push(store.root().to_owned());
        }
    }
    for root in &roots {
        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|entry| !is_temporary_install_entry(entry))
            .filter_map(Result::ok)
        {
            let is_manifest = entry.file_name() == davis_model_api::COMPONENT_MANIFEST_FILENAME
                || entry.file_name() == davis_model_api::LEGACY_MODEL_MANIFEST_FILENAME;
            if !is_manifest {
                continue;
            }
            let Some(directory) = entry.path().parent() else {
                continue;
            };
            if let Ok((manifest_path, manifest)) = ComponentManifest::read_from_directory(directory)
            {
                if manifest.id == id && manifest.version == version {
                    return Ok((absolute_path(&manifest_path)?, manifest));
                }
            }
        }
    }
    Err(RuntimeError::ModelNotFound {
        id: id.to_owned(),
        version: version.to_owned(),
        roots,
    })
}

fn is_temporary_install_entry(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with(".install-"))
}

fn absolute_path(path: &Path) -> Result<PathBuf, RuntimeError> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        let current = std::env::current_dir().map_err(|source| RuntimeError::Io {
            path: PathBuf::from("."),
            source,
        })?;
        Ok(current.join(path))
    }
}

fn media_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("csv") => "text/csv",
        Some("parquet") => "application/vnd.apache.parquet",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

fn validate_component_config(
    schema_path: &Path,
    config: &serde_json::Value,
) -> Result<(), RuntimeError> {
    let bytes = fs::read(schema_path).map_err(|source| RuntimeError::Io {
        path: schema_path.to_owned(),
        source,
    })?;
    let schema: serde_json::Value = serde_json::from_slice(&bytes)?;
    let validator =
        jsonschema::validator_for(&schema).map_err(|error| RuntimeError::InvalidConfigSchema {
            path: schema_path.to_owned(),
            message: error.to_string(),
        })?;
    let errors: Vec<String> = validator
        .iter_errors(config)
        .take(20)
        .map(|error| format!("{}: {}", error.instance_path, error))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(RuntimeError::InvalidModelConfig {
            path: schema_path.to_owned(),
            errors: errors.join("; "),
        })
    }
}

fn new_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("run_{millis}_{}", std::process::id())
}

fn hash_file(path: &Path) -> Result<blake3::Hash, RuntimeError> {
    let mut file = File::open(path).map_err(|source| RuntimeError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = blake3::Hasher::new();
    io::copy(&mut file, &mut hasher).map_err(|source| RuntimeError::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(hasher.finalize())
}

fn hash_component(directory: &Path) -> Result<blake3::Hash, RuntimeError> {
    let mut files: Vec<PathBuf> = WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            !entry.path().components().any(|component| {
                matches!(
                    component.as_os_str().to_str(),
                    Some(
                        ".venv" | "__pycache__" | ".pytest_cache" | ".git" | "target" | "examples"
                    )
                )
            })
        })
        .filter(|entry| entry.file_name() != ".davis-install.json")
        .map(walkdir::DirEntry::into_path)
        .collect();
    files.sort();
    let mut hasher = blake3::Hasher::new();
    for path in files {
        let relative = path.strip_prefix(directory).unwrap_or(&path);
        hasher.update(relative.to_string_lossy().as_bytes());
        let mut file = File::open(&path).map_err(|source| RuntimeError::Io {
            path: path.clone(),
            source,
        })?;
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|source| RuntimeError::Io {
                path: path.clone(),
                source,
            })?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use davis_model_api::InputSource;

    use super::{plan_run, RuntimeError};

    #[test]
    fn resolves_and_verifies_a_prior_run_artifact() {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("project");
        let component = repository.join("components/example/transform");
        fs::create_dir_all(component.join("schemas")).unwrap();
        fs::write(
            component.join("component-manifest.yaml"),
            r"api_version: davis.component/v1alpha1
id: example/transform
name: Example transform
version: 0.1.0
kind: transform
runtime:
  kind: native
  command: [true]
operations: [transform]
inputs:
  - name: table
    media_types: [text/csv]
config_schema: schemas/config.json
outputs:
  artifacts:
    table:
      media_types: [text/csv]
",
        )
        .unwrap();
        fs::write(component.join("schemas/config.json"), "{}").unwrap();

        let run_root = repository.join("davis-runs");
        let prior_output = run_root.join("prior/artifacts");
        fs::create_dir_all(&prior_output).unwrap();
        let artifact_path = prior_output.join("table.csv");
        fs::write(&artifact_path, "value\n1\n").unwrap();
        let bytes = fs::read(&artifact_path).unwrap();
        let digest = blake3::hash(&bytes);
        fs::write(
            run_root.join("prior/result.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "api_version": "davis.result/v1alpha1",
                "run_id": "prior",
                "status": "succeeded",
                "artifacts": {
                    "table": {
                        "path": "table.csv",
                        "media_type": "text/csv",
                        "size": bytes.len(),
                        "object_id": format!("blake3:{digest}")
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let plan_path = repository.join("transform.yaml");
        fs::write(
            &plan_path,
            r"api_version: davis.analysis/v1alpha1
name: chained
component:
  id: example/transform
  version: 0.1.0
  operation: transform
inputs:
  table:
    kind: run_artifact
    run_id: prior
    artifact: table
config: {}
",
        )
        .unwrap();

        let planned = plan_run(&repository, &plan_path, &run_root).unwrap();
        let input = planned.request.inputs.get("table").unwrap();
        assert!(matches!(input.source, InputSource::RunArtifact { .. }));
        assert_eq!(input.resolved.object_id, format!("blake3:{digest}"));

        fs::write(&artifact_path, "value\n2\n").unwrap();
        assert!(matches!(
            plan_run(&repository, &plan_path, &run_root),
            Err(RuntimeError::RunArtifactIntegrity { .. })
        ));
    }
}
