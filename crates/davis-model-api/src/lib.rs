//! Versioned file contracts shared by Davis model clients, runtimes, and components.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const ANALYSIS_API_VERSION: &str = "davis.analysis/v1alpha1";
pub const MODEL_API_VERSION: &str = "davis.model/v1alpha1";
pub const RUN_API_VERSION: &str = "davis.run/v1alpha1";
pub const RESULT_API_VERSION: &str = "davis.result/v1alpha1";

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid YAML in {path}: {source}")]
    Yaml {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    #[error("unsupported API version `{actual}`; expected `{expected}`")]
    ApiVersion {
        actual: String,
        expected: &'static str,
    },
    #[error("analysis plan must contain at least one input source")]
    MissingSources,
    #[error("model manifest ID and version must not be empty")]
    InvalidModelIdentity,
    #[error("model manifest must declare at least one operation")]
    MissingOperations,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisPlan {
    pub api_version: String,
    pub name: String,
    pub model: ModelSelection,
    #[serde(alias = "sources")]
    pub inputs: BTreeMap<String, InputSource>,
    #[serde(default)]
    pub config: Value,
    #[serde(default)]
    pub run: RunMetadata,
}

impl AnalysisPlan {
    /// Reads and validates an analysis plan from YAML.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, the YAML is invalid, or
    /// the common Davis contract is not satisfied.
    pub fn read(path: &Path) -> Result<Self, ContractError> {
        let text = fs::read_to_string(path).map_err(|source| ContractError::Read {
            path: path.to_owned(),
            source,
        })?;
        let plan: Self = serde_yaml::from_str(&text).map_err(|source| ContractError::Yaml {
            path: path.to_owned(),
            source,
        })?;
        plan.validate()?;
        Ok(plan)
    }

    /// Validates the common, component-independent analysis-plan fields.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported API version or an empty input map.
    pub fn validate(&self) -> Result<(), ContractError> {
        require_version(&self.api_version, ANALYSIS_API_VERSION)?;
        if self.inputs.is_empty() {
            return Err(ContractError::MissingSources);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSelection {
    pub component: String,
    pub version: String,
    pub operation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputSource {
    Local {
        path: PathBuf,
        #[serde(default)]
        read: Option<CsvReadOptions>,
    },
    Catalog {
        dataset_id: String,
        file_id: String,
        #[serde(default)]
        revision: Option<String>,
    },
    RunArtifact {
        run_id: String,
        artifact: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CsvReadOptions {
    #[serde(default)]
    pub encoding: Option<String>,
    #[serde(default)]
    pub delimiter: Option<String>,
    #[serde(default)]
    pub null_values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RunMetadata {
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub api_version: String,
    pub id: String,
    pub name: String,
    pub version: String,
    pub runtime: RuntimeDeclaration,
    pub operations: Vec<String>,
    pub inputs: Vec<ModelInput>,
    pub config_schema: PathBuf,
    #[serde(default)]
    pub ui_schema: Option<PathBuf>,
    #[serde(default)]
    pub outputs: OutputDeclaration,
}

impl ModelManifest {
    /// Reads and validates a model-component manifest from YAML.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, the YAML is invalid, or
    /// the common Davis manifest contract is not satisfied.
    pub fn read(path: &Path) -> Result<Self, ContractError> {
        let text = fs::read_to_string(path).map_err(|source| ContractError::Read {
            path: path.to_owned(),
            source,
        })?;
        let manifest: Self = serde_yaml::from_str(&text).map_err(|source| ContractError::Yaml {
            path: path.to_owned(),
            source,
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validates the common, runtime-independent manifest fields.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported API version, an empty identity, or
    /// a manifest without operations.
    pub fn validate(&self) -> Result<(), ContractError> {
        require_version(&self.api_version, MODEL_API_VERSION)?;
        if self.id.trim().is_empty() || self.version.trim().is_empty() {
            return Err(ContractError::InvalidModelIdentity);
        }
        if self.operations.is_empty() {
            return Err(ContractError::MissingOperations);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeclaration {
    pub kind: RuntimeKind,
    pub command: Vec<String>,
    #[serde(default = "default_request_argument")]
    pub request_argument: String,
    #[serde(default)]
    pub lockfile: Option<PathBuf>,
}

fn default_request_argument() -> String {
    "--request".to_owned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    Python,
    Native,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInput {
    pub name: String,
    pub media_types: Vec<String>,
    #[serde(default = "default_true")]
    pub required: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OutputDeclaration {
    #[serde(default)]
    pub standard: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRequest {
    pub api_version: String,
    pub run_id: String,
    pub operation: String,
    pub component: ResolvedComponent,
    pub inputs: BTreeMap<String, ResolvedInput>,
    pub config: Value,
    pub output_directory: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedComponent {
    pub id: String,
    pub version: String,
    pub manifest_path: PathBuf,
    pub source_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedInput {
    pub source: InputSource,
    pub resolved: ResolvedFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedFile {
    pub path: PathBuf,
    pub object_id: String,
    pub size: u64,
    pub media_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunResult {
    pub api_version: String,
    pub run_id: String,
    pub status: RunStatus,
    #[serde(default)]
    pub artifacts: BTreeMap<String, ArtifactDescriptor>,
    #[serde(default)]
    pub extensions: BTreeMap<String, ArtifactDescriptor>,
    #[serde(default)]
    pub error: Option<RunError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDescriptor {
    pub path: PathBuf,
    pub media_type: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub object_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunError {
    pub code: String,
    pub message: String,
}

fn require_version(actual: &str, expected: &'static str) -> Result<(), ContractError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ContractError::ApiVersion {
            actual: actual.to_owned(),
            expected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AnalysisPlan, InputSource, ANALYSIS_API_VERSION};

    #[test]
    fn analysis_plan_accepts_sources_alias() {
        let plan: AnalysisPlan = serde_yaml::from_str(
            r"
api_version: davis.analysis/v1alpha1
name: example
model:
  component: davis/mnl
  version: 0.1.0
  operation: estimate
sources:
  choice_data:
    kind: local
    path: choice.csv
",
        )
        .unwrap();
        assert_eq!(plan.api_version, ANALYSIS_API_VERSION);
        assert!(matches!(
            plan.inputs.get("choice_data"),
            Some(InputSource::Local { .. })
        ));
    }
}
