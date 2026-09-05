//! Versioned file contracts shared by Davis clients, runtimes, and components.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const ANALYSIS_API_VERSION: &str = "davis.analysis/v1alpha1";
pub const COMPONENT_API_VERSION: &str = "davis.component/v1alpha1";
pub const MODEL_API_VERSION: &str = "davis.model/v1alpha1";
pub const RUN_API_VERSION: &str = "davis.run/v1alpha1";
pub const RESULT_API_VERSION: &str = "davis.result/v1alpha1";
pub const COMPONENT_MANIFEST_FILENAME: &str = "component-manifest.yaml";
pub const LEGACY_MODEL_MANIFEST_FILENAME: &str = "model-manifest.yaml";

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
    #[error("component manifest ID and version must not be empty")]
    InvalidComponentIdentity,
    #[error("component manifest must declare at least one operation")]
    MissingOperations,
    #[error("invalid Davis version requirement `{value}`: {source}")]
    InvalidDavisRequirement {
        value: String,
        source: semver::Error,
    },
    #[error("artifact declaration `{0}` must contain at least one valid media type")]
    InvalidArtifactDeclaration(String),
    #[error("component package contains both component-manifest.yaml and model-manifest.yaml")]
    AmbiguousComponentManifest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisPlan {
    pub api_version: String,
    pub name: String,
    #[serde(alias = "model")]
    pub component: ComponentSelection,
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
pub struct ComponentSelection {
    #[serde(rename = "id", alias = "component")]
    pub component: String,
    pub version: String,
    pub operation: String,
}

/// Backward-compatible Rust name for [`ComponentSelection`].
pub type ModelSelection = ComponentSelection;

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
pub struct ComponentManifest {
    pub api_version: String,
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub kind: ComponentKind,
    #[serde(default)]
    pub requires_davis: Option<String>,
    pub runtime: RuntimeDeclaration,
    pub operations: Vec<String>,
    pub inputs: Vec<ComponentInput>,
    pub config_schema: PathBuf,
    #[serde(default)]
    pub ui_schema: Option<PathBuf>,
    #[serde(default)]
    pub outputs: OutputDeclaration,
}

/// Backward-compatible Rust name for [`ComponentManifest`].
pub type ModelManifest = ComponentManifest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    #[default]
    Model,
    Transform,
    Visualize,
}

impl ComponentManifest {
    /// Reads and validates a component manifest from YAML.
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

    /// Finds and reads the canonical or legacy manifest in a component directory.
    ///
    /// # Errors
    ///
    /// Returns an error when both manifest filenames are present, neither file
    /// can be read, or the selected manifest is invalid.
    pub fn read_from_directory(directory: &Path) -> Result<(PathBuf, Self), ContractError> {
        let canonical = directory.join(COMPONENT_MANIFEST_FILENAME);
        let legacy = directory.join(LEGACY_MODEL_MANIFEST_FILENAME);
        match (canonical.is_file(), legacy.is_file()) {
            (true, true) => Err(ContractError::AmbiguousComponentManifest),
            (_, false) => Ok((canonical.clone(), Self::read(&canonical)?)),
            (false, true) => Ok((legacy.clone(), Self::read(&legacy)?)),
        }
    }

    /// Validates the common, runtime-independent manifest fields.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported API version, an empty identity, or
    /// a manifest without operations.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.api_version != COMPONENT_API_VERSION && self.api_version != MODEL_API_VERSION {
            return Err(ContractError::ApiVersion {
                actual: self.api_version.clone(),
                expected: COMPONENT_API_VERSION,
            });
        }
        if self.id.trim().is_empty() || self.version.trim().is_empty() {
            return Err(ContractError::InvalidComponentIdentity);
        }
        if self.operations.is_empty() {
            return Err(ContractError::MissingOperations);
        }
        if let Some(requirement) = &self.requires_davis {
            semver::VersionReq::parse(requirement).map_err(|source| {
                ContractError::InvalidDavisRequirement {
                    value: requirement.clone(),
                    source,
                }
            })?;
        }
        for (name, declaration) in &self.outputs.artifacts {
            if name.trim().is_empty()
                || declaration.media_types.is_empty()
                || declaration
                    .media_types
                    .iter()
                    .any(|media_type| media_type.trim().is_empty())
            {
                return Err(ContractError::InvalidArtifactDeclaration(name.clone()));
            }
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
pub struct ComponentInput {
    pub name: String,
    pub media_types: Vec<String>,
    #[serde(default = "default_true")]
    pub required: bool,
}

/// Backward-compatible Rust name for [`ComponentInput`].
pub type ModelInput = ComponentInput;

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OutputDeclaration {
    #[serde(default)]
    pub standard: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub artifacts: BTreeMap<String, ArtifactDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDeclaration {
    #[serde(default)]
    pub media_types: Vec<String>,
    #[serde(default = "default_true")]
    pub required: bool,
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
    #[serde(default)]
    pub kind: ComponentKind,
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
    use super::{
        AnalysisPlan, ComponentKind, ComponentManifest, ContractError, InputSource, ModelManifest,
        ANALYSIS_API_VERSION,
    };

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

    #[test]
    fn component_manifest_validates_optional_davis_requirement() {
        let valid: ComponentManifest = serde_yaml::from_str(
            r"
api_version: davis.component/v1alpha1
id: example/native
name: Example Native
version: 1.0.0
requires_davis: '>=0.3.5'
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs: []
config_schema: schemas/config.json
",
        )
        .unwrap();
        valid.validate().unwrap();

        let mut invalid = valid;
        invalid.requires_davis = Some("not-a-requirement".to_owned());
        assert!(matches!(
            invalid.validate(),
            Err(ContractError::InvalidDavisRequirement { .. })
        ));
    }

    #[test]
    fn component_alias_and_kinds_preserve_model_compatibility() {
        let plan: AnalysisPlan = serde_yaml::from_str(
            r"
api_version: davis.analysis/v1alpha1
name: transform-example
component:
  id: example/transform
  version: 0.1.0
  operation: transform
inputs:
  table:
    kind: local
    path: input.csv
",
        )
        .unwrap();
        assert_eq!(plan.component.component, "example/transform");
        let serialized = serde_yaml::to_string(&plan).unwrap();
        assert!(serialized.contains("component:\n  id: example/transform"));
        assert!(!serialized.contains("model:"));

        let legacy: ModelManifest = serde_yaml::from_str(
            r"
api_version: davis.model/v1alpha1
id: example/model
name: Legacy model
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs: []
config_schema: schemas/config.json
",
        )
        .unwrap();
        assert_eq!(legacy.kind, ComponentKind::Model);
    }

    #[test]
    fn directory_reader_accepts_legacy_filename_and_rejects_ambiguity() {
        let temporary = tempfile::tempdir().unwrap();
        let legacy_path = temporary.path().join("model-manifest.yaml");
        let manifest = r"api_version: davis.model/v1alpha1
id: example/legacy
name: Legacy
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [run]
inputs: []
config_schema: config.json
";
        std::fs::write(&legacy_path, manifest).unwrap();
        let (selected, loaded) = ComponentManifest::read_from_directory(temporary.path()).unwrap();
        assert_eq!(selected, legacy_path);
        assert_eq!(loaded.id, "example/legacy");

        std::fs::write(
            temporary.path().join("component-manifest.yaml"),
            manifest.replace("davis.model/v1alpha1", "davis.component/v1alpha1"),
        )
        .unwrap();
        assert!(matches!(
            ComponentManifest::read_from_directory(temporary.path()),
            Err(ContractError::AmbiguousComponentManifest)
        ));
    }
}
