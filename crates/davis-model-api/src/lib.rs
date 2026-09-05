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
    #[error("additional input declaration must contain at least one valid media type")]
    InvalidAdditionalInputDeclaration,
    #[error("component package contains both component-manifest.yaml and model-manifest.yaml")]
    AmbiguousComponentManifest,
    #[error("invalid table binding: {0}")]
    InvalidTableBinding(String),
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
        for (name, input) in &self.inputs {
            if let InputSource::TableBinding { binding } = input {
                binding.validate(name)?;
            }
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
    TableBinding {
        #[serde(flatten)]
        binding: TableBinding,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableBinding {
    pub processor: BindingProcessor,
    pub sources: BTreeMap<String, InputSource>,
    pub base: String,
    #[serde(default)]
    pub joins: Vec<TableJoin>,
    pub columns: BTreeMap<String, BoundColumn>,
}

impl TableBinding {
    fn validate(&self, input_name: &str) -> Result<(), ContractError> {
        if self.processor.id.trim().is_empty() || self.processor.version.trim().is_empty() {
            return Err(ContractError::InvalidTableBinding(format!(
                "input `{input_name}` has an empty processor ID or version"
            )));
        }
        if !self.sources.contains_key(&self.base) {
            return Err(ContractError::InvalidTableBinding(format!(
                "input `{input_name}` does not contain base source `{}`",
                self.base
            )));
        }
        for (source_name, source) in &self.sources {
            if matches!(source, InputSource::TableBinding { .. }) {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` nests a table binding in source `{source_name}`"
                )));
            }
        }
        let mut joined = std::collections::BTreeSet::new();
        for join in &self.joins {
            let left_columns = join.left_on.columns();
            let right_columns = join.right_on.columns();
            if join.source == self.base || !self.sources.contains_key(&join.source) {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` has invalid join source `{}`",
                    join.source
                )));
            }
            if !joined.insert(&join.source) {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` joins source `{}` more than once",
                    join.source
                )));
            }
            if left_columns.len() != right_columns.len() {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` has different left and right key lengths for `{}`",
                    join.source
                )));
            }
            if left_columns.is_empty()
                || left_columns
                    .iter()
                    .chain(right_columns.iter())
                    .any(|column| column.trim().is_empty())
            {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` has an empty join key for `{}`",
                    join.source
                )));
            }
        }
        if self.columns.is_empty() {
            return Err(ContractError::InvalidTableBinding(format!(
                "input `{input_name}` must select at least one output column"
            )));
        }
        for (output, column) in &self.columns {
            if output.trim().is_empty()
                || column.column.trim().is_empty()
                || !self.sources.contains_key(&column.source)
            {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` has invalid output column `{output}`"
                )));
            }
            if column.source != self.base && !joined.contains(&column.source) {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` selects from unjoined source `{}`",
                    column.source
                )));
            }
        }
        for source_name in self.sources.keys().filter(|name| *name != &self.base) {
            if !joined.contains(source_name) {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` contains unused source `{source_name}`"
                )));
            }
            if !self
                .columns
                .values()
                .any(|column| &column.source == source_name)
            {
                return Err(ContractError::InvalidTableBinding(format!(
                    "input `{input_name}` selects no columns from joined source `{source_name}`"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingProcessor {
    pub id: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableJoin {
    pub source: String,
    pub left_on: KeyColumns,
    pub right_on: KeyColumns,
    #[serde(default)]
    pub relationship: JoinRelationship,
    #[serde(default)]
    pub how: JoinKind,
    #[serde(default)]
    pub allow_unmatched: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundColumn {
    pub source: String,
    pub column: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KeyColumns {
    One(String),
    Many(Vec<String>),
}

impl KeyColumns {
    #[must_use]
    pub fn columns(&self) -> Vec<&str> {
        match self {
            Self::One(column) => vec![column],
            Self::Many(columns) => columns.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinRelationship {
    #[default]
    ManyToOne,
    OneToOne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinKind {
    #[default]
    Left,
    Inner,
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
    #[serde(default)]
    pub additional_inputs: Option<AdditionalInputDeclaration>,
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
        if self.additional_inputs.as_ref().is_some_and(|declaration| {
            declaration.media_types.is_empty()
                || declaration
                    .media_types
                    .iter()
                    .any(|media_type| media_type.trim().is_empty())
        }) {
            return Err(ContractError::InvalidAdditionalInputDeclaration);
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdditionalInputDeclaration {
    pub media_types: Vec<String>,
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
        AdditionalInputDeclaration, AnalysisPlan, ComponentKind, ComponentManifest, ContractError,
        InputSource, ModelManifest, ANALYSIS_API_VERSION,
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

        let mut invalid = valid.clone();
        invalid.requires_davis = Some("not-a-requirement".to_owned());
        assert!(matches!(
            invalid.validate(),
            Err(ContractError::InvalidDavisRequirement { .. })
        ));

        let mut invalid_additional_inputs = valid;
        invalid_additional_inputs.additional_inputs = Some(AdditionalInputDeclaration {
            media_types: Vec::new(),
        });
        assert!(matches!(
            invalid_additional_inputs.validate(),
            Err(ContractError::InvalidAdditionalInputDeclaration)
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
    fn analysis_plan_validates_a_multi_source_table_binding() {
        let plan: AnalysisPlan = serde_yaml::from_str(
            r"
api_version: davis.analysis/v1alpha1
name: multi-source
component:
  id: davis/mnl
  version: 0.2.0
  operation: estimate
inputs:
  choice_data:
    kind: table_binding
    processor:
      id: davis/csv-transform
      version: 0.4.0
    sources:
      choices:
        kind: local
        path: choices.csv
      persons:
        kind: local
        path: persons.csv
    base: choices
    joins:
      - source: persons
        left_on: person_id
        right_on: person_id
    columns:
      case_id:
        source: choices
        column: person_id
      income:
        source: persons
        column: income
config: {}
",
        )
        .unwrap();

        plan.validate().unwrap();
        assert!(matches!(
            plan.inputs.get("choice_data"),
            Some(InputSource::TableBinding { .. })
        ));
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
