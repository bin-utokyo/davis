//! Versioned file contracts shared by Davis clients, runtimes, and components.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const ANALYSIS_API_VERSION: &str = "davis.analysis/v1alpha1";
pub const COMPONENT_API_VERSION: &str = "davis.component/v1";
pub const LEGACY_COMPONENT_API_VERSION: &str = "davis.component/v1alpha1";
pub const MODEL_API_VERSION: &str = "davis.model/v1alpha1";
pub const RUN_API_VERSION: &str = "davis.run/v1alpha1";
pub const RESULT_API_VERSION: &str = "davis.result/v1alpha1";
pub const COMPONENT_MANIFEST_FILENAME: &str = "component.yaml";
pub const LEGACY_COMPONENT_MANIFEST_FILENAME: &str = "component-manifest.yaml";
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
    #[error("component runtime command must not be empty")]
    EmptyRuntimeCommand,
    #[error("runtime requirement command must not be empty")]
    EmptyRuntimeRequirement,
    #[error("runtime version requirement must declare version_arguments")]
    EmptyRuntimeVersionArguments,
    #[error("invalid runtime version requirement `{value}`: {source}")]
    InvalidRuntimeRequirement {
        value: String,
        source: semver::Error,
    },
    #[error("invalid Davis version requirement `{value}`: {source}")]
    InvalidDavisRequirement {
        value: String,
        source: semver::Error,
    },
    #[error("artifact declaration `{0}` must contain at least one valid media type")]
    InvalidArtifactDeclaration(String),
    #[error("additional input declaration must contain at least one valid media type")]
    InvalidAdditionalInputDeclaration,
    #[error("component package contains multiple manifest files; keep only one of component.yaml, component-manifest.yaml, or model-manifest.yaml")]
    AmbiguousComponentManifest,
    #[error("component manifest must declare exactly one configuration schema source")]
    InvalidConfigurationDeclaration,
    #[error("component manifest may declare at most one presentation source")]
    InvalidPresentationDeclaration,
    #[error("component manifest document reference must be a safe relative path: {0}")]
    UnsafeDocumentReference(PathBuf),
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<ConfigurationDeclaration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<PresentationDeclaration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_schema: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui_schema: Option<PathBuf>,
    #[serde(default)]
    pub outputs: OutputDeclaration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigurationDeclaration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_ref: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationDeclaration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui_ref: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedManifestDocument {
    pub value: Value,
    pub source_path: PathBuf,
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
        let candidates = [
            directory.join(COMPONENT_MANIFEST_FILENAME),
            directory.join(LEGACY_COMPONENT_MANIFEST_FILENAME),
            directory.join(LEGACY_MODEL_MANIFEST_FILENAME),
        ];
        let existing: Vec<_> = candidates
            .into_iter()
            .filter(|path| path.is_file())
            .collect();
        match existing.as_slice() {
            [] => {
                let canonical = directory.join(COMPONENT_MANIFEST_FILENAME);
                Ok((canonical.clone(), Self::read(&canonical)?))
            }
            [path] => Ok((path.clone(), Self::read(path)?)),
            _ => Err(ContractError::AmbiguousComponentManifest),
        }
    }

    /// Resolves the inline or referenced JSON Schema for component configuration.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid source declaration, unsafe reference,
    /// unreadable document, or invalid JSON/YAML document.
    pub fn resolve_configuration(
        &self,
        manifest_path: &Path,
    ) -> Result<ResolvedManifestDocument, ContractError> {
        let inline = self
            .configuration
            .as_ref()
            .and_then(|declaration| declaration.schema.as_ref());
        let reference = self
            .configuration
            .as_ref()
            .and_then(|declaration| declaration.schema_ref.as_ref())
            .or(self.config_schema.as_ref());
        if usize::from(inline.is_some()) + usize::from(reference.is_some()) != 1 {
            return Err(ContractError::InvalidConfigurationDeclaration);
        }
        resolve_document(manifest_path, inline, reference)
    }

    /// Resolves optional GUI presentation hints from the manifest or a referenced file.
    ///
    /// # Errors
    ///
    /// Returns an error when multiple sources are declared or a referenced
    /// document is unsafe, unreadable, or invalid.
    pub fn resolve_presentation(
        &self,
        manifest_path: &Path,
    ) -> Result<Option<ResolvedManifestDocument>, ContractError> {
        let inline = self
            .presentation
            .as_ref()
            .and_then(|declaration| declaration.ui.as_ref());
        let reference = self
            .presentation
            .as_ref()
            .and_then(|declaration| declaration.ui_ref.as_ref())
            .or(self.ui_schema.as_ref());
        if inline.is_some() && reference.is_some() {
            return Err(ContractError::InvalidPresentationDeclaration);
        }
        if inline.is_none() && reference.is_none() {
            return Ok(None);
        }
        resolve_document(manifest_path, inline, reference).map(Some)
    }

    /// Validates the common, runtime-independent manifest fields.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported API version, an empty identity, or
    /// a manifest without operations.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.api_version != COMPONENT_API_VERSION
            && self.api_version != LEGACY_COMPONENT_API_VERSION
            && self.api_version != MODEL_API_VERSION
        {
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
        validate_runtime_declaration(&self.runtime)?;
        validate_manifest_documents(self)?;
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

fn validate_runtime_declaration(runtime: &RuntimeDeclaration) -> Result<(), ContractError> {
    if runtime.command.is_empty() || runtime.command[0].trim().is_empty() {
        return Err(ContractError::EmptyRuntimeCommand);
    }
    for requirement in &runtime.requirements {
        if requirement.command.trim().is_empty() {
            return Err(ContractError::EmptyRuntimeRequirement);
        }
        if let Some(version) = &requirement.version {
            if requirement.version_arguments.is_empty() {
                return Err(ContractError::EmptyRuntimeVersionArguments);
            }
            semver::VersionReq::parse(version).map_err(|source| {
                ContractError::InvalidRuntimeRequirement {
                    value: version.clone(),
                    source,
                }
            })?;
        }
    }
    Ok(())
}

fn validate_manifest_documents(manifest: &ComponentManifest) -> Result<(), ContractError> {
    let configuration = manifest.configuration.as_ref();
    let configuration_sources =
        usize::from(configuration.is_some_and(|declaration| declaration.schema.is_some()))
            + usize::from(
                configuration.is_some_and(|declaration| declaration.schema_ref.is_some()),
            )
            + usize::from(manifest.config_schema.is_some());
    if configuration_sources != 1 {
        return Err(ContractError::InvalidConfigurationDeclaration);
    }
    for reference in configuration
        .and_then(|declaration| declaration.schema_ref.as_ref())
        .into_iter()
        .chain(manifest.config_schema.iter())
    {
        validate_document_reference(reference)?;
    }

    let presentation = manifest.presentation.as_ref();
    let presentation_sources =
        usize::from(presentation.is_some_and(|declaration| declaration.ui.is_some()))
            + usize::from(presentation.is_some_and(|declaration| declaration.ui_ref.is_some()))
            + usize::from(manifest.ui_schema.is_some());
    if presentation_sources > 1 {
        return Err(ContractError::InvalidPresentationDeclaration);
    }
    for reference in presentation
        .and_then(|declaration| declaration.ui_ref.as_ref())
        .into_iter()
        .chain(manifest.ui_schema.iter())
    {
        validate_document_reference(reference)?;
    }
    Ok(())
}

#[must_use]
pub fn is_component_manifest_filename(name: &std::ffi::OsStr) -> bool {
    name == COMPONENT_MANIFEST_FILENAME
        || name == LEGACY_COMPONENT_MANIFEST_FILENAME
        || name == LEGACY_MODEL_MANIFEST_FILENAME
}

fn resolve_document(
    manifest_path: &Path,
    inline: Option<&Value>,
    reference: Option<&PathBuf>,
) -> Result<ResolvedManifestDocument, ContractError> {
    if let Some(value) = inline {
        return Ok(ResolvedManifestDocument {
            value: value.clone(),
            source_path: manifest_path.to_owned(),
        });
    }
    let reference = reference.ok_or(ContractError::InvalidConfigurationDeclaration)?;
    validate_document_reference(reference)?;
    let path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(reference);
    let text = fs::read_to_string(&path).map_err(|source| ContractError::Read {
        path: path.clone(),
        source,
    })?;
    let value = if matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("yaml" | "yml")
    ) {
        serde_yaml::from_str(&text).map_err(|source| ContractError::Yaml {
            path: path.clone(),
            source,
        })?
    } else {
        serde_json::from_str(&text).map_err(|source| ContractError::Json {
            path: path.clone(),
            source,
        })?
    };
    Ok(ResolvedManifestDocument {
        value,
        source_path: path,
    })
}

fn validate_document_reference(reference: &Path) -> Result<(), ContractError> {
    if reference.as_os_str().is_empty()
        || reference.is_absolute()
        || reference
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ContractError::UnsafeDocumentReference(reference.to_owned()));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeclaration {
    #[serde(default)]
    pub executor: RuntimeExecutor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<RuntimeKind>,
    pub command: Vec<String>,
    #[serde(default = "default_request_argument")]
    pub request_argument: String,
    #[serde(default)]
    pub lockfile: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requirements: Vec<RuntimeRequirement>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeExecutor {
    #[default]
    Process,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRequirement {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default = "default_version_arguments")]
    pub version_arguments: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub install: BTreeMap<String, String>,
}

fn default_version_arguments() -> Vec<String> {
    vec!["--version".to_owned()]
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
        InputSource, ModelManifest, RuntimeExecutor, ANALYSIS_API_VERSION,
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
    fn process_runtime_is_language_independent_and_validates_requirements() {
        let valid: ComponentManifest = serde_yaml::from_str(
            r"api_version: davis.component/v1
id: example/r-model
name: Example R model
version: 1.0.0
runtime:
  executor: process
  command: [Rscript, model.R]
  requirements:
    - command: Rscript
      version: '>=4.4'
      install:
        macos: https://cran.r-project.org/
operations: [estimate]
inputs: []
configuration:
  schema: {type: object}
outputs: {}
",
        )
        .unwrap();
        valid.validate().unwrap();
        assert_eq!(valid.runtime.executor, RuntimeExecutor::Process);
        assert!(valid.runtime.kind.is_none());

        let mut invalid = valid;
        invalid.runtime.requirements[0].version = Some("not-semver".to_owned());
        assert!(matches!(
            invalid.validate(),
            Err(ContractError::InvalidRuntimeRequirement { .. })
        ));
    }

    #[test]
    fn resolves_inline_configuration_and_presentation_from_component_yaml() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_path = temporary.path().join("component.yaml");
        std::fs::write(
            &manifest_path,
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
    required: [scale]
    properties:
      scale:
        type: number
presentation:
  ui:
    ui:editor: generic
outputs: {}
",
        )
        .unwrap();

        let (selected, manifest) =
            ComponentManifest::read_from_directory(temporary.path()).unwrap();
        assert_eq!(selected, manifest_path);
        let configuration = manifest.resolve_configuration(&selected).unwrap();
        assert_eq!(configuration.source_path, selected);
        assert_eq!(configuration.value["properties"]["scale"]["type"], "number");
        let presentation = manifest.resolve_presentation(&selected).unwrap().unwrap();
        assert_eq!(presentation.value["ui:editor"], "generic");
    }

    #[test]
    fn rejects_multiple_inline_and_referenced_document_sources() {
        let manifest: ComponentManifest = serde_yaml::from_str(
            r"api_version: davis.component/v1
id: example/ambiguous
name: Ambiguous component
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs: []
configuration:
  schema: {type: object}
config_schema: schemas/config.json
outputs: {}
",
        )
        .unwrap();

        assert!(matches!(
            manifest.validate(),
            Err(ContractError::InvalidConfigurationDeclaration)
        ));

        let unsafe_reference: ComponentManifest = serde_yaml::from_str(
            r"api_version: davis.component/v1
id: example/unsafe
name: Unsafe component
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs: []
configuration:
  schema_ref: ../outside.json
outputs: {}
",
        )
        .unwrap();
        assert!(matches!(
            unsafe_reference.validate(),
            Err(ContractError::UnsafeDocumentReference(_))
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
