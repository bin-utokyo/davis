use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use davis_client::{remote::DavisService, session};
use davis_core::CatalogCache;
use davis_model_api::{AnalysisPlan, ComponentManifest, InputSource, RunResult};
use davis_runtime::{
    distinct_csv_values, execute_plan, inspect_csv, list_components, load_component, validate_plan,
    CompletedRun, CsvProfile, DistinctValues,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct ValidPlanResponse {
    valid: bool,
    plan: PathBuf,
    component: ComponentResponse,
}

#[derive(Serialize)]
struct ComponentResponse {
    id: String,
    version: String,
    manifest: PathBuf,
}

#[derive(Serialize)]
struct ComponentEditorResponse {
    manifest: ComponentManifest,
    config_schema: Value,
    ui_schema: Value,
    ui_extensions: BTreeMap<String, UiExtensionResponse>,
}

#[derive(Serialize)]
struct UiExtensionResponse {
    api_version: String,
    html: String,
}

#[derive(Serialize)]
struct EditablePlanResponse {
    yaml: String,
    plan: AnalysisPlan,
    editor: ComponentEditorResponse,
    resolved_sources: BTreeMap<String, PathBuf>,
}

#[derive(Serialize)]
struct ArtifactPreviewResponse {
    name: String,
    media_type: String,
    content: Value,
}

#[derive(Serialize)]
struct RunHistoryResponse {
    runs: Vec<RunHistoryEntry>,
    warnings: Vec<String>,
}

#[derive(Serialize)]
struct RunHistoryEntry {
    run: CompletedRun,
    plan_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    tags: Vec<String>,
    modified_at_unix_ms: u64,
}

#[derive(Serialize)]
struct CatalogFileResponse {
    dataset_id: String,
    file_id: String,
    path: String,
    title: String,
    size: u64,
    columns: Vec<String>,
}

#[derive(Serialize)]
struct DownloadedCatalogFileResponse {
    dataset_id: String,
    file_id: String,
    path: PathBuf,
}

fn catalog_service(locale: &str) -> Result<DavisService, String> {
    let stored = session::load()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| {
            if locale == "en" {
                "Run `davis login <URL>` before using Davis Catalog.".to_owned()
            } else {
                "Davis Catalogを使うには，先に`davis login <URL>`を実行してください．".to_owned()
            }
        })?;
    DavisService::new(&stored.service_url, Some(stored.token)).map_err(|error| error.to_string())
}

#[tauri::command]
async fn catalog_files(locale: String) -> Result<Vec<CatalogFileResponse>, String> {
    let english = locale == "en";
    let catalog = catalog_service(&locale)?
        .catalog()
        .await
        .map_err(|error| error.to_string())?;
    Ok(catalog
        .datasets
        .into_iter()
        .flat_map(|dataset| {
            let dataset_id = dataset.id;
            dataset
                .files
                .into_iter()
                .filter(|file| {
                    Path::new(&file.path).extension().is_some_and(|extension| {
                        extension.eq_ignore_ascii_case("csv")
                            || extension.eq_ignore_ascii_case("tsv")
                    })
                })
                .map(move |file| CatalogFileResponse {
                    dataset_id: dataset_id.clone(),
                    file_id: file.id,
                    title: file.schema.as_ref().map_or_else(
                        || file.path.clone(),
                        |schema| {
                            if english {
                                schema.name.en.clone()
                            } else {
                                schema.name.ja.clone()
                            }
                        },
                    ),
                    path: file.path,
                    size: file.size,
                    columns: file.schema.map_or_else(Vec::new, |schema| {
                        schema
                            .columns
                            .into_iter()
                            .map(|column| column.name)
                            .collect()
                    }),
                })
        })
        .collect())
}

#[tauri::command]
async fn download_catalog_file(
    dataset_id: String,
    file_id: String,
    locale: String,
) -> Result<DownloadedCatalogFileResponse, String> {
    let service = catalog_service(&locale)?;
    let manifest = service
        .manifest(&dataset_id)
        .await
        .map_err(|error| error.to_string())?;
    let selected = manifest
        .select_files(std::slice::from_ref(&file_id))
        .map_err(|error| error.to_string())?;
    let cache = CatalogCache::for_user().map_err(|error| error.to_string())?;
    service
        .download_manifest(&cache.object_store(), &selected, |_, _, _, _| {})
        .await
        .map_err(|error| error.to_string())?;
    let path = cache
        .materialize_file(&manifest, &file_id)
        .map_err(|error| error.to_string())?;
    Ok(DownloadedCatalogFileResponse {
        dataset_id,
        file_id,
        path,
    })
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn inspect_csv_file(path: PathBuf) -> Result<CsvProfile, String> {
    inspect_csv(&path).map_err(|error| error.to_string())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn inspect_distinct_values(path: PathBuf, column: String) -> Result<DistinctValues, String> {
    distinct_csv_values(&path, &column, 200).map_err(|error| error.to_string())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn component_editor_definition(
    repository: PathBuf,
    component_id: String,
    version: String,
) -> Result<ComponentEditorResponse, String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    editor_definition(&repository, &component_id, &version)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn component_editor_definitions(
    repository: PathBuf,
) -> Result<Vec<ComponentEditorResponse>, String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    let editors = list_components(&repository)
        .into_iter()
        .filter_map(|(path, manifest)| editor_response(&path, manifest).ok())
        .filter(|editor| editor.ui_schema["version"] == "davis.ui/v1")
        .collect();
    Ok(editors)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn load_analysis_plan_for_editing(
    repository: PathBuf,
    path: PathBuf,
) -> Result<EditablePlanResponse, String> {
    ensure_arguments(&repository, &path)?;
    let yaml = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read analysis plan {}: {error}", path.display()))?;
    let plan = AnalysisPlan::read(&path).map_err(|error| error.to_string())?;
    let plan_directory = path.parent().unwrap_or_else(|| Path::new("."));
    let mut resolved_sources = BTreeMap::new();
    for (slot, source) in &plan.inputs {
        collect_resolved_sources(source, slot, plan_directory, &mut resolved_sources)?;
    }
    let editor = editor_definition(
        &repository,
        &plan.component.component,
        &plan.component.version,
    )?;
    Ok(EditablePlanResponse {
        yaml,
        plan,
        editor,
        resolved_sources,
    })
}

fn collect_resolved_sources(
    source: &InputSource,
    name: &str,
    plan_directory: &Path,
    resolved: &mut BTreeMap<String, PathBuf>,
) -> Result<(), String> {
    match source {
        InputSource::Local { path, .. } => {
            resolved.insert(
                name.to_owned(),
                if path.is_relative() {
                    plan_directory.join(path)
                } else {
                    path.clone()
                },
            );
        }
        InputSource::TableBinding { binding } => {
            for (nested_name, nested) in &binding.sources {
                collect_resolved_sources(
                    nested,
                    &format!("{name}/{nested_name}"),
                    plan_directory,
                    resolved,
                )?;
            }
        }
        InputSource::Catalog {
            dataset_id,
            file_id,
            revision,
        } => {
            let path = CatalogCache::for_user()
                .and_then(|cache| cache.resolve_file(dataset_id, file_id, revision.as_deref()))
                .map_err(|error| error.to_string())?;
            resolved.insert(name.to_owned(), path);
        }
        InputSource::RunArtifact { .. } => {}
    }
    Ok(())
}

fn editor_definition(
    repository: &Path,
    component_id: &str,
    version: &str,
) -> Result<ComponentEditorResponse, String> {
    let (manifest_path, manifest) =
        load_component(repository, component_id, version).map_err(|error| error.to_string())?;
    editor_response(&manifest_path, manifest)
}

fn editor_response(
    manifest_path: &Path,
    manifest: ComponentManifest,
) -> Result<ComponentEditorResponse, String> {
    let config_schema = manifest
        .resolve_configuration(manifest_path)
        .map_err(|error| error.to_string())?
        .value;
    let ui_schema = manifest
        .resolve_presentation(manifest_path)
        .map_err(|error| error.to_string())?
        .map_or_else(
            || Value::Object(serde_json::Map::new()),
            |document| document.value,
        );
    let ui_schema = normalize_editor_presentation(&manifest, &config_schema, ui_schema)?;
    if ui_schema["version"] == "davis.ui/v1" {
        validate_editor_presentation(&manifest, &config_schema, &ui_schema)?;
    }
    let ui_extensions = load_ui_extensions(manifest_path, &ui_schema)?;
    Ok(ComponentEditorResponse {
        manifest,
        config_schema,
        ui_schema,
        ui_extensions,
    })
}

fn load_ui_extensions(
    manifest_path: &Path,
    ui_schema: &Value,
) -> Result<BTreeMap<String, UiExtensionResponse>, String> {
    const EXTENSION_SIZE_LIMIT: u64 = 512 * 1024;
    let mut loaded = BTreeMap::new();
    let Some(extensions) = ui_schema.get("extensions") else {
        return Ok(loaded);
    };
    let extensions = extensions
        .as_array()
        .ok_or("presentation.ui.extensions must be an array")?;
    let package_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let canonical_root = fs::canonicalize(package_root).map_err(|error| {
        format!(
            "failed to access component package {}: {error}",
            package_root.display()
        )
    })?;
    for extension in extensions {
        let extension = extension
            .as_object()
            .ok_or("every UI extension must be an object")?;
        let id = extension
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or("every UI extension requires id")?;
        if !id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        {
            return Err(format!("UI extension id `{id}` is not portable"));
        }
        let api_version = extension
            .get("api_version")
            .and_then(Value::as_str)
            .unwrap_or("");
        if api_version != "davis.widget/v1" {
            return Err(format!(
                "UI extension `{id}` requires unsupported API `{api_version}`"
            ));
        }
        let source = extension
            .get("source")
            .and_then(Value::as_str)
            .map(Path::new)
            .ok_or_else(|| format!("UI extension `{id}` requires source"))?;
        if source.as_os_str().is_empty()
            || source.is_absolute()
            || source
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(format!(
                "UI extension `{id}` source must be a safe package-relative path"
            ));
        }
        let path = package_root.join(source);
        let canonical_path = fs::canonicalize(&path).map_err(|error| {
            format!("failed to access UI extension {}: {error}", path.display())
        })?;
        if !canonical_path.starts_with(&canonical_root) || !canonical_path.is_file() {
            return Err(format!("UI extension `{id}` resolves outside its package"));
        }
        let size = fs::metadata(&canonical_path)
            .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
            .len();
        if size > EXTENSION_SIZE_LIMIT {
            return Err(format!("UI extension `{id}` exceeds 512 KiB"));
        }
        let html = fs::read_to_string(&canonical_path)
            .map_err(|error| format!("failed to read UI extension {}: {error}", path.display()))?;
        if loaded
            .insert(
                id.to_owned(),
                UiExtensionResponse {
                    api_version: api_version.to_owned(),
                    html,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate UI extension id `{id}`"));
        }
    }
    Ok(loaded)
}

fn normalize_editor_presentation(
    manifest: &ComponentManifest,
    config_schema: &Value,
    ui_schema: Value,
) -> Result<Value, String> {
    if ui_schema["version"] == "davis.ui/v1" {
        return Ok(ui_schema);
    }
    let editor = ui_schema["ui:editor"].as_str().unwrap_or("");
    if editor == "schema-form" {
        let mut normalized = ui_schema["ui:form"]
            .as_object()
            .cloned()
            .ok_or("legacy schema-form requires ui:form")?;
        normalized.insert(
            "version".to_owned(),
            Value::String("davis.ui/v1".to_owned()),
        );
        if let Some(results) = ui_schema.get("ui:results") {
            normalized.insert("results".to_owned(), results.clone());
        }
        if let Some(sections) = normalized.get_mut("sections").and_then(Value::as_array_mut) {
            for section in sections {
                if let Some(object) = section.as_object_mut() {
                    if let Some(path) = object
                        .remove("path")
                        .and_then(|value| value.as_str().map(str::to_owned))
                    {
                        object.insert("bind".to_owned(), Value::String(dot_path_to_pointer(&path)));
                    }
                    for key in ["alternatives_from", "parameters_from"] {
                        if let Some(path) =
                            object.get(key).and_then(Value::as_str).map(str::to_owned)
                        {
                            object
                                .insert(key.to_owned(), Value::String(dot_path_to_pointer(&path)));
                        }
                    }
                }
            }
        }
        return Ok(Value::Object(normalized));
    }
    if editor != "linear-utility" {
        return Ok(ui_schema);
    }

    let input = manifest
        .inputs
        .first()
        .ok_or("legacy linear-utility editor requires an input")?;
    let labels = ui_schema["roles"]["ui:labels"].clone();
    let mut sections = Vec::new();
    if schema_contains_pointer(config_schema, "/roles") {
        sections.push(serde_json::json!({"bind": "/roles", "widget": "column-map", "input": input.name, "title": "役割列", "labels": labels}));
    }
    if schema_contains_pointer(config_schema, "/terms") {
        sections.push(serde_json::json!({"bind": "/terms", "widget": "utility-terms", "input": input.name, "title": "効用term", "allow_constant": true, "alternatives_from": "/roles/alternative_id"}));
    }
    for path in ["parameters", "estimation"] {
        let pointer = format!("/{path}");
        if schema_contains_pointer(config_schema, &pointer) {
            sections.push(serde_json::json!({"bind": pointer, "widget": "auto", "title": path}));
        }
    }
    Ok(serde_json::json!({
        "version": "davis.ui/v1",
        "inputs": {
            input.name.clone(): {
                "title": "入力データ",
                "widget": "table-binding",
                "preparation": ui_schema.get("ui:inputPreparation").cloned().unwrap_or(Value::Null)
            }
        },
        "sections": sections,
        "results": ui_schema.get("ui:results").cloned().unwrap_or_else(|| Value::Array(Vec::new()))
    }))
}

fn dot_path_to_pointer(path: &str) -> String {
    format!("/{}", path.replace('.', "/"))
}

#[allow(clippy::too_many_lines)]
fn validate_editor_presentation(
    manifest: &ComponentManifest,
    config_schema: &Value,
    ui_schema: &Value,
) -> Result<(), String> {
    if ui_schema["version"] != "davis.ui/v1" {
        return Err("presentation.ui.version must be davis.ui/v1".to_owned());
    }
    let form = ui_schema
        .as_object()
        .ok_or("presentation.ui must be an object")?;
    let inputs = form
        .get("inputs")
        .and_then(Value::as_object)
        .ok_or("davis.ui/v1 requires inputs")?;
    let declared_inputs: std::collections::BTreeSet<_> = manifest
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect();
    if inputs.values().any(|metadata| !metadata.is_object()) {
        return Err("every UI input must be an object".to_owned());
    }
    if inputs
        .keys()
        .any(|name| !declared_inputs.contains(name.as_str()))
    {
        return Err("UI inputs contains a slot not declared by the component".to_owned());
    }
    if manifest
        .inputs
        .iter()
        .any(|input| input.required && !inputs.contains_key(&input.name))
    {
        return Err("UI inputs must contain every required component input".to_owned());
    }
    let sections = form
        .get("sections")
        .and_then(Value::as_array)
        .filter(|sections| !sections.is_empty())
        .ok_or("davis.ui/v1 requires at least one section")?;
    let mut section_paths = std::collections::BTreeSet::new();
    for section in sections {
        let section = section
            .as_object()
            .ok_or("every UI section must be an object")?;
        let path = section.get("bind").and_then(Value::as_str).unwrap_or("");
        if path.is_empty() {
            return Err("every UI section requires bind".to_owned());
        }
        if !section_paths.insert(path) {
            return Err(format!("duplicate UI section bind `{path}`"));
        }
        if !schema_contains_pointer(config_schema, path) {
            return Err(format!(
                "UI section `{path}` does not exist in configuration.schema"
            ));
        }
        let widget = section
            .get("widget")
            .and_then(Value::as_str)
            .unwrap_or("auto");
        if let Some(input) = section.get("input").and_then(Value::as_str) {
            if !inputs.contains_key(input) {
                return Err(format!(
                    "UI section `{path}` refers to input `{input}` missing from inputs"
                ));
            }
        } else if matches!(widget, "column-map" | "utility-terms") {
            return Err(format!("UI widget `{widget}` at `{path}` requires input"));
        }
        if let Some(reference) = section.get("alternatives_from").and_then(Value::as_str) {
            if !schema_contains_pointer(config_schema, reference) {
                return Err(format!(
                    "UI section `{path}` has invalid alternatives_from `{reference}`"
                ));
            }
        }
        if let Some(reference) = section.get("parameters_from").and_then(Value::as_str) {
            if !schema_contains_pointer(config_schema, reference) {
                return Err(format!(
                    "UI section `{path}` has invalid parameters_from `{reference}`"
                ));
            }
        } else if widget == "parameter-settings" {
            return Err(format!(
                "UI widget `parameter-settings` at `{path}` requires parameters_from"
            ));
        }
        if widget == "nests" && section.get("alternatives_from").is_none() {
            return Err(format!(
                "UI widget `nests` at `{path}` requires alternatives_from"
            ));
        }
        if let Some(extension_id) = widget.strip_prefix("extension:") {
            let declared = ui_schema["extensions"]
                .as_array()
                .is_some_and(|extensions| {
                    extensions
                        .iter()
                        .any(|extension| extension["id"] == extension_id)
                });
            if !declared {
                return Err(format!(
                    "UI section `{path}` refers to undeclared extension `{extension_id}`"
                ));
            }
        }
        if let Some(context) = section.get("context") {
            let context = context
                .as_object()
                .ok_or_else(|| format!("UI section `{path}` context must be an object"))?;
            for (name, provider) in context {
                let provider = provider.as_object().ok_or_else(|| {
                    format!("UI section `{path}` context `{name}` must be an object")
                })?;
                match provider.get("provider").and_then(Value::as_str) {
                    Some("config") => {
                        let reference = provider.get("path").and_then(Value::as_str).unwrap_or("");
                        if !schema_contains_pointer(config_schema, reference) {
                            return Err(format!(
                                "UI section `{path}` context `{name}` has invalid config path `{reference}`"
                            ));
                        }
                    }
                    Some("columns") => validate_context_input(path, name, provider, inputs)?,
                    Some("distinct-values") => {
                        validate_context_input(path, name, provider, inputs)?;
                        let reference = provider
                            .get("column_from")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if !schema_contains_pointer(config_schema, reference) {
                            return Err(format!(
                                "UI section `{path}` context `{name}` has invalid column_from `{reference}`"
                            ));
                        }
                    }
                    Some(provider) => {
                        return Err(format!(
                            "UI section `{path}` context `{name}` uses unknown provider `{provider}`"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "UI section `{path}` context `{name}` requires provider"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_context_input(
    section_path: &str,
    context_name: &str,
    provider: &serde_json::Map<String, Value>,
    inputs: &serde_json::Map<String, Value>,
) -> Result<(), String> {
    let input = provider.get("input").and_then(Value::as_str).unwrap_or("");
    if !inputs.contains_key(input) {
        return Err(format!(
            "UI section `{section_path}` context `{context_name}` refers to unknown input `{input}`"
        ));
    }
    Ok(())
}

fn schema_contains_pointer(schema: &Value, path: &str) -> bool {
    path.trim_start_matches('/')
        .split('/')
        .try_fold(schema, |current, segment| {
            current.get("properties")?.get(segment)
        })
        .is_some()
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn render_analysis_plan(plan: Value) -> Result<String, String> {
    render_plan(plan)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn render_yaml_value(value: Value) -> Result<String, String> {
    serde_yaml::to_string(&value).map_err(|error| error.to_string())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn parse_yaml_value(yaml: String) -> Result<Value, String> {
    serde_yaml::from_str(&yaml).map_err(|error| error.to_string())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn save_analysis_plan(repository: PathBuf, path: PathBuf, plan: Value) -> Result<PathBuf, String> {
    let yaml = render_plan(plan)?;
    save_validated_yaml(&repository, &path, &yaml)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn save_analysis_plan_yaml(
    repository: PathBuf,
    path: PathBuf,
    yaml: String,
) -> Result<PathBuf, String> {
    let plan: AnalysisPlan = serde_yaml::from_str(&yaml)
        .map_err(|error| format!("invalid analysis plan YAML: {error}"))?;
    plan.validate().map_err(|error| error.to_string())?;
    save_validated_yaml(&repository, &path, &yaml)
}

fn save_validated_yaml(repository: &Path, path: &Path, yaml: &str) -> Result<PathBuf, String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| "analysis plan must have a parent directory".to_owned())?;
    if !parent.is_dir() {
        return Err(format!(
            "output directory does not exist: {}",
            parent.display()
        ));
    }
    let mut candidate = tempfile::Builder::new()
        .prefix(".davis-plan-")
        .suffix(".yaml")
        .tempfile_in(parent)
        .map_err(|error| format!("failed to create validation file: {error}"))?;
    candidate
        .write_all(yaml.as_bytes())
        .map_err(|error| format!("failed to write validation file: {error}"))?;
    validate_plan(repository, candidate.path()).map_err(|error| error.to_string())?;
    fs::write(path, yaml)
        .map_err(|error| format!("failed to save analysis plan {}: {error}", path.display()))?;
    Ok(path.to_owned())
}

fn render_plan(value: Value) -> Result<String, String> {
    let plan: AnalysisPlan =
        serde_json::from_value(value).map_err(|error| format!("invalid analysis plan: {error}"))?;
    plan.validate().map_err(|error| error.to_string())?;
    serde_yaml::to_string(&plan).map_err(|error| format!("failed to encode analysis plan: {error}"))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn validate_analysis_plan(repository: PathBuf, plan: PathBuf) -> Result<ValidPlanResponse, String> {
    ensure_arguments(&repository, &plan)?;
    let validated = validate_plan(&repository, &plan).map_err(|error| error.to_string())?;
    Ok(ValidPlanResponse {
        valid: true,
        plan: validated.plan_path,
        component: ComponentResponse {
            id: validated.manifest.id,
            version: validated.manifest.version,
            manifest: validated.manifest_path,
        },
    })
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn run_analysis_plan(repository: PathBuf, plan: PathBuf) -> Result<CompletedRun, String> {
    ensure_arguments(&repository, &plan)?;
    let run_root = repository.join("davis-runs");
    execute_plan(&repository, &plan, &run_root).map_err(|error| error.to_string())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn list_project_runs(repository: PathBuf) -> Result<RunHistoryResponse, String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    let run_root = repository.join("davis-runs");
    if !run_root.exists() {
        return Ok(RunHistoryResponse {
            runs: Vec::new(),
            warnings: Vec::new(),
        });
    }
    let entries = fs::read_dir(&run_root)
        .map_err(|error| format!("failed to read {}: {error}", run_root.display()))?;
    let mut runs = Vec::new();
    let mut warnings = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                warnings.push(format!(
                    "Run directory entryを読み込めませんでした: {error}"
                ));
                continue;
            }
        };
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let run_id = entry.file_name().to_string_lossy().into_owned();
        let run = match read_completed_run(&run_root, &run_id) {
            Ok(run) => run,
            Err(error) => {
                warnings.push(error);
                continue;
            }
        };
        let plan = AnalysisPlan::read(&entry.path().join("model.yaml")).ok();
        let modified_at_unix_ms = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| {
                u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
            });
        runs.push(RunHistoryEntry {
            plan_name: plan
                .as_ref()
                .map_or_else(|| run_id.clone(), |plan| plan.name.clone()),
            label: plan.as_ref().and_then(|plan| plan.run.label.clone()),
            tags: plan.map_or_else(Vec::new, |plan| plan.run.tags),
            run,
            modified_at_unix_ms,
        });
    }
    runs.sort_by(|left, right| {
        right
            .modified_at_unix_ms
            .cmp(&left.modified_at_unix_ms)
            .then_with(|| right.run.request.run_id.cmp(&left.run.request.run_id))
    });
    Ok(RunHistoryResponse { runs, warnings })
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn load_project_run(repository: PathBuf, run_id: String) -> Result<CompletedRun, String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    read_completed_run(&repository.join("davis-runs"), &run_id)
}

fn read_completed_run(run_root: &Path, run_id: &str) -> Result<CompletedRun, String> {
    if run_id.is_empty()
        || Path::new(run_id).components().count() != 1
        || run_id == "."
        || run_id == ".."
    {
        return Err(format!("invalid run id: {run_id}"));
    }
    let run_directory = run_root.join(run_id);
    let run_path = run_directory.join("run.json");
    let mut run: CompletedRun = serde_json::from_slice(
        &fs::read(&run_path)
            .map_err(|error| format!("failed to read {}: {error}", run_path.display()))?,
    )
    .map_err(|error| format!("invalid run record {}: {error}", run_path.display()))?;
    if run.request.run_id != run_id || run.result.run_id != run_id {
        return Err(format!(
            "run record {} does not match directory name `{run_id}`",
            run_path.display()
        ));
    }
    run.run_directory = run_directory;
    Ok(run)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn open_run_directory(repository: PathBuf, run_id: String) -> Result<(), String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    if run_id.is_empty()
        || Path::new(&run_id).components().count() != 1
        || run_id == "."
        || run_id == ".."
    {
        return Err("invalid run id".to_owned());
    }

    let run_directory = repository.join("davis-runs").join(run_id);
    if !run_directory.is_dir() || !run_directory.join("run.json").is_file() {
        return Err(format!(
            "run directory does not exist: {}",
            run_directory.display()
        ));
    }

    open::that_detached(&run_directory)
        .map_err(|error| format!("failed to open run directory: {error}"))
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn preview_run_artifact(
    repository: PathBuf,
    run_id: String,
    artifact: String,
) -> Result<ArtifactPreviewResponse, String> {
    const PREVIEW_BYTE_LIMIT: u64 = 5 * 1024 * 1024;
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    if run_id.is_empty()
        || Path::new(&run_id).components().count() != 1
        || run_id == "."
        || run_id == ".."
    {
        return Err("invalid run id".to_owned());
    }
    let run_directory = repository.join("davis-runs").join(&run_id);
    let result_path = run_directory.join("result.json");
    let result: RunResult = serde_json::from_slice(
        &fs::read(&result_path)
            .map_err(|error| format!("failed to read {}: {error}", result_path.display()))?,
    )
    .map_err(|error| format!("invalid run result {}: {error}", result_path.display()))?;
    let descriptor = result
        .artifacts
        .get(&artifact)
        .or_else(|| result.extensions.get(&artifact))
        .ok_or_else(|| format!("run `{run_id}` does not contain artifact `{artifact}`"))?;
    if descriptor.path.is_absolute()
        || descriptor.path.components().any(|part| {
            matches!(
                part,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(format!("artifact `{artifact}` has an unsafe path"));
    }
    let artifact_root = run_directory.join("artifacts");
    let path = artifact_root.join(&descriptor.path);
    let canonical_root = fs::canonicalize(&artifact_root)
        .map_err(|error| format!("failed to access {}: {error}", artifact_root.display()))?;
    let canonical_path = fs::canonicalize(&path)
        .map_err(|error| format!("failed to access {}: {error}", path.display()))?;
    if !canonical_path.starts_with(canonical_root) {
        return Err(format!(
            "artifact `{artifact}` resolves outside its run directory"
        ));
    }
    let metadata = fs::metadata(&path)
        .map_err(|error| format!("failed to access {}: {error}", path.display()))?;
    if metadata.len() > PREVIEW_BYTE_LIMIT {
        return Err(format!("artifact `{artifact}` is too large to preview"));
    }
    let bytes =
        fs::read(&path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let content = match descriptor.media_type.as_str() {
        "application/json" => serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid JSON artifact {}: {error}", path.display()))?,
        "text/csv" => preview_csv(&path, &bytes)?,
        media_type => {
            return Err(format!(
                "artifact media type `{media_type}` cannot be previewed"
            ))
        }
    };
    Ok(ArtifactPreviewResponse {
        name: artifact,
        media_type: descriptor.media_type.clone(),
        content,
    })
}

fn preview_csv(path: &Path, bytes: &[u8]) -> Result<Value, String> {
    const ROW_LIMIT: usize = 200;
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(bytes);
    let columns: Vec<String> = reader
        .headers()
        .map_err(|error| format!("invalid CSV artifact {}: {error}", path.display()))?
        .iter()
        .map(str::to_owned)
        .collect();
    let mut rows = Vec::new();
    let mut truncated = false;
    for record in reader.records() {
        let record =
            record.map_err(|error| format!("invalid CSV artifact {}: {error}", path.display()))?;
        if rows.len() == ROW_LIMIT {
            truncated = true;
            break;
        }
        rows.push(record.iter().map(str::to_owned).collect::<Vec<_>>());
    }
    Ok(serde_json::json!({ "columns": columns, "rows": rows, "truncated": truncated }))
}

fn ensure_arguments(repository: &Path, plan: &Path) -> Result<(), String> {
    if !repository.is_dir() {
        return Err(format!(
            "repository does not exist: {}",
            repository.display()
        ));
    }
    if !plan.is_file() {
        return Err(format!("analysis plan does not exist: {}", plan.display()));
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            catalog_files,
            download_catalog_file,
            inspect_csv_file,
            inspect_distinct_values,
            component_editor_definition,
            component_editor_definitions,
            load_analysis_plan_for_editing,
            render_analysis_plan,
            render_yaml_value,
            parse_yaml_value,
            save_analysis_plan,
            save_analysis_plan_yaml,
            validate_analysis_plan,
            run_analysis_plan,
            list_project_runs,
            load_project_run,
            open_run_directory,
            preview_run_artifact
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Davis desktop application");
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::{
        component_editor_definitions, editor_definition, editor_response, list_project_runs,
        load_analysis_plan_for_editing, load_project_run, normalize_editor_presentation,
        parse_yaml_value, preview_csv, render_plan, render_yaml_value, save_analysis_plan_yaml,
        validate_editor_presentation, ComponentManifest, Value,
    };

    fn repository() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    #[test]
    fn lists_and_loads_persisted_runs_without_trusting_recorded_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let run_directory = temporary.path().join("davis-runs/example-run");
        std::fs::create_dir_all(&run_directory).unwrap();
        std::fs::write(
            run_directory.join("model.yaml"),
            r"api_version: davis.analysis/v1alpha1
name: comparison-plan
component: {id: example/model, version: 1.0.0, operation: estimate}
inputs:
  data: {kind: local, path: data.csv}
config: {}
run:
  label: Baseline
  tags: [comparison]
",
        )
        .unwrap();
        std::fs::write(
            run_directory.join("run.json"),
            serde_json::to_vec(&json!({
                "run_directory": "/untrusted/old/location",
                "request": {
                    "api_version": "davis.run/v1alpha1",
                    "run_id": "example-run",
                    "operation": "estimate",
                    "component": {
                        "id": "example/model",
                        "version": "1.0.0",
                        "kind": "model",
                        "manifest_path": "/component.yaml",
                        "source_digest": "blake3:0000000000000000000000000000000000000000000000000000000000000000"
                    },
                    "inputs": {},
                    "config": {},
                    "output_directory": "/untrusted/old/location/artifacts"
                },
                "result": {
                    "api_version": "davis.result/v1alpha1",
                    "run_id": "example-run",
                    "status": "succeeded",
                    "artifacts": {},
                    "extensions": {},
                    "error": null
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let broken_directory = temporary.path().join("davis-runs/broken-run");
        std::fs::create_dir_all(&broken_directory).unwrap();
        std::fs::write(broken_directory.join("run.json"), b"not JSON").unwrap();

        let history = list_project_runs(temporary.path().to_owned()).unwrap();
        assert_eq!(history.warnings.len(), 1);
        assert!(history.warnings[0].contains("broken-run"));
        assert!(history.warnings[0].contains("run.json"));
        assert_eq!(history.runs.len(), 1);
        assert_eq!(history.runs[0].plan_name, "comparison-plan");
        assert_eq!(history.runs[0].label.as_deref(), Some("Baseline"));
        assert_eq!(history.runs[0].tags, ["comparison"]);
        assert_eq!(history.runs[0].run.run_directory, run_directory);

        let loaded =
            load_project_run(temporary.path().to_owned(), "example-run".to_owned()).unwrap();
        assert_eq!(loaded.run_directory, run_directory);
        assert!(load_project_run(temporary.path().to_owned(), "../escape".to_owned()).is_err());
    }

    #[test]
    fn renders_a_typed_multi_source_plan() {
        let yaml = render_plan(json!({
            "api_version": "davis.analysis/v1alpha1",
            "name": "gui-plan",
            "component": {"id": "davis/mnl", "version": "0.3.1", "operation": "estimate"},
            "inputs": {"choice_data": {
                "kind": "table_binding",
                "processor": {"id": "davis/csv-transform", "version": "0.4.0"},
                "sources": {
                    "choices": {"kind": "local", "path": "/tmp/choices.csv"},
                    "persons": {"kind": "local", "path": "/tmp/persons.csv"}
                },
                "base": "choices",
                "joins": [{
                    "source": "persons", "left_on": "case_id", "right_on": "person_id",
                    "relationship": "many_to_one", "how": "left", "allow_unmatched": false
                }],
                "columns": {
                    "case_id": {"source": "choices", "column": "case_id"},
                    "income": {"source": "persons", "column": "income"}
                }
            }},
            "config": {
                "roles": {"case_id": "case_id", "alternative_id": "alternative", "chosen": "chosen"},
                "terms": [{"parameter": "beta_income", "column": "income"}]
            }
        }))
        .unwrap();

        assert!(yaml.contains("kind: table_binding"));
        assert!(yaml.contains("beta_income"));
    }

    #[test]
    fn rejects_an_invalid_binding() {
        let error = render_plan(json!({
            "api_version": "davis.analysis/v1alpha1",
            "name": "invalid",
            "component": {"id": "davis/mnl", "version": "0.3.1", "operation": "estimate"},
            "inputs": {"choice_data": {
                "kind": "table_binding",
                "processor": {"id": "davis/csv-transform", "version": "0.4.0"},
                "sources": {"choices": {"kind": "local", "path": "/tmp/choices.csv"}},
                "base": "missing",
                "columns": {"case_id": {"source": "choices", "column": "case_id"}}
            }}
        }))
        .unwrap_err();

        assert!(error.contains("does not contain base source"));
    }

    #[test]
    fn loads_manifest_driven_editor_metadata() {
        let editor = editor_definition(&repository(), "davis/mnl", "0.3.1").unwrap();
        assert_eq!(editor.manifest.id, "davis/mnl");
        assert_eq!(editor.ui_schema["version"], "davis.ui/v1");
        assert_eq!(
            editor.ui_schema["inputs"]["choice_data"]["widget"],
            "table-binding"
        );
        assert!(editor.config_schema["properties"]["roles"]["required"].is_array());
        assert!(
            editor.config_schema["properties"]["roles"]["properties"]["case_id"]["oneOf"]
                .is_array()
        );
        let editors = component_editor_definitions(repository()).unwrap();
        assert!(editors.iter().any(|item| item.manifest.id == "davis/mnl"));
        assert!(editors.iter().any(|item| item.manifest.id == "davis/nl"));
        assert!(editors.iter().any(|item| item.manifest.id == "davis/rl"));
    }

    #[test]
    fn adapts_the_legacy_linear_editor_without_model_specific_logic() {
        let manifest_path = repository().join("components/davis-mnl/component.yaml");
        let manifest = ComponentManifest::read(&manifest_path).unwrap();
        let config = manifest
            .resolve_configuration(&manifest_path)
            .unwrap()
            .value;
        let legacy = json!({
            "ui:editor": "linear-utility",
            "ui:inputPreparation": {"component": "davis/csv-transform", "version": "0.4.0"},
            "roles": {"ui:labels": {"case_id": "ケースID"}},
            "ui:results": []
        });

        let normalized = normalize_editor_presentation(&manifest, &config, legacy).unwrap();

        assert_eq!(normalized["version"], "davis.ui/v1");
        assert_eq!(
            normalized["inputs"]["choice_data"]["widget"],
            "table-binding"
        );
        assert!(normalized["sections"]
            .as_array()
            .unwrap()
            .iter()
            .any(|section| section["widget"] == "utility-terms"));
    }

    #[test]
    fn round_trips_a_section_yaml_value() {
        let value = json!({"initial": -1.0, "fixed": false});
        let yaml = render_yaml_value(value.clone()).unwrap();
        assert_eq!(parse_yaml_value(yaml).unwrap(), value);
    }

    #[test]
    fn loads_schema_forms_for_nested_and_recursive_logit() {
        let repository = repository();
        let nested = editor_definition(&repository, "davis/nl", "0.1.1").unwrap();
        assert_eq!(nested.ui_schema["version"], "davis.ui/v1");
        assert!(nested.ui_schema["sections"]
            .as_array()
            .unwrap()
            .iter()
            .any(|section| section["widget"] == "extension:nest-editor"));
        let nest_section = nested.ui_schema["sections"]
            .as_array()
            .unwrap()
            .iter()
            .find(|section| section["bind"] == "/nests")
            .unwrap();
        assert!(nest_section["description"]["ja"]
            .as_str()
            .unwrap()
            .contains("最上位scaleは1"));
        assert_eq!(
            nest_section["labels"]["estimate"]["en"],
            "Estimate (initial value at right)"
        );
        assert_eq!(
            nest_section["context"]["alternatives"]["provider"],
            "distinct-values"
        );
        assert_eq!(
            nest_section["context"]["alternatives"]["input"],
            "choice_data"
        );
        assert_eq!(
            nested.ui_extensions["nest-editor"].api_version,
            "davis.widget/v1"
        );
        assert!(nested.ui_extensions["nest-editor"]
            .html
            .contains("set-value"));

        let recursive = editor_definition(&repository, "davis/rl", "0.1.1").unwrap();
        assert_eq!(recursive.ui_schema["version"], "davis.ui/v1");
        assert_eq!(recursive.ui_schema["inputs"].as_object().unwrap().len(), 2);
        assert!(recursive.ui_schema["sections"]
            .as_array()
            .unwrap()
            .iter()
            .any(|section| section["widget"] == "parameter-settings"));

        let nested_plan = repository.join("components/davis-nl/examples/minimal/model.yaml");
        let loaded_nested =
            load_analysis_plan_for_editing(repository.clone(), nested_plan).unwrap();
        assert!(loaded_nested.resolved_sources["choice_data"].is_file());

        let recursive_plan = repository.join("components/davis-rl/examples/minimal/model.yaml");
        let loaded_recursive = load_analysis_plan_for_editing(repository, recursive_plan).unwrap();
        assert!(loaded_recursive.resolved_sources["network"].is_file());
        assert!(loaded_recursive.resolved_sources["observations"].is_file());
    }

    #[test]
    fn preserves_an_unknown_widget_for_section_fallback() {
        let manifest_path = repository().join("components/davis-nl/component.yaml");
        let manifest = ComponentManifest::read(&manifest_path).unwrap();
        let mut ui = manifest
            .resolve_presentation(&manifest_path)
            .unwrap()
            .unwrap()
            .value;
        ui["sections"][0]["widget"] = Value::String("mystery".to_owned());

        let config = manifest
            .resolve_configuration(&manifest_path)
            .unwrap()
            .value;
        validate_editor_presentation(&manifest, &config, &ui).unwrap();
    }

    #[test]
    fn loads_inline_editor_metadata_from_one_manifest_file() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_path = temporary.path().join("component.yaml");
        std::fs::write(
            &manifest_path,
            r"api_version: davis.component/v1
id: example/inline-editor
name: Inline editor
version: 1.0.0
runtime:
  kind: native
  command: [example]
operations: [estimate]
inputs:
  - name: table
    media_types: [text/csv]
configuration:
  schema:
    type: object
    properties:
      scale: {type: number}
presentation:
  ui:
    ui:editor: schema-form
    ui:form:
      inputs:
        table: {title: Table}
      sections:
        - {path: scale, widget: object}
outputs: {}
",
        )
        .unwrap();
        let manifest = davis_model_api::ComponentManifest::read(&manifest_path).unwrap();

        let editor = editor_response(&manifest_path, manifest).unwrap();

        assert_eq!(
            editor.config_schema["properties"]["scale"]["type"],
            "number"
        );
        assert_eq!(editor.ui_schema["version"], "davis.ui/v1");
        assert_eq!(editor.ui_schema["sections"][0]["bind"], "/scale");
    }

    #[test]
    fn loads_an_existing_plan_with_absolute_local_sources() {
        let repository = repository();
        let plan = repository.join("components/davis-mnl/examples/multi-source/model.yaml");
        let loaded = load_analysis_plan_for_editing(repository, plan).unwrap();
        assert!(loaded.yaml.contains("multi-source-mode-choice"));
        let path = &loaded.resolved_sources["choice_data/choices"];
        assert!(path.is_absolute());
        assert!(path.is_file());
        let input = &loaded.plan.inputs["choice_data"];
        let davis_model_api::InputSource::TableBinding { binding } = input else {
            panic!("expected table binding");
        };
        let davis_model_api::InputSource::Local { path, .. } = &binding.sources["choices"] else {
            panic!("expected local source");
        };
        assert!(path.is_relative());
    }

    #[test]
    fn namespaces_sources_of_multiple_table_binding_inputs() {
        let temporary = tempfile::tempdir().unwrap();
        let network = repository().join("components/davis-rl/examples/minimal/network.csv");
        let observations =
            repository().join("components/davis-rl/examples/minimal/observations.csv");
        let plan = temporary.path().join("model.yaml");
        std::fs::write(
            &plan,
            format!(
                r"api_version: davis.analysis/v1alpha1
name: namespaced-bindings
component: {{id: davis/rl, version: 0.1.1, operation: estimate}}
inputs:
  network:
    kind: table_binding
    processor: {{id: davis/csv-transform, version: 0.4.0}}
    sources: {{data: {{kind: local, path: {}}}}}
    base: data
    columns: {{link_id: {{source: data, column: link_id}}}}
  observations:
    kind: table_binding
    processor: {{id: davis/csv-transform, version: 0.4.0}}
    sources: {{data: {{kind: local, path: {}}}}}
    base: data
    columns: {{trip_id: {{source: data, column: trip_id}}}}
config: {{}}
",
                network.display(),
                observations.display()
            ),
        )
        .unwrap();

        let loaded = load_analysis_plan_for_editing(repository(), plan).unwrap();

        assert_eq!(loaded.resolved_sources["network/data"], network);
        assert_eq!(loaded.resolved_sources["observations/data"], observations);
    }

    #[test]
    fn invalid_yaml_does_not_overwrite_an_existing_plan() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("model.yaml");
        std::fs::write(&target, "original").unwrap();
        let choices = repository().join("components/davis-mnl/examples/multi-source/choices.csv");
        let yaml = format!(
            "api_version: davis.analysis/v1alpha1\nname: invalid\ncomponent:\n  id: davis/mnl\n  version: 0.3.1\n  operation: estimate\ninputs:\n  choice_data:\n    kind: local\n    path: {}\nconfig:\n  roles:\n    case_id: case_id\n    alternative_id: alternative\n    chosen: chosen\n",
            choices.display()
        );
        let error = save_analysis_plan_yaml(repository(), target.clone(), yaml).unwrap_err();
        assert!(error.contains("terms"));
        assert_eq!(std::fs::read_to_string(target).unwrap(), "original");
    }

    #[test]
    fn previews_csv_artifacts_as_a_bounded_table() {
        let content = preview_csv(
            PathBuf::from("parameters.csv").as_path(),
            b"name,estimate\nbeta,-1.25\n",
        )
        .unwrap();
        assert_eq!(content["columns"], json!(["name", "estimate"]));
        assert_eq!(content["rows"], json!([["beta", "-1.25"]]));
        assert_eq!(content["truncated"], false);
    }
}
