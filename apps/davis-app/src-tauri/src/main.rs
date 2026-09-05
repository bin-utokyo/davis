use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

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
        .filter(|editor| editor.ui_schema["ui:editor"] == "linear-utility")
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
        collect_local_sources(source, slot, plan_directory, &mut resolved_sources);
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

fn collect_local_sources(
    source: &InputSource,
    name: &str,
    plan_directory: &Path,
    resolved: &mut BTreeMap<String, PathBuf>,
) {
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
                collect_local_sources(nested, nested_name, plan_directory, resolved);
            }
        }
        InputSource::Catalog { .. } | InputSource::RunArtifact { .. } => {}
    }
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
    Ok(ComponentEditorResponse {
        manifest,
        config_schema,
        ui_schema,
    })
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn render_analysis_plan(plan: Value) -> Result<String, String> {
    render_plan(plan)
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
            inspect_csv_file,
            inspect_distinct_values,
            component_editor_definition,
            component_editor_definitions,
            load_analysis_plan_for_editing,
            render_analysis_plan,
            save_analysis_plan,
            save_analysis_plan_yaml,
            validate_analysis_plan,
            run_analysis_plan,
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
        component_editor_definitions, editor_definition, editor_response,
        load_analysis_plan_for_editing, preview_csv, render_plan, save_analysis_plan_yaml,
    };

    fn repository() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    #[test]
    fn renders_a_typed_multi_source_plan() {
        let yaml = render_plan(json!({
            "api_version": "davis.analysis/v1alpha1",
            "name": "gui-plan",
            "component": {"id": "davis/mnl", "version": "0.2.0", "operation": "estimate"},
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
            "component": {"id": "davis/mnl", "version": "0.2.0", "operation": "estimate"},
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
        let editor = editor_definition(&repository(), "davis/mnl", "0.2.0").unwrap();
        assert_eq!(editor.manifest.id, "davis/mnl");
        assert_eq!(editor.ui_schema["ui:editor"], "linear-utility");
        assert!(editor.config_schema["properties"]["roles"]["required"].is_array());
        let editors = component_editor_definitions(repository()).unwrap();
        assert!(editors.iter().any(|item| item.manifest.id == "davis/mnl"));
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
inputs: []
configuration:
  schema:
    type: object
    properties:
      scale: {type: number}
presentation:
  ui:
    ui:editor: generic
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
        assert_eq!(editor.ui_schema["ui:editor"], "generic");
    }

    #[test]
    fn loads_an_existing_plan_with_absolute_local_sources() {
        let repository = repository();
        let plan = repository.join("components/davis-mnl/examples/multi-source/model.yaml");
        let loaded = load_analysis_plan_for_editing(repository, plan).unwrap();
        assert!(loaded.yaml.contains("multi-source-mode-choice"));
        let path = &loaded.resolved_sources["choices"];
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
    fn invalid_yaml_does_not_overwrite_an_existing_plan() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("model.yaml");
        std::fs::write(&target, "original").unwrap();
        let choices = repository().join("components/davis-mnl/examples/multi-source/choices.csv");
        let yaml = format!(
            "api_version: davis.analysis/v1alpha1\nname: invalid\ncomponent:\n  id: davis/mnl\n  version: 0.2.0\n  operation: estimate\ninputs:\n  choice_data:\n    kind: local\n    path: {}\nconfig:\n  roles:\n    case_id: case_id\n    alternative_id: alternative\n    chosen: chosen\n",
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
