use std::fs;
use std::path::{Path, PathBuf};

use davis_model_api::AnalysisPlan;
use davis_runtime::{execute_plan, inspect_csv, validate_plan, CompletedRun, CsvProfile};
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

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn inspect_csv_file(path: PathBuf) -> Result<CsvProfile, String> {
    inspect_csv(&path).map_err(|error| error.to_string())
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn render_analysis_plan(plan: Value) -> Result<String, String> {
    render_plan(plan)
}

#[tauri::command]
#[allow(clippy::needless_pass_by_value)]
fn save_analysis_plan(path: PathBuf, plan: Value) -> Result<PathBuf, String> {
    let yaml = render_plan(plan)?;
    let parent = path
        .parent()
        .ok_or_else(|| "analysis plan must have a parent directory".to_owned())?;
    if !parent.is_dir() {
        return Err(format!(
            "output directory does not exist: {}",
            parent.display()
        ));
    }
    fs::write(&path, yaml)
        .map_err(|error| format!("failed to save analysis plan {}: {error}", path.display()))?;
    Ok(path)
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
            render_analysis_plan,
            save_analysis_plan,
            validate_analysis_plan,
            run_analysis_plan,
            open_run_directory
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Davis desktop application");
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::render_plan;

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
}
