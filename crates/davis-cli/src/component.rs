use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use davis_model_api::{
    ArtifactDeclaration, ArtifactProfile, ComponentInput, ComponentKind, ComponentManifest,
    ConfigurationDeclaration, OutputDeclaration, PresentationDeclaration, RuntimeDeclaration,
    RuntimeExecutor, RuntimeRequirement, COMPONENT_API_VERSION, COMPONENT_MANIFEST_FILENAME,
};
use davis_runtime::{
    validate_component_package, ComponentStore, InstalledComponent, ValidatedComponentPackage,
};
use serde::Serialize;
use serde_json::json;

use crate::{ComponentCommand, ScaffoldKind, ScaffoldTemplate};

#[derive(Debug, Serialize)]
struct ScaffoldedComponent {
    path: PathBuf,
    manifest_path: PathBuf,
    example_plan: Option<PathBuf>,
    id: String,
    version: String,
}

pub(crate) async fn handle_install(
    source: String,
    version: Option<String>,
    registry: Option<String>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = ComponentStore::for_user()?;
    let path = PathBuf::from(&source);
    let installed = if path.exists() || looks_like_explicit_path(&source, &path) {
        if version.is_some() || registry.is_some() {
            return Err(
                "--version and --registry can only be used with an official component name".into(),
            );
        }
        store.install(&path)?
    } else {
        let downloaded =
            crate::component_registry::download(&source, version.as_deref(), registry.as_deref())
                .await?;
        let origin = format!("registry:{}@{}", downloaded.id(), downloaded.version());
        store.install_with_origin(downloaded.path(), Some(origin))?
    };
    print_installed(&installed, json, "Installed")?;
    if !json {
        if let Some(example) = minimal_example_plan(&installed.path) {
            println!("Example plan: {}", example.display());
            println!("Try it: davis model run {}", example.display());
        }
    }
    Ok(())
}

fn minimal_example_plan(component: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(component.join("examples/minimal")).ok()?;
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|extension| {
                extension.eq_ignore_ascii_case("yaml") || extension.eq_ignore_ascii_case("yml")
            })
        })
        .min()
}

pub(crate) fn handle_component(
    command: ComponentCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ComponentCommand::Scaffold {
            path,
            id,
            name,
            kind,
            template,
            runtime_command,
            operations,
            json,
        } => {
            let scaffolded =
                scaffold_component(&path, id, name, kind, template, runtime_command, operations)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&scaffolded)?);
            } else {
                println!("Created: {}", scaffolded.path.display());
                println!("Manifest: {}", scaffolded.manifest_path.display());
                if let Some(example_plan) = &scaffolded.example_plan {
                    println!("Example plan: {}", example_plan.display());
                    println!("Next: run `davis model run {}`", example_plan.display());
                } else {
                    println!(
                        "Next: add the program, then run `davis component validate {}`",
                        scaffolded.path.display()
                    );
                }
            }
        }
        ComponentCommand::Validate { path, json } => {
            let validated = validate_component_package(&path)?;
            print_validated(&validated, json)?;
        }
        ComponentCommand::List { json } => {
            let store = ComponentStore::for_user()?;
            let installed = store.list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&installed)?);
            } else if installed.is_empty() {
                println!("No model components are installed.");
            } else {
                for component in installed {
                    println!(
                        "{} {}\t{}",
                        component.id,
                        component.version,
                        component.path.display()
                    );
                }
            }
        }
        ComponentCommand::Inspect { id, version, json } => {
            let store = ComponentStore::for_user()?;
            let installed = store.inspect(&id, version.as_deref())?;
            print_installed(&installed, json, "Component")?;
        }
        ComponentCommand::Remove { id, version, json } => {
            let store = ComponentStore::for_user()?;
            let removed = store.remove(&id, version.as_deref())?;
            print_installed(&removed, json, "Removed")?;
        }
        ComponentCommand::Pack {
            path,
            out,
            name,
            requires_davis,
            json,
        } => {
            let packed = crate::component_pack::pack(
                &path,
                &out,
                name.as_deref(),
                requires_davis.as_deref(),
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&packed)?);
            } else {
                println!("Bundle: {}", packed.bundle_path.display());
                println!("Entry: {}", packed.entry_path.display());
                println!("Digest: {}", packed.entry.bundle.blake3);
            }
        }
        ComponentCommand::Registry { entries, out, json } => {
            let registry = crate::component_pack::registry(&entries, &out)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&registry)?);
            } else {
                println!("Registry: {}", out.display());
                println!("Components: {}", registry.components.len());
            }
        }
    }
    Ok(())
}

fn scaffold_component(
    path: &Path,
    id: String,
    name: Option<String>,
    kind: ScaffoldKind,
    template: Option<ScaffoldTemplate>,
    runtime_command: Vec<String>,
    operations: Vec<String>,
) -> Result<ScaffoldedComponent, Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!("scaffold destination already exists: {}", path.display()).into());
    }
    let runtime_command = if template.is_some() {
        vec!["python3".to_owned(), "component.py".to_owned()]
    } else {
        runtime_command
    };
    let mut manifest = build_scaffold_manifest(id, name, kind, runtime_command, operations);
    if template.is_some() {
        apply_python_template(&mut manifest);
    }
    manifest.validate()?;

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path)?;
    let manifest_path = path.join(COMPONENT_MANIFEST_FILENAME);
    let write_result = write_scaffold_files(path, &manifest_path, &manifest, template);
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(path);
        return Err(error);
    }
    let validated = match validate_component_package(path) {
        Ok(validated) => validated,
        Err(error) => {
            let _ = fs::remove_dir_all(path);
            return Err(error.into());
        }
    };
    let example_plan = template.map(|_| {
        fs::canonicalize(path.join("examples/minimal/analysis.yaml"))
            .unwrap_or_else(|_| path.join("examples/minimal/analysis.yaml"))
    });
    Ok(ScaffoldedComponent {
        path: validated.source,
        manifest_path: validated.manifest_path,
        example_plan,
        id: validated.manifest.id,
        version: validated.manifest.version,
    })
}

fn build_scaffold_manifest(
    id: String,
    name: Option<String>,
    kind: ScaffoldKind,
    runtime_command: Vec<String>,
    operations: Vec<String>,
) -> ComponentManifest {
    let name = name.unwrap_or_else(|| {
        id.rsplit('/')
            .next()
            .filter(|segment| !segment.is_empty())
            .unwrap_or(&id)
            .to_owned()
    });
    let kind = match kind {
        ScaffoldKind::Model => ComponentKind::Model,
        ScaffoldKind::Transform => ComponentKind::Transform,
        ScaffoldKind::Visualize => ComponentKind::Visualize,
    };
    let operations = if operations.is_empty() {
        vec![match kind {
            ComponentKind::Model => "estimate",
            ComponentKind::Transform => "transform",
            ComponentKind::Visualize => "visualize",
        }
        .to_owned()]
    } else {
        operations
    };
    ComponentManifest {
        api_version: COMPONENT_API_VERSION.to_owned(),
        id,
        name,
        version: "0.1.0".to_owned(),
        kind,
        requires_davis: Some(">=0.5.0".to_owned()),
        runtime: RuntimeDeclaration {
            executor: RuntimeExecutor::Process,
            kind: None,
            command: runtime_command,
            request_argument: "--request".to_owned(),
            lockfile: None,
            requirements: Vec::new(),
        },
        operations,
        inputs: Vec::new(),
        additional_inputs: None,
        configuration: Some(ConfigurationDeclaration {
            schema: Some(json!({
                "type": "object",
                "additionalProperties": false
            })),
            schema_ref: None,
        }),
        presentation: Some(PresentationDeclaration {
            ui: Some(json!({
                "version": "davis.ui/v1",
                "inputs": {},
                "sections": [],
                "results": []
            })),
            ui_ref: None,
        }),
        config_schema: None,
        ui_schema: None,
        outputs: OutputDeclaration {
            standard: Vec::new(),
            extensions: Vec::new(),
            artifacts: BTreeMap::new(),
        },
    }
}

fn apply_python_template(manifest: &mut ComponentManifest) {
    manifest.inputs = vec![ComponentInput {
        name: "table".to_owned(),
        media_types: vec!["text/csv".to_owned()],
        required: true,
    }];
    manifest.runtime.requirements = vec![RuntimeRequirement {
        command: "python3".to_owned(),
        version: None,
        version_arguments: vec!["--version".to_owned()],
        install: BTreeMap::new(),
    }];
    manifest.configuration = Some(ConfigurationDeclaration {
        schema: Some(json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["columns"],
            "properties": {
                "columns": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["value"],
                    "properties": {"value": {"type": "string"}}
                }
            }
        })),
        schema_ref: None,
    });
    manifest.presentation = Some(PresentationDeclaration {
        ui: Some(json!({
            "version": "davis.ui/v1",
            "inputs": {
                "table": {
                    "title": "入力CSV",
                    "description": "処理するCSVを選びます．",
                    "widget": "table-binding"
                }
            },
            "sections": [{
                "bind": "/columns",
                "widget": "column-map",
                "input": "table",
                "title": "役割列",
                "labels": {"value": "集計する数値列"}
            }],
            "results": [
                {"artifact": "summary", "title": "集計結果", "widget": "key-value"},
                {"artifact": "output_table", "title": "出力表", "widget": "table"}
            ]
        })),
        ui_ref: None,
    });
    manifest.outputs.artifacts.insert(
        "output_table".to_owned(),
        ArtifactDeclaration {
            media_types: vec!["text/csv".to_owned()],
            required: true,
            profile: Some(ArtifactProfile::Table),
        },
    );
    manifest.outputs.artifacts.insert(
        "summary".to_owned(),
        ArtifactDeclaration {
            media_types: vec!["application/json".to_owned()],
            required: true,
            profile: Some(ArtifactProfile::Metrics),
        },
    );
}

fn write_scaffold_files(
    path: &Path,
    manifest_path: &Path,
    manifest: &ComponentManifest,
    template: Option<ScaffoldTemplate>,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(manifest_path, serde_yaml::to_string(manifest)?)?;
    if template.is_none() {
        return Ok(());
    }
    fs::write(path.join("component.py"), PYTHON_COMPONENT)?;
    let examples = path.join("examples/minimal");
    fs::create_dir_all(&examples)?;
    fs::write(examples.join("input.csv"), "id,value\n1,10\n2,20\n3,30\n")?;
    let plan = json!({
        "api_version": "davis.analysis/v1alpha1",
        "name": "minimal-example",
        "component": {
            "id": manifest.id,
            "version": manifest.version,
            "operation": manifest.operations[0]
        },
        "inputs": {"table": {"kind": "local", "path": "input.csv"}},
        "config": {"columns": {"value": "value"}},
        "run": {"label": "minimal-example", "tags": ["example", "scaffold"]}
    });
    fs::write(
        examples.join("analysis.yaml"),
        serde_yaml::to_string(&plan)?,
    )?;
    fs::write(
        path.join("README.md"),
        format!(
            "# {}\n\nThis runnable scaffold demonstrates the Davis process contract.\n\n## Try it\n\n```console\ndavis component validate .\ndavis model run examples/minimal/analysis.yaml\n```\n\nEdit `component.yaml` to describe your inputs, settings, and outputs. Replace the calculation in `component.py`, but keep reading the resolved input paths from `request.json` and writing `run-result.json`.\n",
            manifest.name
        ),
    )?;
    Ok(())
}

const PYTHON_COMPONENT: &str = r#"from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    source = Path(request["inputs"]["table"]["resolved"]["path"])
    output = Path(request["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    value_column = request["config"]["columns"]["value"]
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or value_column not in rows[0]:
        raise ValueError(f"column was not found: {value_column}")
    values = [float(row[value_column]) for row in rows]
    shutil.copyfile(source, output / "output.csv")
    (output / "summary.json").write_text(
        json.dumps({"rows": len(rows), "sum": sum(values)}, indent=2),
        encoding="utf-8",
    )
    result = {
        "api_version": "davis.result/v1alpha1",
        "run_id": request["run_id"],
        "status": "succeeded",
        "artifacts": {
            "output_table": {"path": "output.csv", "media_type": "text/csv"},
            "summary": {"path": "summary.json", "media_type": "application/json"},
        },
        "extensions": {},
    }
    (output / "run-result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
"#;

fn print_validated(
    package: &ValidatedComponentPackage,
    json: bool,
) -> Result<(), serde_json::Error> {
    if json {
        println!("{}", serde_json::to_string_pretty(package)?);
    } else {
        println!(
            "Valid: {} {}",
            package.manifest.id, package.manifest.version
        );
        println!("Manifest: {}", package.manifest_path.display());
        println!("Configuration: {}", package.configuration_source.display());
        if let Some(path) = &package.presentation_source {
            println!("Presentation: {}", path.display());
        }
        println!("Command: {}", package.manifest.runtime.command.join(" "));
    }
    Ok(())
}

fn print_installed(
    component: &InstalledComponent,
    json: bool,
    label: &str,
) -> Result<(), serde_json::Error> {
    if json {
        println!("{}", serde_json::to_string_pretty(component)?);
    } else {
        println!("{label}: {} {}", component.id, component.version);
        println!("Name: {}", component.name);
        println!("Kind: {:?}", component.kind);
        println!("Path: {}", component.path.display());
        println!("Source: {}", component.source);
        println!("Digest: {}", component.source_digest);
    }
    Ok(())
}

fn looks_like_explicit_path(source: &str, path: &std::path::Path) -> bool {
    path.is_absolute()
        || source == "."
        || source == ".."
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with("~/")
        || source.starts_with(".\\")
        || source.starts_with("..\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_bundled_minimal_example_plan() {
        let component =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../components/davis-mnl");
        assert_eq!(
            minimal_example_plan(&component),
            Some(component.join("examples/minimal/model.yaml"))
        );
    }

    #[test]
    fn scaffolds_a_valid_self_contained_component_without_overwriting() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("example-component");

        let scaffolded = scaffold_component(
            &path,
            "example/calculator".to_owned(),
            None,
            ScaffoldKind::Transform,
            None,
            vec!["calculator".to_owned()],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(scaffolded.path, fs::canonicalize(&path).unwrap());
        let validated = validate_component_package(&path).unwrap();
        assert_eq!(validated.manifest.name, "calculator");
        assert_eq!(validated.manifest.kind, ComponentKind::Transform);
        assert_eq!(validated.manifest.operations, ["transform"]);
        assert!(matches!(
            scaffold_component(
                &path,
                "example/replacement".to_owned(),
                None,
                ScaffoldKind::Model,
                None,
                vec!["replacement".to_owned()],
                Vec::new(),
            ),
            Err(error) if error.to_string().contains("already exists")
        ));

        let invalid_path = temporary.path().join("invalid-component");
        assert!(scaffold_component(
            &invalid_path,
            "../escape".to_owned(),
            None,
            ScaffoldKind::Model,
            None,
            vec!["example".to_owned()],
            Vec::new(),
        )
        .is_err());
        assert!(!invalid_path.exists());
    }

    #[test]
    fn python_template_runs_through_the_davis_runtime() {
        let repository = tempfile::tempdir().unwrap();
        let component_path = repository.path().join("standalone-example-component");

        let scaffolded = scaffold_component(
            &component_path,
            "example/calculator".to_owned(),
            Some("Calculator".to_owned()),
            ScaffoldKind::Transform,
            Some(ScaffoldTemplate::Python),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let plan_path = scaffolded.example_plan.expect("example plan");

        assert!(component_path.join("component.py").is_file());
        assert!(component_path.join("README.md").is_file());
        let completed = davis_runtime::execute_plan(
            repository.path(),
            &plan_path,
            &repository.path().join("davis-runs"),
        )
        .unwrap();

        assert_eq!(
            completed.result.status,
            davis_model_api::RunStatus::Succeeded
        );
        assert!(completed.result.artifacts.contains_key("output_table"));
        assert!(completed.result.artifacts.contains_key("summary"));
        assert_eq!(
            completed.result.artifacts["summary"].profile,
            Some(ArtifactProfile::Metrics)
        );
        let summary = fs::read_to_string(
            completed
                .run_directory
                .join("artifacts")
                .join("summary.json"),
        )
        .unwrap();
        assert!(summary.contains("60.0"));
    }
}
