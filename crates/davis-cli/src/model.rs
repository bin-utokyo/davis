use std::path::{Path, PathBuf};

use davis_runtime::{execute_plan, inspect_csv, plan_run, validate_plan};

use crate::ModelCommand;

#[allow(clippy::too_many_lines)]
pub(crate) fn handle(
    repository: &Path,
    command: ModelCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ModelCommand::Inspect { path, json } => {
            let path = resolve(repository, &path);
            let profile = inspect_csv(&path)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&profile)?);
            } else {
                println!("CSV: {}", profile.path.display());
                println!("Encoding: {}", profile.encoding);
                println!("Delimiter: {:?}", profile.delimiter);
                println!("Rows sampled: {}", profile.rows_sampled);
                for column in profile.columns {
                    let warning = if column.warnings.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", column.warnings.join(", "))
                    };
                    println!(
                        "- {}: {}, nulls {}, unique sample {}{}",
                        column.name,
                        column.inferred_type,
                        column.null_count,
                        column.unique_sample,
                        warning
                    );
                }
            }
        }
        ModelCommand::Validate { plan, json } => {
            let plan = resolve(repository, &plan);
            let validated = validate_plan(repository, &plan)?;
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "valid": true,
                        "plan": validated.plan_path,
                        "component": {
                            "id": validated.manifest.id,
                            "version": validated.manifest.version,
                            "kind": validated.manifest.kind,
                            "manifest": validated.manifest_path,
                        }
                    })
                );
            } else {
                println!("Analysis plan is valid");
                println!("Plan: {}", validated.plan_path.display());
                println!(
                    "Component: {} {}",
                    validated.manifest.id, validated.manifest.version
                );
                println!("Kind: {:?}", validated.manifest.kind);
                println!("Manifest: {}", validated.manifest_path.display());
            }
        }
        ModelCommand::Plan {
            plan,
            run_root,
            json,
        } => {
            let plan = resolve(repository, &plan);
            let run_root = resolve(repository, &run_root);
            let planned = plan_run(repository, &plan, &run_root)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&planned)?);
            } else {
                println!("Run ID: {}", planned.request.run_id);
                println!(
                    "Component: {} {}",
                    planned.request.component.id, planned.request.component.version
                );
                println!("Operation: {}", planned.request.operation);
                for (name, input) in planned.request.inputs {
                    println!(
                        "Input {name}: {} ({}, {})",
                        input.resolved.path.display(),
                        input.resolved.media_type,
                        input.resolved.object_id
                    );
                }
                println!("Output: {}", planned.request.output_directory.display());
            }
        }
        ModelCommand::Run {
            plan,
            run_root,
            json,
        } => {
            let plan = resolve(repository, &plan);
            let run_root = resolve(repository, &run_root);
            let completed = execute_plan(repository, &plan, &run_root)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&completed)?);
            } else {
                println!("Run succeeded: {}", completed.request.run_id);
                println!("Run directory: {}", completed.run_directory.display());
                for (name, artifact) in completed.result.artifacts {
                    println!("Artifact {name}: {}", artifact.path.display());
                }
            }
        }
    }
    Ok(())
}

fn resolve(repository: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        repository.join(path)
    }
}
