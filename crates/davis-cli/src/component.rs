use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use davis_model_api::{
    ComponentKind, ComponentManifest, ConfigurationDeclaration, OutputDeclaration,
    PresentationDeclaration, RuntimeDeclaration, RuntimeExecutor, COMPONENT_API_VERSION,
    COMPONENT_MANIFEST_FILENAME,
};
use davis_runtime::{
    validate_component_package, ComponentStore, InstalledComponent, ValidatedComponentPackage,
};
use serde::Serialize;
use serde_json::json;

use crate::{ComponentCommand, ScaffoldKind};

#[derive(Debug, Serialize)]
struct ScaffoldedComponent {
    path: PathBuf,
    manifest_path: PathBuf,
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
    Ok(())
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
            runtime_command,
            operations,
            json,
        } => {
            let scaffolded =
                scaffold_component(&path, id, name, kind, runtime_command, operations)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&scaffolded)?);
            } else {
                println!("Created: {}", scaffolded.path.display());
                println!("Manifest: {}", scaffolded.manifest_path.display());
                println!(
                    "Next: add the program, then run `davis component validate {}`",
                    scaffolded.path.display()
                );
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
    runtime_command: Vec<String>,
    operations: Vec<String>,
) -> Result<ScaffoldedComponent, Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!("scaffold destination already exists: {}", path.display()).into());
    }
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
    let manifest = ComponentManifest {
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
            ui: Some(json!({ "ui:editor": "generic" })),
            ui_ref: None,
        }),
        config_schema: None,
        ui_schema: None,
        outputs: OutputDeclaration {
            standard: Vec::new(),
            extensions: Vec::new(),
            artifacts: BTreeMap::new(),
        },
    };
    manifest.validate()?;

    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path)?;
    let manifest_path = path.join(COMPONENT_MANIFEST_FILENAME);
    let write_result = serde_yaml::to_string(&manifest)
        .map_err(Into::into)
        .and_then(|yaml| fs::write(&manifest_path, yaml).map_err(Into::into));
    if let Err(error) = write_result {
        let _ = fs::remove_dir(path);
        return Err(error);
    }
    let validated = match validate_component_package(path) {
        Ok(validated) => validated,
        Err(error) => {
            let _ = fs::remove_file(&manifest_path);
            let _ = fs::remove_dir(path);
            return Err(error.into());
        }
    };
    Ok(ScaffoldedComponent {
        path: validated.source,
        manifest_path: validated.manifest_path,
        id: validated.manifest.id,
        version: validated.manifest.version,
    })
}

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
    fn scaffolds_a_valid_self_contained_component_without_overwriting() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("example-component");

        let scaffolded = scaffold_component(
            &path,
            "example/calculator".to_owned(),
            None,
            ScaffoldKind::Transform,
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
            vec!["example".to_owned()],
            Vec::new(),
        )
        .is_err());
        assert!(!invalid_path.exists());
    }
}
