use std::path::PathBuf;

use davis_runtime::{ComponentStore, InstalledComponent};

use crate::{ComponentCommand, InstallCommand};

pub(crate) async fn handle_install(
    command: InstallCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        InstallCommand::Component {
            source,
            version,
            registry,
            json,
        } => {
            let store = ComponentStore::for_user()?;
            let path = PathBuf::from(&source);
            let installed = if path.exists() || looks_like_explicit_path(&source, &path) {
                if version.is_some() || registry.is_some() {
                    return Err(
                        "--version and --registry can only be used with an official component name"
                            .into(),
                    );
                }
                store.install(&path)?
            } else {
                let downloaded = crate::component_registry::download(
                    &source,
                    version.as_deref(),
                    registry.as_deref(),
                )
                .await?;
                let origin = format!("registry:{}@{}", downloaded.id(), downloaded.version());
                store.install_with_origin(downloaded.path(), Some(origin))?
            };
            print_installed(&installed, json, "Installed")?;
        }
    }
    Ok(())
}

pub(crate) fn handle_component(
    command: ComponentCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
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
