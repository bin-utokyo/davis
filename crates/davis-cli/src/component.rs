use davis_runtime::{ComponentStore, InstalledComponent};

use crate::{ComponentCommand, InstallCommand};

pub(crate) fn handle_install(command: InstallCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        InstallCommand::Component { path, json } => {
            let installed = ComponentStore::for_user()?.install(&path)?;
            print_installed(&installed, json, "Installed")?;
        }
    }
    Ok(())
}

pub(crate) fn handle_component(
    command: ComponentCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = ComponentStore::for_user()?;
    match command {
        ComponentCommand::List { json } => {
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
            let installed = store.inspect(&id, version.as_deref())?;
            print_installed(&installed, json, "Component")?;
        }
        ComponentCommand::Remove { id, version, json } => {
            let removed = store.remove(&id, version.as_deref())?;
            print_installed(&removed, json, "Removed")?;
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
        println!("Source: {}", component.source.display());
        println!("Digest: {}", component.source_digest);
    }
    Ok(())
}
