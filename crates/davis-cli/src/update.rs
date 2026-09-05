use std::fs;
use std::io::{BufRead, IsTerminal, Write};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::{StatusCode, Url};
use semver::Version;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::session;

const DEFAULT_UPDATE_URL: &str =
    "https://github.com/bin-utokyo/davis/releases/latest/download/latest-version.json";
const RELEASE_DOWNLOAD_ROOT: &str = "https://github.com/bin-utokyo/davis/releases/download";
const UPDATE_INTERVAL_SECONDS: u64 = 24 * 60 * 60;
const UPDATE_STATE_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
struct ReleaseInfo {
    schema_version: u32,
    latest: String,
    minimum_supported: String,
    released_at: String,
    release_url: String,
    message: LocalizedMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct LocalizedMessage {
    ja: String,
    en: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct UpdateState {
    version: u32,
    checked_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateStatus {
    Current,
    Available,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Japanese,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstallOutcome {
    #[cfg(not(target_os = "windows"))]
    Completed,
    #[cfg(target_os = "windows")]
    Started,
}

#[derive(Debug, Error)]
enum UpdateError {
    #[error("invalid update metadata URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("failed to retrieve update metadata: {0}")]
    Request(#[from] reqwest::Error),
    #[error("update metadata returned HTTP {0}")]
    Http(StatusCode),
    #[error("unsupported update information version: {0}")]
    UnsupportedSchema(u32),
    #[error("invalid release version: {0}")]
    InvalidVersion(String),
}

pub async fn check_automatically() {
    if std::env::var_os("DAVIS_DISABLE_UPDATE_CHECK").is_some() {
        return;
    }
    let now = unix_time();
    if !check_is_due(now) {
        return;
    }
    record_attempt(now);
    if let Ok(release) = fetch_release_info().await {
        print_release_status(&release, false);
    }
}

pub async fn check_explicitly(assume_yes: bool) -> Result<(), Box<dyn std::error::Error>> {
    let now = unix_time();
    let release = fetch_release_info().await?;
    record_attempt(now);
    print_release_status(&release, true);
    if classify(&release)? == UpdateStatus::Current {
        return Ok(());
    }
    let language = Language::detect();
    let approved = if assume_yes {
        true
    } else {
        if !std::io::stdin().is_terminal() {
            return Err("standard input is not a terminal; use `davis update --yes` to install without confirmation".into());
        }
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        prompt_to_install(
            &mut stdin.lock(),
            &mut stdout.lock(),
            language,
            &release.latest,
        )?
    };
    if !approved {
        match language {
            Language::Japanese => println!("更新を中止しました．"),
            Language::English => println!("Update cancelled."),
        }
        return Ok(());
    }
    install_release(&release, language)?;
    Ok(())
}

async fn fetch_release_info() -> Result<ReleaseInfo, UpdateError> {
    let endpoint = update_endpoint()?;
    let response = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(8))
        .build()?
        .get(endpoint)
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(UpdateError::Http(response.status()));
    }
    let release = response.json::<ReleaseInfo>().await?;
    if release.schema_version != 1 {
        return Err(UpdateError::UnsupportedSchema(release.schema_version));
    }
    classify(&release)?;
    Ok(release)
}

fn update_endpoint() -> Result<Url, UpdateError> {
    if let Ok(endpoint) = std::env::var("DAVIS_UPDATE_URL") {
        return Ok(Url::parse(&endpoint)?);
    }
    Ok(Url::parse(DEFAULT_UPDATE_URL)?)
}

fn classify(release: &ReleaseInfo) -> Result<UpdateStatus, UpdateError> {
    let current = parse_version(env!("CARGO_PKG_VERSION"))?;
    let latest = parse_version(&release.latest)?;
    let minimum = parse_version(&release.minimum_supported)?;
    if current < minimum {
        Ok(UpdateStatus::Required)
    } else if current < latest {
        Ok(UpdateStatus::Available)
    } else {
        Ok(UpdateStatus::Current)
    }
}

fn parse_version(value: &str) -> Result<Version, UpdateError> {
    Version::parse(value.trim_start_matches('v'))
        .map_err(|_| UpdateError::InvalidVersion(value.to_owned()))
}

fn print_release_status(release: &ReleaseInfo, explicit: bool) {
    let Ok(status) = classify(release) else {
        return;
    };
    let language = Language::detect();
    if status == UpdateStatus::Current {
        if explicit {
            match language {
                Language::Japanese => {
                    println!("Davis v{}は最新版です．", env!("CARGO_PKG_VERSION"));
                }
                Language::English => {
                    println!("Davis v{} is up to date.", env!("CARGO_PKG_VERSION"));
                }
            }
        }
        return;
    }

    let message = match language {
        Language::Japanese => &release.message.ja,
        Language::English => &release.message.en,
    };
    match (language, status) {
        (Language::Japanese, UpdateStatus::Required) => eprintln!(
            "\n警告: Davis v{}は現在のDavis releaseの最低対応versionを下回っています．\n最新版: v{} ({})\n{}",
            env!("CARGO_PKG_VERSION"),
            release.latest,
            release.released_at,
            message
        ),
        (Language::Japanese, UpdateStatus::Available) => eprintln!(
            "\nDavis v{}が利用できます (現在: v{})．\n{}",
            release.latest,
            env!("CARGO_PKG_VERSION"),
            message
        ),
        (Language::English, UpdateStatus::Required) => eprintln!(
            "\nWarning: Davis v{} is older than the minimum version supported by the current Davis release.\nLatest: v{} ({})\n{}",
            env!("CARGO_PKG_VERSION"),
            release.latest,
            release.released_at,
            message
        ),
        (Language::English, UpdateStatus::Available) => eprintln!(
            "\nDavis v{} is available (current: v{}).\n{}",
            release.latest,
            env!("CARGO_PKG_VERSION"),
            message
        ),
        (_, UpdateStatus::Current) => return,
    }
    if explicit {
        eprintln!("{}", release.release_url);
    } else {
        match language {
            Language::Japanese => {
                eprintln!("更新方法を確認するには`davis update`を実行してください．");
            }
            Language::English => eprintln!("Run `davis update` to see the update command."),
        }
    }
}

fn prompt_to_install(
    input: &mut impl BufRead,
    output: &mut impl Write,
    language: Language,
    version: &str,
) -> std::io::Result<bool> {
    loop {
        match language {
            Language::Japanese => write!(output, "Davis v{version}へ更新しますか? [y/N] ")?,
            Language::English => write!(output, "Update to Davis v{version}? [y/N] ")?,
        }
        output.flush()?;
        let mut answer = String::new();
        if input.read_line(&mut answer)? == 0 {
            return Ok(false);
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "" | "n" | "no" => return Ok(false),
            _ => match language {
                Language::Japanese => writeln!(output, "yまたはnを入力してください．")?,
                Language::English => writeln!(output, "Please answer y or n.")?,
            },
        }
    }
}

fn install_release(
    release: &ReleaseInfo,
    language: Language,
) -> Result<(), Box<dyn std::error::Error>> {
    let tag = format!("v{}", parse_version(&release.latest)?);
    let install_directory = std::env::current_exe()?
        .parent()
        .ok_or("current Davis executable has no parent directory")?
        .to_path_buf();
    match (
        language,
        install_platform_release(&tag, &install_directory)?,
    ) {
        #[cfg(not(target_os = "windows"))]
        (Language::Japanese, InstallOutcome::Completed) => {
            println!("Davis {tag}への更新が完了しました．");
        }
        #[cfg(not(target_os = "windows"))]
        (Language::English, InstallOutcome::Completed) => {
            println!("Davis was updated to {tag}.");
        }
        #[cfg(target_os = "windows")]
        (Language::Japanese, InstallOutcome::Started) => {
            println!("更新を開始しました．このcommandの終了後にDavis {tag}へ置き換えます．");
        }
        #[cfg(target_os = "windows")]
        (Language::English, InstallOutcome::Started) => {
            println!("Update started. Davis will be replaced with {tag} after this command exits.");
        }
    }
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn install_platform_release(
    tag: &str,
    install_directory: &std::path::Path,
) -> Result<InstallOutcome, Box<dyn std::error::Error>> {
    let installer_url = format!("{RELEASE_DOWNLOAD_ROOT}/{tag}/install.sh");
    let status = Command::new("sh")
        .arg("-c")
        .arg("curl --proto '=https' --tlsv1.2 -fsSL \"$DAVIS_INSTALLER_URL\" | sh")
        .env("DAVIS_INSTALLER_URL", installer_url)
        .env("DAVIS_VERSION", tag)
        .env("DAVIS_INSTALL_DIR", install_directory)
        .status()?;
    if !status.success() {
        return Err(format!("Davis installer exited with {status}").into());
    }
    Ok(InstallOutcome::Completed)
}

#[cfg(target_os = "windows")]
fn install_platform_release(
    tag: &str,
    install_directory: &std::path::Path,
) -> Result<InstallOutcome, Box<dyn std::error::Error>> {
    let installer_url = format!("{RELEASE_DOWNLOAD_ROOT}/{tag}/install.ps1");
    let command = "$parentProcess = Get-Process -Id ([int]$env:DAVIS_PARENT_PID) -ErrorAction SilentlyContinue; if ($parentProcess) { $parentProcess.WaitForExit() }; irm $env:DAVIS_INSTALLER_URL | iex";
    let child = Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ])
        .env("DAVIS_INSTALLER_URL", installer_url)
        .env("DAVIS_VERSION", tag)
        .env("DAVIS_INSTALL_DIR", install_directory)
        .env("DAVIS_PARENT_PID", std::process::id().to_string())
        .spawn()?;
    drop(child);
    Ok(InstallOutcome::Started)
}

impl Language {
    fn detect() -> Self {
        if let Ok(value) = std::env::var("DAVIS_LANGUAGE") {
            return Self::from_locale(&value);
        }
        ["LC_ALL", "LC_MESSAGES", "LANG"]
            .into_iter()
            .find_map(|name| std::env::var(name).ok())
            .map_or(Self::English, |value| Self::from_locale(&value))
    }

    fn from_locale(value: &str) -> Self {
        if value.to_ascii_lowercase().starts_with("ja") {
            Self::Japanese
        } else {
            Self::English
        }
    }
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn check_is_due(now: u64) -> bool {
    let Ok(path) = session::config_path("update-check.toml") else {
        return true;
    };
    let Ok(contents) = fs::read_to_string(path) else {
        return true;
    };
    let Ok(state) = toml::from_str::<UpdateState>(&contents) else {
        return true;
    };
    state.version != UPDATE_STATE_VERSION
        || now.saturating_sub(state.checked_at) >= UPDATE_INTERVAL_SECONDS
}

fn record_attempt(checked_at: u64) {
    let Ok(path) = session::config_path("update-check.toml") else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    let state = UpdateState {
        version: UPDATE_STATE_VERSION,
        checked_at,
    };
    let Ok(contents) = toml::to_string(&state) else {
        return;
    };
    if fs::create_dir_all(parent).is_ok() {
        let _ = fs::write(path, contents);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classify, prompt_to_install, Language, LocalizedMessage, ReleaseInfo, UpdateStatus,
    };
    use std::io::Cursor;

    fn release(latest: &str, minimum_supported: &str) -> ReleaseInfo {
        ReleaseInfo {
            schema_version: 1,
            latest: latest.into(),
            minimum_supported: minimum_supported.into(),
            released_at: "2026-08-21".into(),
            release_url: "https://example.test/release".into(),
            message: LocalizedMessage {
                ja: "更新があります．".into(),
                en: "An update is available.".into(),
            },
        }
    }

    #[test]
    fn classifies_current_available_and_required_versions() {
        assert_eq!(
            classify(&release(env!("CARGO_PKG_VERSION"), "0.1.0")).unwrap(),
            UpdateStatus::Current
        );
        assert_eq!(
            classify(&release("99.0.0", "0.1.0")).unwrap(),
            UpdateStatus::Available
        );
        assert_eq!(
            classify(&release("99.0.0", "99.0.0")).unwrap(),
            UpdateStatus::Required
        );
    }

    #[test]
    fn bundled_release_metadata_requires_the_current_cli_contract() {
        let release: ReleaseInfo =
            serde_json::from_str(include_str!("../../../release/latest-version.json"))
                .expect("release metadata should be valid");

        assert_eq!(release.latest, env!("CARGO_PKG_VERSION"));
        assert_eq!(release.minimum_supported, env!("CARGO_PKG_VERSION"));
        assert_eq!(classify(&release).unwrap(), UpdateStatus::Current);
    }

    #[test]
    fn detects_japanese_locale_explicitly() {
        assert_eq!(Language::from_locale("ja_JP.UTF-8"), Language::Japanese);
        assert_eq!(Language::from_locale("en_US.UTF-8"), Language::English);
    }

    #[test]
    fn update_prompt_accepts_yes() {
        let mut input = Cursor::new(b"y\n");
        let mut output = Vec::new();

        assert!(
            prompt_to_install(&mut input, &mut output, Language::Japanese, "1.2.3")
                .expect("prompt should accept input")
        );
        assert_eq!(
            String::from_utf8(output).expect("prompt should be UTF-8"),
            "Davis v1.2.3へ更新しますか? [y/N] "
        );
    }

    #[test]
    fn update_prompt_defaults_to_no() {
        let mut input = Cursor::new(b"\n");
        let mut output = Vec::new();

        assert!(
            !prompt_to_install(&mut input, &mut output, Language::English, "1.2.3")
                .expect("prompt should accept input")
        );
    }
}
