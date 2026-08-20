use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::{StatusCode, Url};
use semver::Version;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::session;

const DEFAULT_SERVICE_URL: &str = "https://davis-web.davis-bin.workers.dev";
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

#[derive(Debug, Error)]
enum UpdateError {
    #[error("invalid update service URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("failed to contact the update service: {0}")]
    Request(#[from] reqwest::Error),
    #[error("update service returned HTTP {0}")]
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

pub async fn check_explicitly() -> Result<(), Box<dyn std::error::Error>> {
    let now = unix_time();
    let release = fetch_release_info().await?;
    record_attempt(now);
    print_release_status(&release, true);
    Ok(())
}

async fn fetch_release_info() -> Result<ReleaseInfo, UpdateError> {
    let endpoint = update_endpoint()?;
    let response = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(2))
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
    let service_url = session::load()
        .ok()
        .flatten()
        .or_else(|| session::load_operator().ok().flatten())
        .map_or_else(
            || DEFAULT_SERVICE_URL.to_owned(),
            |stored| stored.service_url,
        );
    Ok(Url::parse(&service_url)?.join("/api/v1/version")?)
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
            "\n警告: Davis v{}は現在のserviceの最低対応versionを下回っています．\n最新版: v{} ({})\n{}",
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
            "\nWarning: Davis v{} is older than the minimum version supported by this service.\nLatest: v{} ({})\n{}",
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
        print_install_command(language);
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

fn print_install_command(language: Language) {
    #[cfg(target_os = "windows")]
    let command =
        "irm https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.ps1 | iex";
    #[cfg(not(target_os = "windows"))]
    let command = "curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.sh | sh";

    match language {
        Language::Japanese => eprintln!("次のcommandを実行して更新してください:\n  {command}"),
        Language::English => eprintln!("Run the following command to update:\n  {command}"),
    }
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
    use super::{classify, Language, LocalizedMessage, ReleaseInfo, UpdateStatus};

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
            classify(&release("0.2.0", "0.1.0")).unwrap(),
            UpdateStatus::Available
        );
        assert_eq!(
            classify(&release("0.2.0", "0.2.0")).unwrap(),
            UpdateStatus::Required
        );
    }

    #[test]
    fn detects_japanese_locale_explicitly() {
        assert_eq!(Language::from_locale("ja_JP.UTF-8"), Language::Japanese);
        assert_eq!(Language::from_locale("en_US.UTF-8"), Language::English);
    }
}
