use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use thiserror::Error;

const SESSION_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    pub service_url: String,
    pub token: String,
    pub expires_at: String,
}

impl Session {
    #[must_use]
    pub fn new(service_url: String, token: String, expires_at: String) -> Self {
        Self {
            version: SESSION_VERSION,
            service_url,
            token,
            expires_at,
        }
    }
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Davis session directory could not be determined")]
    DirectoryUnavailable,
    #[error("failed to read session {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid session configuration in {path}: {source}")]
    Invalid {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("unsupported session version: {0}")]
    UnsupportedVersion(u32),
    #[error("failed to write session {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to serialize session: {0}")]
    Serialize(#[from] toml::ser::Error),
}

pub fn load() -> Result<Option<Session>, SessionError> {
    load_from(session_path("session.toml")?)
}

pub fn load_operator() -> Result<Option<Session>, SessionError> {
    load_from(session_path("operator-session.toml")?)
}

fn load_from(path: PathBuf) -> Result<Option<Session>, SessionError> {
    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(SessionError::Read { path, source }),
    };
    let session: Session = toml::from_str(&contents).map_err(|source| SessionError::Invalid {
        path: path.clone(),
        source,
    })?;
    if session.version != SESSION_VERSION {
        return Err(SessionError::UnsupportedVersion(session.version));
    }
    Ok(Some(session))
}

pub fn save(session: &Session) -> Result<PathBuf, SessionError> {
    save_to(session, session_path("session.toml")?)
}

pub fn save_operator(session: &Session) -> Result<PathBuf, SessionError> {
    save_to(session, session_path("operator-session.toml")?)
}

fn save_to(session: &Session, path: PathBuf) -> Result<PathBuf, SessionError> {
    let parent = path.parent().ok_or(SessionError::DirectoryUnavailable)?;
    fs::create_dir_all(parent).map_err(|source| SessionError::Write {
        path: parent.to_path_buf(),
        source,
    })?;
    let contents = toml::to_string(session)?;
    let mut temporary = NamedTempFile::new_in(parent).map_err(|source| SessionError::Write {
        path: path.clone(),
        source,
    })?;
    restrict_permissions(temporary.path())?;
    temporary
        .write_all(contents.as_bytes())
        .map_err(|source| SessionError::Write {
            path: path.clone(),
            source,
        })?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| SessionError::Write {
            path: path.clone(),
            source,
        })?;
    temporary
        .persist(&path)
        .map_err(|error| SessionError::Write {
            path: path.clone(),
            source: error.error,
        })?;
    restrict_permissions(&path)?;
    Ok(path)
}

pub fn clear() -> Result<bool, SessionError> {
    clear_path(session_path("session.toml")?)
}

pub fn clear_operator() -> Result<bool, SessionError> {
    clear_path(session_path("operator-session.toml")?)
}

fn clear_path(path: PathBuf) -> Result<bool, SessionError> {
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(SessionError::Write { path, source }),
    }
}

fn session_path(filename: &str) -> Result<PathBuf, SessionError> {
    if let Some(directory) = std::env::var_os("DAVIS_CONFIG_HOME") {
        return Ok(PathBuf::from(directory).join(filename));
    }
    #[cfg(target_os = "windows")]
    if let Some(directory) = std::env::var_os("APPDATA") {
        return Ok(PathBuf::from(directory).join("davis").join(filename));
    }
    #[cfg(target_os = "macos")]
    if let Some(directory) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(directory)
            .join("Library/Application Support/davis")
            .join(filename));
    }
    if let Some(directory) = std::env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(directory).join("davis").join(filename));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|directory| directory.join(".config/davis").join(filename))
        .ok_or(SessionError::DirectoryUnavailable)
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) -> Result<(), SessionError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|source| {
        SessionError::Write {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> Result<(), SessionError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{clear, clear_operator, load, load_operator, save, save_operator, Session};
    use std::sync::Mutex;

    static ENVIRONMENT: Mutex<()> = Mutex::new(());

    #[test]
    fn session_round_trip_and_clear() {
        let _guard = ENVIRONMENT.lock().unwrap();
        let directory = tempfile::tempdir().unwrap();
        std::env::set_var("DAVIS_CONFIG_HOME", directory.path());
        let expected = Session::new(
            "https://example.test".into(),
            "secret-token".into(),
            "2026-09-01T00:00:00.000Z".into(),
        );
        let path = save(&expected).unwrap();
        assert_eq!(load().unwrap(), Some(expected));
        assert!(clear().unwrap());
        assert!(!path.exists());
        let operator = Session::new(
            "https://operator.example.test".into(),
            "operator-token".into(),
            "2026-09-01T08:00:00.000Z".into(),
        );
        let operator_path = save_operator(&operator).unwrap();
        assert_eq!(load_operator().unwrap(), Some(operator));
        assert!(clear_operator().unwrap());
        assert!(!operator_path.exists());
        std::env::remove_var("DAVIS_CONFIG_HOME");
    }
}
