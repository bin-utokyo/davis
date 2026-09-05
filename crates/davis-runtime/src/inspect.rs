use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::io::{Read, Take};
use std::path::{Path, PathBuf};

use encoding_rs::SHIFT_JIS;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const INSPECTION_BYTE_LIMIT: u64 = 4 * 1024 * 1024;
const INSPECTION_ROW_LIMIT: usize = 10_000;

#[derive(Debug, Error)]
pub enum InspectError {
    #[error("failed to read CSV {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse CSV {path}: {source}")]
    Csv { path: PathBuf, source: csv::Error },
    #[error("CSV has no header row: {0}")]
    MissingHeader(PathBuf),
    #[error("CSV {path} does not contain column `{column}`")]
    MissingColumn { path: PathBuf, column: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvProfile {
    pub path: PathBuf,
    pub encoding: String,
    pub delimiter: String,
    pub rows_sampled: usize,
    pub truncated: bool,
    pub columns: Vec<ColumnProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    pub inferred_type: String,
    pub null_count: usize,
    pub unique_sample: usize,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistinctValues {
    pub path: PathBuf,
    pub column: String,
    pub values: Vec<String>,
    pub rows_sampled: usize,
    pub truncated: bool,
}

#[derive(Default)]
struct MutableProfile {
    null_count: usize,
    values: HashSet<String>,
    kind: Option<ValueKind>,
    leading_zero: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    Boolean,
    Integer,
    Float,
    String,
}

/// Inspects a bounded sample of a local CSV without changing the file.
///
/// # Errors
///
/// Returns an error when the file cannot be read or the sampled CSV records
/// cannot be parsed consistently.
pub fn inspect_csv(path: &Path) -> Result<CsvProfile, InspectError> {
    let mut bytes = Vec::new();
    let file = File::open(path).map_err(|source| InspectError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut limited: Take<File> = file.take(INSPECTION_BYTE_LIMIT);
    limited
        .read_to_end(&mut bytes)
        .map_err(|source| InspectError::Io {
            path: path.to_owned(),
            source,
        })?;
    let truncated =
        std::fs::metadata(path).is_ok_and(|metadata| metadata.len() > bytes.len() as u64);
    let (text, encoding) = decode(&bytes);
    let delimiter = detect_delimiter(&text);
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|source| InspectError::Csv {
            path: path.to_owned(),
            source,
        })?
        .clone();
    if headers.is_empty() {
        return Err(InspectError::MissingHeader(path.to_owned()));
    }
    let mut profiles: Vec<MutableProfile> = (0..headers.len())
        .map(|_| MutableProfile::default())
        .collect();
    let mut rows_sampled = 0;
    for record in reader.records().take(INSPECTION_ROW_LIMIT) {
        let record = record.map_err(|source| InspectError::Csv {
            path: path.to_owned(),
            source,
        })?;
        rows_sampled += 1;
        for (index, profile) in profiles.iter_mut().enumerate() {
            let value = record.get(index).unwrap_or("").trim();
            if value.is_empty() || matches!(value, "NA" | "N/A" | "null" | "NULL") {
                profile.null_count += 1;
                continue;
            }
            if profile.values.len() < 1_000 {
                profile.values.insert(value.to_owned());
            }
            profile.leading_zero |= has_significant_leading_zero(value);
            profile.kind = Some(merge_kind(profile.kind, infer_kind(value)));
        }
    }
    let columns = headers
        .iter()
        .zip(profiles)
        .map(|(name, profile)| ColumnProfile {
            name: name.to_owned(),
            inferred_type: kind_name(profile.kind).to_owned(),
            null_count: profile.null_count,
            unique_sample: profile.values.len(),
            warnings: if profile.leading_zero {
                vec!["leading-zero-values".to_owned()]
            } else {
                Vec::new()
            },
        })
        .collect();
    Ok(CsvProfile {
        path: path.to_owned(),
        encoding: encoding.to_owned(),
        delimiter: char::from(delimiter).to_string(),
        rows_sampled,
        truncated,
        columns,
    })
}

/// Returns sorted distinct values from the bounded CSV inspection sample.
///
/// # Errors
///
/// Returns an error when the file cannot be read, parsed, or does not contain
/// the requested column.
pub fn distinct_csv_values(
    path: &Path,
    column: &str,
    value_limit: usize,
) -> Result<DistinctValues, InspectError> {
    let mut bytes = Vec::new();
    let file = File::open(path).map_err(|source| InspectError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut limited: Take<File> = file.take(INSPECTION_BYTE_LIMIT);
    limited
        .read_to_end(&mut bytes)
        .map_err(|source| InspectError::Io {
            path: path.to_owned(),
            source,
        })?;
    let byte_truncated =
        std::fs::metadata(path).is_ok_and(|metadata| metadata.len() > bytes.len() as u64);
    let (text, _) = decode(&bytes);
    let delimiter = detect_delimiter(&text);
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|source| InspectError::Csv {
            path: path.to_owned(),
            source,
        })?
        .clone();
    let index = headers
        .iter()
        .position(|name| name == column)
        .ok_or_else(|| InspectError::MissingColumn {
            path: path.to_owned(),
            column: column.to_owned(),
        })?;
    let mut values = BTreeSet::new();
    let mut rows_sampled = 0;
    let mut value_truncated = false;
    for record in reader.records().take(INSPECTION_ROW_LIMIT) {
        let record = record.map_err(|source| InspectError::Csv {
            path: path.to_owned(),
            source,
        })?;
        rows_sampled += 1;
        let value = record.get(index).unwrap_or("").trim();
        if value.is_empty() {
            continue;
        }
        if values.len() < value_limit || values.contains(value) {
            values.insert(value.to_owned());
        } else {
            value_truncated = true;
        }
    }
    Ok(DistinctValues {
        path: path.to_owned(),
        column: column.to_owned(),
        values: values.into_iter().collect(),
        rows_sampled,
        truncated: byte_truncated || rows_sampled == INSPECTION_ROW_LIMIT || value_truncated,
    })
}

fn decode(bytes: &[u8]) -> (String, &'static str) {
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    if let Ok(text) = std::str::from_utf8(bytes) {
        return (text.to_owned(), "utf-8");
    }
    let (text, _, _) = SHIFT_JIS.decode(bytes);
    (text.into_owned(), "cp932")
}

fn detect_delimiter(text: &str) -> u8 {
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    [',', '\t', ';', '|']
        .into_iter()
        .max_by_key(|delimiter| line.matches(*delimiter).count())
        .unwrap_or(',') as u8
}

fn infer_kind(value: &str) -> ValueKind {
    if matches!(value.to_ascii_lowercase().as_str(), "true" | "false") {
        ValueKind::Boolean
    } else if value.parse::<i64>().is_ok() {
        ValueKind::Integer
    } else if value.parse::<f64>().is_ok() {
        ValueKind::Float
    } else {
        ValueKind::String
    }
}

fn merge_kind(current: Option<ValueKind>, next: ValueKind) -> ValueKind {
    match (current, next) {
        (None, next) => next,
        (Some(ValueKind::String), _)
        | (_, ValueKind::String)
        | (Some(ValueKind::Integer), ValueKind::Boolean)
        | (Some(ValueKind::Boolean), ValueKind::Integer) => ValueKind::String,
        (Some(ValueKind::Float), _) | (_, ValueKind::Float) => ValueKind::Float,
        (Some(current), _) => current,
    }
}

fn kind_name(kind: Option<ValueKind>) -> &'static str {
    match kind {
        Some(ValueKind::Boolean) => "boolean",
        Some(ValueKind::Integer) => "integer",
        Some(ValueKind::Float) => "float",
        Some(ValueKind::String) => "string",
        None => "unknown",
    }
}

fn has_significant_leading_zero(value: &str) -> bool {
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    unsigned.len() > 1
        && unsigned.starts_with('0')
        && unsigned.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{distinct_csv_values, inspect_csv};

    #[test]
    fn reports_leading_zero_identifiers() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sample.csv");
        fs::write(&path, "person_id,value\n001,1.5\n002,2.0\n").unwrap();
        let profile = inspect_csv(&path).unwrap();
        assert_eq!(profile.encoding, "utf-8");
        assert_eq!(profile.rows_sampled, 2);
        assert_eq!(profile.columns[0].warnings, ["leading-zero-values"]);
        assert_eq!(profile.columns[1].inferred_type, "float");
    }

    #[test]
    fn returns_bounded_distinct_column_values() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sample.csv");
        fs::write(
            &path,
            "case_id,alternative\n001,train\n001,car\n002,train\n002,walk\n",
        )
        .unwrap();
        let values = distinct_csv_values(&path, "alternative", 10).unwrap();
        assert_eq!(values.values, ["car", "train", "walk"]);
        assert_eq!(values.rows_sampled, 4);
        assert!(!values.truncated);
    }
}
