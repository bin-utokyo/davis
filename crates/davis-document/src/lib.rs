//! Deterministic documentation generated from Davis schemas.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use davis_core::{FileSchema, ObjectRef};
use fullbleed::{Asset, AssetBundle, AssetKind, FullBleed};
use thiserror::Error;

const INTER: &[u8] = include_bytes!("../assets/fonts/Inter-Variable.ttf");
const BIZ_UD_GOTHIC_REGULAR: &[u8] = include_bytes!("../assets/fonts/BIZUDPGothic-Regular.ttf");
const BIZ_UD_GOTHIC_BOLD: &[u8] = include_bytes!("../assets/fonts/BIZUDPGothic-Bold.ttf");

static PDF_ENGINE: OnceLock<Result<FullBleed, String>> = OnceLock::new();

const README_CSS: &str = include_str!("../assets/templates/readme.css");

#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("failed to render PDF: {0}")]
    Render(String),
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Japanese,
    English,
}

impl Language {
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Japanese => ".ja.pdf",
            Self::English => ".en.pdf",
        }
    }
}

/// Generates a stable PDF byte stream from one schema and immutable object reference.
///
/// # Errors
///
/// Returns an error when the bundled fonts cannot be loaded or PDF rendering fails.
pub fn render_schema_pdf(
    schema: &FileSchema,
    object: &ObjectRef,
    language: Language,
) -> Result<Vec<u8>, DocumentError> {
    let html = schema_html(schema, object, language);
    let engine = PDF_ENGINE
        .get_or_init(build_pdf_engine)
        .as_ref()
        .map_err(|error| DocumentError::Render(error.clone()))?;
    engine
        .render_to_buffer(&html, README_CSS)
        .map_err(|error| DocumentError::Render(error.to_string()))
}

fn build_pdf_engine() -> Result<FullBleed, String> {
    let mut assets = AssetBundle::default();
    for (name, bytes) in [
        ("Inter", INTER),
        ("BIZ UDPGothic Regular", BIZ_UD_GOTHIC_REGULAR),
        ("BIZ UDPGothic Bold", BIZ_UD_GOTHIC_BOLD),
    ] {
        assets.add(Asset::new(
            name.to_owned(),
            AssetKind::Font,
            bytes.to_vec(),
            None,
            true,
        ));
    }
    FullBleed::builder()
        .register_bundle(assets)
        .build()
        .map_err(|error| error.to_string())
}

/// Writes a generated PDF only when its bytes changed.
///
/// # Errors
///
/// Returns an error when the destination cannot be read or written.
pub fn write_pdf_if_changed(path: &Path, contents: &[u8]) -> Result<bool, DocumentError> {
    if fs::read(path).is_ok_and(|current| current == contents) {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| DocumentError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    }
    fs::write(path, contents).map_err(|source| DocumentError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(true)
}

fn schema_html(schema: &FileSchema, object: &ObjectRef, language: Language) -> String {
    let (overview, license, columns, column_name, data_type, description, hash_label, footer) =
        match language {
            Language::Japanese => (
                "概要",
                "ライセンス",
                "データカラム仕様",
                "カラム名",
                "データ型",
                "説明",
                "ハッシュ値 (BLAKE3)",
                "最終更新: Git履歴を参照<br>&copy;東大交通研",
            ),
            Language::English => (
                "Overview",
                "License",
                "Data Column Specification",
                "Column Name",
                "Data Type",
                "Description",
                "Hash Value (BLAKE3)",
                "Last Updated: see Git history<br>&copy;BinN, UTokyo",
            ),
        };
    let title = escape_html(localized(&schema.name, language));
    let city = schema
        .city
        .as_ref()
        .map_or("", |value| localized(value, language));
    let year = schema.year.map_or_else(String::new, |year| match language {
        Language::Japanese => format!("{year}年"),
        Language::English => year.to_string(),
    });
    let summary = [year.as_str(), city]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" / ");
    let mut rows = String::new();
    for column in &schema.columns {
        let column_description = column
            .description
            .as_ref()
            .map_or("", |value| localized(value, language));
        write!(
            rows,
            "<tr><td class=\"column-name\">{}</td><td class=\"column-schema\"><code>{}</code></td><td class=\"column-description\">{}</td></tr>",
            escape_html(&column.name),
            escape_html(&column.data_type),
            escape_html_with_breaks(column_description),
        )
        .expect("writing HTML to a String cannot fail");
    }
    format!(
        r#"<!doctype html><html><head><meta charset="UTF-8"/><title>README: {title}</title></head><body><header><h1>{title}</h1><p class="subtitle">{}</p></header><section><h2>{overview}</h2><p>{}</p></section><section><h2>{license}</h2><p>{}</p></section><section><h2>{columns}</h2><table><thead><tr><th class="column-name">{column_name}</th><th class="column-schema">{data_type}</th><th class="column-description">{description}</th></tr></thead><tbody>{rows}</tbody></table></section><footer><h2>{hash_label}</h2><pre class="hash"><code>{}</code></pre></footer><div id="footer-info"><p>{footer}</p></div></body></html>"#,
        escape_html(&summary),
        escape_html(
            schema
                .description
                .as_ref()
                .map_or("", |value| localized(value, language))
        ),
        escape_html(
            schema
                .license
                .as_ref()
                .map_or("", |value| localized(value, language))
        ),
        escape_html(object.oid.digest()),
    )
}

fn localized(value: &davis_core::LocalizedText, language: Language) -> &str {
    match language {
        Language::Japanese => &value.ja,
        Language::English => &value.en,
    }
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn escape_html_with_breaks(value: &str) -> String {
    let collapsed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut output = String::new();
    let mut line_width = 0_usize;
    for character in collapsed.chars() {
        let width = usize::from(!character.is_ascii()) + 1;
        if line_width > 0 && line_width + width > 46 {
            output.push_str("<br>");
            line_width = 0;
            if character == ' ' {
                continue;
            }
        }
        output.push_str(&escape_html(&character.to_string()));
        line_width += width;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{render_schema_pdf, Language};
    use davis_core::{FileSchema, LocalizedText, ObjectRef};

    #[test]
    fn generated_pdf_has_a_pdf_header() {
        let schema = FileSchema {
            name: LocalizedText {
                ja: "テスト".into(),
                en: "Test".into(),
            },
            description: None,
            city: None,
            year: None,
            license: None,
            columns: Vec::new(),
        };
        let object = ObjectRef {
            oid: "blake3:aabbcc".parse().unwrap(),
            size: 12,
        };
        let pdf = render_schema_pdf(&schema, &object, Language::Japanese).unwrap();
        assert!(pdf.starts_with(b"%PDF-"));
        assert_eq!(
            pdf,
            render_schema_pdf(&schema, &object, Language::Japanese).unwrap()
        );
    }
}
