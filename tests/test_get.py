"""Regression tests for the ``davis get`` command."""

from datetime import UTC, datetime
from pathlib import Path

from dataset_cli.commands.get import _collect_targets
from dataset_cli.schemas.manifest import Manifest


def test_collect_targets_preserves_source_extensions_in_pdf_names() -> None:
    """PDF destinations must distinguish files that only differ by extension."""
    manifest = Manifest.model_validate(
        {
            "manifest_version": "1",
            "cli_version": "0.1.0",
            "generated_at": datetime.now(tz=UTC),
            "bootstrap_package_url": "https://example.com/dvc-bootstrap.zip",
            "bootstrap_package_hash": "hash",
            "datasets": {
                "example/sample": {
                    "name": {"ja": "サンプル", "en": "Sample"},
                    "dvc_files": ["data/example/sample.csv.dvc"],
                    "pdf_urls": {
                        "sample.csv": {
                            "ja": "https://example.com/sample.csv.ja.pdf",
                            "en": "https://example.com/sample.csv.en.pdf",
                        },
                        "sample.xlsx": {
                            "ja": "https://example.com/sample.xlsx.ja.pdf",
                            "en": "https://example.com/sample.xlsx.en.pdf",
                        },
                    },
                },
            },
        },
    )

    _, pdf_targets = _collect_targets("example/sample", manifest, Path("output"))

    assert [destination for _, destination in pdf_targets] == [
        Path("output/data/example/sample.csv.ja.pdf"),
        Path("output/data/example/sample.csv.en.pdf"),
        Path("output/data/example/sample.xlsx.ja.pdf"),
        Path("output/data/example/sample.xlsx.en.pdf"),
    ]
