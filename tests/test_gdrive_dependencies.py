"""Regression tests for the DVC Google Drive dependency stack."""


def test_gdrive_backend_imports_with_resolved_dependencies() -> None:
    """The backend must not fail while resolving OpenSSL constants."""
    import OpenSSL.crypto  # noqa: F401, PLC0415
    from dvc_gdrive import GDriveFileSystem  # noqa: PLC0415

    assert GDriveFileSystem.protocol == "gdrive"
