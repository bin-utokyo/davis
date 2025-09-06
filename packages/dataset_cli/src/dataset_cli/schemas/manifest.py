# ./src/dataset_cli/src/dataset_cli/schemas/manifest.py
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator

from dataset_cli.schemas.dataset_config import LocalizedStr


class PdfUrls(BaseModel):
    ja: HttpUrl
    en: HttpUrl


class DatasetInfo(BaseModel):
    name: LocalizedStr
    description: LocalizedStr | None = None
    year: int | None = None
    dvc_files: list[str] = Field(default_factory=list)
    pdf_urls: dict[str, PdfUrls] = Field(default_factory=dict)

    @field_validator("dvc_files")
    @classmethod
    def check_dvc_files(cls, v: list[str], info: Any) -> list[str]:  # noqa: ANN401
        # デフォルト [] の場合は許可
        if v == [] and info.data.get("dvc_files") is None:
            return v
        # それ以外で空ならエラー
        if len(v) == 0:
            msg = "dvc_files must contain at least one item after initialization"
            raise ValueError(
                msg,
            )
        return v


class Manifest(BaseModel):
    manifest_version: str
    cli_version: str
    generated_at: datetime
    bootstrap_package_url: HttpUrl
    bootstrap_package_hash: str
    datasets: dict[str, DatasetInfo]
