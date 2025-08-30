# ./src/dataset_cli/src/dataset_cli/utils/io.py

import datetime
import hashlib
import mimetypes
from pathlib import Path

import charset_normalizer
import yaml
from pydantic import BaseModel, HttpUrl, ValidationError
from rich import print as rprint
from rich.progress import track

from dataset_cli.schemas import DatasetConfig, Manifest
from dataset_cli.schemas.manifest import DatasetInfo, PdfUrls


def parse_yaml_and_validate[T: BaseModel](
    yaml_path: Path,
    pydantic_model_class: type[T],
) -> T:
    """
    指定されたYAMLファイルをパースし、指定されたPydanticモデルで検証します。
    """
    if not yaml_path.exists():
        msg = f"YAMLファイルが見つかりません: {yaml_path}"
        raise FileNotFoundError(msg)
    if not yaml_path.is_file() or yaml_path.suffix.lower() not in (".yaml", ".yml"):
        msg = f"指定されたパスはYAMLファイルではありません: {yaml_path}"
        raise ValueError(msg)

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return pydantic_model_class.model_validate(data)
    except yaml.YAMLError as e:
        msg = f"YAMLの読み込みに失敗しました: {yaml_path}"
        raise RuntimeError(msg) from e
    except ValidationError as e:
        rprint(f"[bold red]スキーマ検証エラー in {yaml_path}:[/bold red]")
        rprint(e)
        raise  # TRY201: Use raise without specifying exception name


def generate_file_hash(file_path: Path) -> str:
    """
    ファイルのSHA-256ハッシュを生成します。
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def infer_file_type(file_path: str | Path) -> str | None:
    """
    指定されたファイルのタイプを推測します。

    Args
    ----
    file_path: str | Path
        ファイルのパス。

    Returns
    -------
    str: 推測されたファイルタイプ (例: "csv", "json", "yaml").
    """
    file_type, _ = mimetypes.guess_type(file_path)
    return file_type


def detect_encoding(file_path: str, nbytes: int = 4096) -> str:
    """
    ファイルのエンコーディングを自動判定する。
    """
    with Path(file_path).open("rb") as f:
        raw = f.read(nbytes)
    result = charset_normalizer.from_bytes(raw).best()
    if result is not None:
        return result.encoding
    return "utf-8"


def generate_manifest_data(
    cli_version: str,
    bootstrap_url: str,
    repo_url: str,
    branch: str,
) -> Manifest:
    """
    リポジリ内の全データセット情報をスキャンし、Manifestオブジェクトを生成します。
    """
    rprint("🔎 [bold]データファイルをスキャンしています...[/bold]")
    data_root = Path("data")
    if not data_root.is_dir():
        msg = "'data' ディレクトリが見つかりません。リポジトリのルートで実行してください。"
        raise FileNotFoundError(msg)

    dvc_files = sorted(data_root.rglob("*.dvc"))
    rprint(f"  - {len(dvc_files)} 個の .dvc ファイルを発見しました。")

    datasets: dict[str, DatasetInfo] = {}

    for dvc_file in track(dvc_files, description="スキーマとURLを処理中..."):
        original_file = dvc_file.with_suffix("")
        schema_file = dvc_file.with_suffix(".schema.yaml")

        if not schema_file.exists():
            rprint(
                f"[yellow]警告: スキーマファイルが見つかりません: {schema_file}。スキップします。[/yellow]",
            )
            continue

        try:
            schema_config = parse_yaml_and_validate(schema_file, DatasetConfig)
        except (FileNotFoundError, ValueError, RuntimeError, ValidationError):
            continue

        # データセットIDは 'data/' を除いたディレクトリパス
        dataset_id = original_file.parent.relative_to(data_root).as_posix()

        if dataset_id not in datasets:
            datasets[dataset_id] = DatasetInfo(
                name=schema_config.name,
                description=schema_config.description,
                year=schema_config.year,
                dvc_files=[],
            )

        datasets[dataset_id].dvc_files.append(dvc_file.as_posix())

        # PDFへのURLを構築
        pdf_ja_path = original_file.with_suffix(original_file.suffix + ".ja.pdf")
        pdf_en_path = original_file.with_suffix(original_file.suffix + ".en.pdf")
        if pdf_ja_path.exists() and pdf_en_path.exists():
            pdf_base_url = f"{repo_url}/blob/{branch}"
            pdf_url_ja, pdf_url_en = (
                HttpUrl(f"{pdf_base_url}/{pdf_ja_path.as_posix()}"),
                HttpUrl(f"{pdf_base_url}/{pdf_en_path.as_posix()}"),
            )
            datasets[dataset_id].pdf_urls[original_file.name] = PdfUrls(
                ja=pdf_url_ja,
                en=pdf_url_en,
            )

    return Manifest(
        manifest_version="1.1",
        cli_version=cli_version,
        generated_at=datetime.datetime.now(datetime.UTC),
        bootstrap_package_url=HttpUrl(bootstrap_url),
        datasets=datasets,
    )
