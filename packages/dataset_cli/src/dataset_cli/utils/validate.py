import hashlib
from pathlib import Path

import charset_normalizer
import polars as pl
import typer
import yaml
from pydantic import ValidationError
from rich import print as rprint

from dataset_cli.schemas.dataset_config import DatasetConfig
from dataset_cli.schemas.polars import get_polars_data_type
from dataset_cli.utils.parser import infer_file_type, parse_yaml_and_validate


def validate_paths(file_path: str, schema_path: Path) -> None:
    """データファイルとスキーマファイルの存在を検証する。"""
    if not Path(file_path).exists():
        rprint(f"[red]ファイルが見つかりません: {file_path}[/red]")
        raise typer.Exit(code=1)

    if not schema_path.exists():
        rprint(f"[red]スキーマファイルが見つかりません: {schema_path}[/red]")
        raise typer.Exit(code=1)


def load_and_validate_schema(
    schema_path: Path,
    *,
    quiet: bool = False,
) -> DatasetConfig:
    """スキーマファイルを読み込み、その形式を検証する。"""
    try:
        schema = parse_yaml_and_validate(schema_path, DatasetConfig)
        if not quiet:
            rprint("[green]スキーマは有効です。[/green]")

    except ValidationError as e:
        rprint("[red]スキーマの形式が無効です。[/red]")
        rprint(e.json(indent=2))
        raise typer.Exit(code=1) from e

    return schema


def read_data_with_schema(
    file_path: str,
    schema_path: Path,
    encoding: str | None = None,
    *,
    write_hash: bool = True,
    quiet: bool = False,
) -> tuple[pl.DataFrame, DatasetConfig]:
    """ファイル種別に応じてデータを読み込み、スキーマを適用する。"""
    file_type = infer_file_type(file_path)

    schema_config = load_and_validate_schema(schema_path, quiet=quiet)

    # Polars用のスキーマ辞書を作成
    schema_dict = pl.Schema(
        {col.name: get_polars_data_type(col.type_) for col in schema_config.columns},
    )

    try:
        match file_type:
            case "text/csv":
                enc = encoding if encoding else detect_encoding(file_path)
                loaded_dataframe = pl.read_csv(
                    file_path,
                    schema=schema_dict,
                    try_parse_dates=True,
                    encoding=enc,
                )
            case (
                "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                loaded_dataframe = pl.read_excel(
                    file_path,
                    schema_overrides=schema_dict,
                )
            case "application/json":
                loaded_dataframe = pl.read_json(file_path, schema=schema_dict)
            case "application/x-parquet":
                loaded_dataframe = pl.read_parquet(file_path, schema=schema_dict)
            case _:
                msg = f"サポートされていないファイルタイプ: {file_type}"
                raise NotImplementedError(msg)

        # 読み込み後の列チェック
        if set(loaded_dataframe.columns) != set(schema_dict.keys()):
            rprint("[red]ファイルの列がスキーマと一致しません。[/red]")
            rprint(f"ファイル側: {sorted(loaded_dataframe.columns)}")
            rprint(f"スキーマ側: {sorted(schema_dict.keys())}")
            raise typer.Exit(code=1)

    except (pl.exceptions.ShapeError, pl.exceptions.SchemaError) as e:
        rprint("[red]ファイルはスキーマに適合していません。[/red]")
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    if write_hash:
        try:
            write_file_hash(file_path, schema_path)
        except ValueError as e:
            rprint(f"[red]ハッシュの書き込みに失敗しました: {e}[/red]")
            raise typer.Exit(code=1) from e

    return loaded_dataframe, schema_config


def generate_file_hash(file_or_dir_path: str | Path) -> str:
    """ファイルまたはディレクトリのSHA-256ハッシュを生成する。"""
    sha256 = hashlib.sha256()
    path = (
        file_or_dir_path
        if isinstance(file_or_dir_path, Path)
        else Path(file_or_dir_path)
    )

    chunk_size = 8192

    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256.update(chunk)
        return sha256.hexdigest().lower()

    if path.is_dir():
        include_extensions = {".csv", ".json", ".parquet", ".xlsx", ".xls"}
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() not in include_extensions:
                continue
            if file.is_file():
                relative_path = file.relative_to(path).as_posix().encode("utf-8")
                sha256.update(relative_path)
                # ファイル内容も追加
                with file.open("rb") as f:
                    for chunk in iter(lambda: f.read(chunk_size), b""):
                        sha256.update(chunk)
        return sha256.hexdigest().lower()

    msg = f"無効なパス: {file_or_dir_path}"
    raise ValueError(msg)


def write_file_hash(
    file_path: str | Path,
    schema_path: Path,
) -> None:
    """ファイルのハッシュをスキーマファイルに書き込む。"""
    file_hash = generate_file_hash(file_path)
    try:
        schema_config = load_and_validate_schema(schema_path, quiet=True)
        schema_config.hash_ = file_hash
        with schema_path.open("w", encoding="utf-8") as f:
            yaml.dump(schema_config.model_dump(), f, allow_unicode=True)
        rprint(
            f"[green]スキーマファイルにハッシュを保存しました[/green] {file_hash}",
        )
    except ValidationError as e:
        rprint("[red]スキーマの形式が無効です。[/red]")
        rprint(e.json(indent=2))
        raise typer.Exit(code=1) from e


def validate_file_hash(
    file_path: str | Path,
    expected_hash: str,
) -> bool:
    """ファイルのハッシュが期待値と一致するか検証する。"""
    actual_hash = generate_file_hash(file_path)
    if actual_hash != expected_hash:
        rprint(
            f"[red]ファイルのハッシュが一致しません。\n期待値: {expected_hash}\n実際の値: {actual_hash}[/red]",
        )
        return False
    return True


def detect_encoding(file_path: str, nbytes: int = 4096) -> str:
    """ファイルのエンコーディングを自動判定する。"""
    with Path(file_path).open("rb") as f:
        raw = f.read(nbytes)
    result = charset_normalizer.from_bytes(raw).best()
    if result is not None:
        return result.encoding
    return "utf-8"
