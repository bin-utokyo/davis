import datetime
import itertools as it
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml
from pydantic import ValidationError
from rich import print as rprint
from rich.progress import track

from dataset_cli.schemas.dataset_config import (
    ColumnConfig,
    DatasetConfig,
    LocalizedStr,
)
from dataset_cli.schemas.polars import get_polars_data_type_name
from dataset_cli.utils.i18n import _
from dataset_cli.utils.parser import infer_file_type, parse_yaml_and_validate
from dataset_cli.utils.pdf import create_readme_pdf
from dataset_cli.utils.validate import (
    detect_encoding,
    read_data_with_schema,
    validate_paths,
    write_file_hash,
)

app = typer.Typer()

JST_YEAR = datetime.datetime.now(
    tz=datetime.timezone(datetime.timedelta(hours=9)),
).year


@app.command()
def validate_schema(
    config_path: Annotated[
        str,
        typer.Argument(help=_("検証するデータセット設定ファイルのパス")),
    ],
) -> DatasetConfig:
    """データセット設定ファイルを検証する。"""
    try:
        config = parse_yaml_and_validate(Path(config_path), DatasetConfig)
        rprint(_("[green]データセット設定は有効です。[/green]"))
    except ValidationError as e:
        rprint(_("[red]データセット設定の形式が無効です。[/red]"))
        rprint(e.json(indent=2))
        raise typer.Exit(code=1) from e

    return config


@app.command()
def validate(
    file_or_dir_path: Annotated[
        str,
        typer.Argument(help=_("検証するファイルまたはディレクトリのパス")),
    ],
    schema_path_str: Annotated[
        str | None,
        typer.Option("--schema", "-s", help=_("検証に使用するスキーマファイルのパス")),
    ] = None,
    encoding: Annotated[
        str | None,
        typer.Option(help=_("ファイルエンコーディング（例: utf-8, cp932 など）")),
    ] = None,
) -> None:
    """
    指定されたファイルまたはディレクトリを検証する。

    ファイルの場合はスキーマに適合するかチェックし、
    ディレクトリの場合は全てのファイルを検証する。
    """
    path_obj = Path(file_or_dir_path)
    if not path_obj.exists():
        rprint(
            _("[red]指定されたパスが存在しません: {file_or_dir_path}[/red]").format(
                file_or_dir_path=file_or_dir_path,
            ),
        )
        raise typer.Exit(code=1)

    if path_obj.is_file():
        _validate_file(file_or_dir_path, schema_path_str, encoding)
    elif path_obj.is_dir():
        _validate_directory(file_or_dir_path, schema_path_str, encoding)
    else:
        rprint(
            _("[red]不明なパスのタイプです: {file_or_dir_path}[/red]").format(
                file_or_dir_path=file_or_dir_path,
            ),
        )
        raise typer.Exit(code=1)


def _validate_file(
    file_path: str,
    schema_path_str: str | None,
    encoding: str | None,
) -> None:
    schema_path = (
        Path(schema_path_str) if schema_path_str else Path(file_path + ".schema.yaml")
    )
    validate_paths(file_path, schema_path)
    read_data_with_schema(file_path, schema_path, encoding=encoding)
    rprint(_("[green]ファイルはスキーマに適合しています。[/green]"))


def _validate_directory(
    dir_path: str,
    schema_path_str: str | None,
    encoding: str | None,
) -> None:
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir():
        rprint(
            _("[red]無効なディレクトリ: {dir_path}[/red]").format(
                dir_path=dir_path,
            ),
        )
        raise typer.Exit(code=1)

    file_paths = list(
        it.chain.from_iterable(
            dir_path_obj.glob(ext)
            for ext in ("*.csv", "*.xls", "*.xlsx", "*.json", "*.parquet")
        ),
    )

    if not file_paths:
        rprint(_("[yellow]検証対象のファイルが見つかりませんでした。[/yellow]"))
        return

    schema_path = (
        Path(schema_path_str) if schema_path_str else dir_path_obj / "schema.yaml"
    )
    if not schema_path.exists():
        rprint(
            _("[red]スキーマファイルが見つかりません: {schema_path}[/red]").format(
                schema_path=schema_path,
            ),
        )
        raise typer.Exit(code=1)

    write_file_hash(dir_path, schema_path)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                read_data_with_schema,
                f.as_posix(),
                schema_path,
                encoding,
                write_hash=False,
                quiet=True,
            ): f
            for f in file_paths
        }
        for future in track(
            as_completed(futures),
            total=len(futures),
            description="検証中",
        ):
            try:
                future.result()
            except Exception as e:  # noqa: BLE001
                rprint(_("[red]エラー: {e}[/red]").format(e=e))


@app.command()
def infer_schema(
    file_path: Annotated[str, typer.Argument(help="スキーマを推測するファイルのパス")],
    encoding: Annotated[
        str | None,
        typer.Option(help=_("ファイルエンコーディング（例: utf-8, cp932 など）")),
    ] = None,
) -> None:
    """指定されたファイルからスキーマを推測する。"""
    path = Path(file_path)
    if not path.exists():
        rprint(
            _("[red]ファイルが見つかりません: {file_path}[/red]").format(
                file_path=file_path,
            ),
        )
        raise typer.Exit(code=1)

    file_type = infer_file_type(file_path)
    if not file_type:
        rprint(_("[red]ファイルタイプを推測できませんでした。[/red]"))
        raise typer.Exit(code=1)

    enc = encoding if encoding else detect_encoding(file_path)

    try:
        match file_type:
            case "text/csv":
                input_dataframe = pl.read_csv(
                    file_path,
                    try_parse_dates=True,
                    encoding=enc,
                )
            case (
                "application/vnd.ms-excel"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                input_dataframe = pl.read_excel(file_path)
            case "application/json":
                input_dataframe = pl.read_json(file_path)
            case "application/x-parquet":
                input_dataframe = pl.read_parquet(file_path)
            case _:
                msg = _("サポートされていないファイルタイプ: {file_type}").format(
                    file_type=file_type,
                )
                raise NotImplementedError(msg)  # noqa: TRY301

        schema = DatasetConfig(
            name=LocalizedStr(
                ja=typer.prompt(_("データセットの名称（日本語）")),
                en=typer.prompt(_("データセットの名称（英語）")),
            ),
            description=LocalizedStr(
                ja=typer.prompt(_("データセットの説明（日本語）")),
                en=typer.prompt(_("データセットの説明（英語）")),
            ),
            license_=LocalizedStr(
                ja=typer.prompt(
                    _("データセットのライセンス（日本語）"),
                    default=f"{JST_YEAR}年度の行動モデル夏の学校のための利用のみ許可します。",
                ),
                en=typer.prompt(
                    _("データセットのライセンス（英語）"),
                    default=f"Use is permitted only for the {JST_YEAR} Summer Course on Behavior Modeling in Transportation Networks.",
                ),
            ),
            city=LocalizedStr(
                ja=typer.prompt(_("データセットの都市（日本語）")),
                en=typer.prompt(_("データセットの都市（英語）")),
            ),
            year=typer.prompt(_("データセットの年"), type=int),
            columns=[
                ColumnConfig(
                    name=name,
                    type_=get_polars_data_type_name(dtype),
                    description=None,
                )
                for name, dtype in zip(
                    input_dataframe.columns,
                    input_dataframe.schema.values(),
                    strict=True,
                )
            ],
            hash_=None,
        )

        schema_path = path.with_suffix(path.suffix + ".schema.yaml")
        if schema_path.exists():
            rprint(
                _(
                    "[yellow]スキーマファイルが既に存在します: {schema_path}[/yellow]",
                ).format(
                    schema_path=schema_path,
                ),
            )
            if not typer.confirm(
                _("既存のスキーマファイルを上書きしますか？"),
                default=False,
            ):
                rprint(_("[red]スキーマの推測を中止しました。[/red]"))
                raise typer.Exit(code=1)  # noqa: TRY301

        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with schema_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                schema.model_dump(),
                f,
                indent=4,
                allow_unicode=True,
                sort_keys=False,
            )

        rprint(
            _("[green]スキーマを {schema_path} に保存しました。[/green]").format(
                schema_path=schema_path,
            ),
        )

    except Exception as e:
        rprint(
            _("[red]スキーマの推測に失敗しました: {e}[/red]").format(
                e=e,
            ),
        )
        raise typer.Exit(code=1) from e


@app.command()
def generate_readme(
    file_path: Annotated[
        str,
        typer.Argument(help=_("READMEを生成するデータセットファイルのパス")),
    ],
) -> None:
    """指定されたファイルのREADMEを生成する。"""
    for lang in ("ja", "en"):
        create_readme_pdf(file_path=Path(file_path), lang=lang)
