# ./src/dataset_cli/src/dataset_cli/commands/info.py

import webbrowser
from typing import Annotated

import httpx
import typer
from pydantic import HttpUrl
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dataset_cli.schemas.dataset_config import (
    DatasetConfig,
)
from dataset_cli.utils.api import get_latest_manifest
from dataset_cli.utils.config import get_repo_url, load_user_config
from dataset_cli.utils.i18n import _
from dataset_cli.utils.parser import parse_yaml_and_validate

MIN_RECOMMENDED_WIDTH = 120  # この幅 (文字数) より狭いと警告


def list_datasets() -> None:
    """利用可能なデータセットの一覧を表示します。"""
    try:
        manifest = get_latest_manifest()
    except httpx.HTTPStatusError as e:
        rprint(
            _(
                "[bold red]エラー: GitHubリリースへのアクセスに失敗しました (HTTP {status_code})[/bold red]",
            ).format(status_code=e.response.status_code),
        )
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        rprint(_("[bold red]エラー: {e}[/bold red]").format(e=e))
        raise typer.Exit(code=1) from e

    table = Table(
        title=_("利用可能なデータセット (v{cli_version})").format(
            cli_version=manifest.cli_version,
        ),
        expand=True,
    )
    table.add_column(_("ID"), style="cyan", no_wrap=True, min_width=20)
    table.add_column(_("名前 (ja)"), style="magenta")
    table.add_column(_("名前 (en)"), style="green")
    table.add_column(_("年"), style="yellow")

    for dataset_id, info in sorted(manifest.datasets.items()):
        table.add_row(
            dataset_id,
            info.name.ja,
            info.name.en,
            str(info.year) if info.year else "-",
        )

    rprint(table)

    console = Console()
    if console.width < MIN_RECOMMENDED_WIDTH:
        rprint(
            Panel(
                _(
                    "[bold]ターミナルの幅が狭いため、表が見づらい可能性があります。幅を広げることをお勧めします (現在の幅: {width}文字）。[/bold]",
                ).format(width=console.width),
                title=_("注意"),
            ),
        )


def show_info(  # noqa: C901, PLR0912, PLR0915
    dataset_id: Annotated[
        str,
        typer.Argument(
            help=_("`davis list`で表示されるデータセットID"),
            show_default=False,
        ),
    ],
    *,
    open_in_browser: Annotated[
        bool,
        typer.Option(
            "--open",
            "-o",
            help=_("関連ドキュメントのPDFリンクをブラウザで開く"),
        ),
    ] = False,
) -> None:
    """指定されたデータセットの詳細情報を表示します。"""
    try:
        manifest = get_latest_manifest()
    except httpx.HTTPStatusError as e:
        rprint(
            _(
                "[bold red]エラー: GitHubリリースへのアクセスに失敗しました (HTTP {status_code})[/bold red]",
            ).format(status_code=e.response.status_code),
        )
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        rprint(_("[bold red]エラー: {e}[/bold red]").format(e=e))
        raise typer.Exit(code=1) from e

    dataset = manifest.datasets.get(dataset_id)
    if not dataset:
        rprint(
            _(
                "[bold red]エラー: データセット '{dataset_id}' は見つかりません。[/bold red]",
            ).format(dataset_id=dataset_id),
        )
        raise typer.Exit(code=1)

    rprint(
        Panel(
            _(
                "[bold cyan]{name_ja}[/bold cyan]\n[dim]{name_en}[/dim]",
            ).format(name_ja=dataset.name.ja, name_en=dataset.name.en),
            title=_("データセット情報: {dataset_id}").format(dataset_id=dataset_id),
        ),
    )

    if dataset.description:
        rprint(_("\n[bold]概要[/bold]"))
        rprint(f"  (ja) {dataset.description.ja}")
        rprint(f"  (en) {dataset.description.en}")
        rprint("")

    if dataset.dvc_files:
        file_table = Table(title=_("含まれるファイル"), expand=True)
        file_table.add_column(_("ファイルパス"), style="cyan")
        for dvc_path in sorted(dataset.dvc_files):
            original_path = dvc_path.removesuffix(".dvc")
            file_table.add_row(original_path)
        rprint(file_table)

    if dataset.dvc_files:
        for dvc_path in sorted(dataset.dvc_files):
            base_file = dvc_path.removesuffix(".dvc")
            schema_file_path = f"{base_file}.schema.yaml"
            repo_url = get_repo_url()
            schema_raw_url = f"{repo_url}/raw/main/{schema_file_path}"

            try:
                schema_content: DatasetConfig = parse_yaml_and_validate(
                    HttpUrl(schema_raw_url),
                    DatasetConfig,
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == httpx.codes.NOT_FOUND:
                    rprint(
                        _(
                            "[yellow]警告: スキーマファイルが見つかりません。"
                            "表形式のファイルではないか、スキーマが提供されていない可能性があります。"
                            "[/yellow]",
                        ).format(schema_file_path=schema_file_path),
                    )
                    continue
                rprint(
                    _(
                        "[bold red]エラー: スキーマファイルへのアクセスに失敗しました (HTTP {status_code})[/bold red]",
                    ).format(status_code=e.response.status_code),
                )
                raise typer.Exit(code=1) from e

            col_table = Table(
                title=_("スキーマ"),
                expand=True,
            )
            col_table.add_column(_("列名"), style="cyan")
            col_table.add_column(_("データ型"), style="magenta")
            col_table.add_column(_("説明"))
            lang = load_user_config().get("lang", "ja")
            for col in schema_content.columns:
                description = col.description and (
                    col.description.ja if lang == "ja" else col.description.en
                )
                col_table.add_row(
                    col.name,
                    col.type_.name,
                    description or "-",
                )
            rprint(col_table)

    if dataset.pdf_urls:
        pdf_table = Table(title=_("関連ドキュメント (PDF)"), expand=True)
        pdf_table.add_column(_("ファイル名"), style="green")
        pdf_table.add_column(_("リンク"), style="blue")
        for filename, urls in sorted(dataset.pdf_urls.items()):
            pdf_table.add_row(f"{filename} (ja)", f"[link={urls.ja}]{urls.ja}[/link]")
            pdf_table.add_row(f"{filename} (en)", f"[link={urls.en}]{urls.en}[/link]")
        rprint(pdf_table)

        if open_in_browser:
            user_config = load_user_config()
            lang = user_config.get("lang", "ja")
            if lang not in ("ja", "en"):
                lang = "ja"
            rprint(_("ブラウザでPDFリンクを開きます..."))
            match lang:
                case "ja":
                    for urls in dataset.pdf_urls.values():
                        webbrowser.open(str(urls.ja))
                case "en":
                    for urls in dataset.pdf_urls.values():
                        webbrowser.open(str(urls.en))
