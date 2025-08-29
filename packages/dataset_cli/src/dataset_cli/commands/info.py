# ./src/dataset_cli/src/dataset_cli/commands/info.py

from typing import Annotated

import httpx
import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from dataset_cli.utils.api import get_latest_manifest


def list_datasets() -> None:
    """利用可能なデータセットの一覧を表示します。"""
    try:
        manifest = get_latest_manifest()
    except httpx.HTTPStatusError as e:
        rprint(
            f"[bold red]エラー: GitHubリリースへのアクセスに失敗しました (HTTP {e.response.status_code})[/bold red]",
        )
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        rprint(f"[bold red]エラー: {e}[/bold red]")
        raise typer.Exit(code=1) from e

    table = Table(
        title=f"利用可能なデータセット (v{manifest.cli_version})",
        expand=True,
    )
    table.add_column("ID", style="cyan", no_wrap=True, min_width=20)
    table.add_column("名前 (ja)", style="magenta")
    table.add_column("名前 (en)", style="green")
    table.add_column("年", style="yellow")

    for dataset_id, info in sorted(manifest.datasets.items()):
        table.add_row(
            dataset_id,
            info.name.ja,
            info.name.en,
            str(info.year) if info.year else "-",
        )

    rprint(table)


def show_info(
    dataset_id: Annotated[
        str,
        typer.Argument(
            help="`davis list`で表示されるデータセットID",
            show_default=False,
        ),
    ],
) -> None:
    """指定されたデータセットの詳細情報を表示します。"""
    try:
        manifest = get_latest_manifest()
    except httpx.HTTPStatusError as e:
        rprint(
            f"[bold red]エラー: GitHubリリースへのアクセスに失敗しました (HTTP {e.response.status_code})[/bold red]",
        )
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        rprint(f"[bold red]エラー: {e}[/bold red]")
        raise typer.Exit(code=1) from e

    dataset = manifest.datasets.get(dataset_id)
    if not dataset:
        rprint(
            f"[bold red]エラー: データセット '{dataset_id}' は見つかりません。[/bold red]",
        )
        raise typer.Exit(code=1)

    rprint(
        Panel(
            f"[bold cyan]{dataset.name.ja}[/bold cyan]\n[dim]{dataset.name.en}[/dim]",
            title=f"データセット情報: {dataset_id}",
            expand=False,
        ),
    )

    if dataset.description:
        rprint("\n[bold]概要[/bold]")
        rprint(f"  (ja) {dataset.description.ja}")
        rprint(f"  (en) {dataset.description.en}")

    if dataset.dvc_files:
        file_table = Table(title="含まれるファイル")
        file_table.add_column("ファイルパス", style="cyan")
        for dvc_path in sorted(dataset.dvc_files):
            original_path = dvc_path.removesuffix(".dvc")
            file_table.add_row(original_path)
        rprint(file_table)

    if dataset.pdf_urls:
        pdf_table = Table(title="関連ドキュメント (PDF)")
        pdf_table.add_column("ファイル名", style="green")
        pdf_table.add_column("リンク", style="blue")
        for filename, urls in sorted(dataset.pdf_urls.items()):
            pdf_table.add_row(f"{filename} (ja)", f"[link={urls.ja}]{urls.ja}[/link]")
            pdf_table.add_row(f"{filename} (en)", f"[link={urls.en}]{urls.en}[/link]")
        rprint(pdf_table)
