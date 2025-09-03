# ./src/dataset_cli/src/dataset_cli/commands/admin/release.py

import zipfile
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from dataset_cli.utils.api import get_repo_url
from dataset_cli.utils.io import generate_manifest_data


def generate_manifest(
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="出力ファイルパス",
            writable=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("dist/manifest.json"),
    repo_url: Annotated[
        str,
        typer.Option(help="GitHubリポジトリのURL (例: https://github.com/user/repo)"),
    ] = get_repo_url(),
    tag: Annotated[
        str,
        typer.Option(help="リリース対象のGitタグ (例: v1.0.0)"),
    ] = "v0.0.0",
    branch: Annotated[
        str,
        typer.Option(help="PDFなどのBlobリンクの基準となるブランチ"),
    ] = "main",
) -> None:
    """
    manifest.json を自動生成します。

    このコマンドは 'dvc url' を使用するため、DVCの認証設定が必要です。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rprint(f"[bold]'{output_path}' を生成します...[/bold]")
    bootstrap_filename = "dvc-bootstrap.zip"
    bootstrap_url = f"{repo_url}/releases/download/{tag}/{bootstrap_filename}"

    try:
        manifest = generate_manifest_data(
            cli_version=tag,
            bootstrap_url=bootstrap_url,
            repo_url=repo_url,
            branch=branch,
        )
        output_path.write_text(
            manifest.model_dump_json(indent=2, by_alias=True),
            encoding="utf-8",
        )
        rprint(f"  - [green]✓[/green] '{output_path}' を生成しました。")
    except Exception as e:
        rprint(f"[bold red]manifest.jsonの生成に失敗しました: {e}[/bold red]")
        raise typer.Exit(1) from e


def create_bootstrap(
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="出力ファイルパス",
            writable=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("dist/dvc-bootstrap.zip"),
) -> None:
    """
    dvc-bootstrap.zip を作成します。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rprint(f"[bold]'{output_path}' を作成します...[/bold]")
    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # .dvc/config をzipに追加
            config_path = Path(".dvc/config")
            if config_path.exists():
                zipf.write(config_path, arcname=".dvc/config")
            else:
                rprint(
                    f"  - [yellow]W[/yellow] '{config_path}' が見つかりません。スキップします。",
                )

            # dataディレクトリ以下の.dvcファイルをzipに追加
            data_dir = Path("data")
            dvc_files = list(data_dir.rglob("*.dvc"))
            if not dvc_files:
                rprint(
                    f"  - [yellow]W[/yellow] '{data_dir}' 内に.dvcファイルが見つかりません。",
                )
                # bootstrapファイルとしては不完全だが、処理は継続
            for dvc_file in dvc_files:
                zipf.write(dvc_file, arcname=dvc_file.as_posix())
        rprint(f"  - [green]✓[/green] '{output_path}' を作成しました。")
    except Exception as e:
        rprint(f"[bold red]dvc-bootstrap.zipの作成に失敗しました: {e}[/bold red]")
        raise typer.Exit(1) from e
