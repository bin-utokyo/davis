# ./src/dataset_cli/src/dataset_cli/commands/admin/release.py

import zipfile
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from dataset_cli.utils.api import get_repo_url
from dataset_cli.utils.i18n import _
from dataset_cli.utils.io import generate_file_hash, generate_manifest_data


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
        bootstrap_hash = create_bootstrap(
            Path("dist/dvc-bootstrap.zip"),
        )
        manifest = generate_manifest_data(
            cli_version=tag,
            bootstrap_url=bootstrap_url,
            repo_url=repo_url,
            branch=branch,
            bootstrap_package_hash=bootstrap_hash,
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
            help=_("出力ファイルパス"),
            writable=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("dist/dvc-bootstrap.zip"),
) -> str:
    """
    dvc-bootstrap.zip を作成します。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rprint(
        _("[bold]'{output_path}' を作成します...[/bold]").format(
            output_path=output_path,
        ),
    )
    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # .dvc/config をzipに追加
            config_path = Path(".dvc/config")
            if config_path.exists():
                zipf.write(config_path, arcname=".dvc/config")
            else:
                rprint(
                    _(
                        "  - [yellow]W[/yellow] '{config_path}' が見つかりません。スキップします。",
                    ).format(config_path=config_path),
                )

            # dataディレクトリ以下の.dvcファイルをzipに追加
            data_dir = Path("data")
            dvc_files = list(data_dir.rglob("*.dvc"))
            if not dvc_files:
                rprint(
                    _(
                        "  - [yellow]W[/yellow] '{data_dir}' 内に.dvcファイルが見つかりません。",
                    ).format(data_dir=data_dir),
                )
                # bootstrapファイルとしては不完全だが、処理は継続
            for dvc_file in dvc_files:
                zipf.write(dvc_file, arcname=dvc_file.as_posix())
        rprint(
            _("  - [green]✓[/green] '{output_path}' を作成しました。").format(
                output_path=output_path,
            ),
        )

        # Calculate hash after creation
        bootstrap_hash = generate_file_hash(output_path)
        rprint(
            _(
                "  - [dim]生成されたブートストラップパッケージのハッシュ: {bootstrap_hash}[/dim]",
            ).format(bootstrap_hash=bootstrap_hash),
        )
    except Exception as e:
        rprint(
            _("[bold red]dvc-bootstrap.zipの作成に失敗しました: {e}[/bold red]").format(
                e=e,
            ),
        )
        raise typer.Exit(1) from e
    return bootstrap_hash
