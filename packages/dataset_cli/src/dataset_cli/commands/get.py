# ./src/dataset_cli/src/dataset_cli/commands/get.py

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich import print as rprint
from rich.progress import Progress

from dataset_cli.utils.api import get_latest_manifest


def get_dataset(
    dataset_id: Annotated[
        str,
        typer.Argument(
            help="`davis list`で表示されるデータセットID",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="保存先のディレクトリ",
            writable=True,
            file_okay=False,
            resolve_path=True,
            show_default="現在のディレクトリ",
        ),
    ] = Path(),
) -> None:
    """
    指定されたIDのデータセットをDVCを使ってダウンロードします。

    このコマンドを実行するには、`dvc`がインストールされている必要があります。
    """
    try:
        manifest = get_latest_manifest()
    except httpx.HTTPStatusError as e:
        rprint(
            f"[bold red]エラー: GitHubリリースへのアクセスに失敗しました (HTTP {e.response.status_code})[/bold red]",
        )
        rprint(
            "[dim]ネットワーク接続を確認するか、リポジトリが公開されているか確認してください。[/dim]",
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
        rprint("[dim]`davis list`で利用可能なデータセットを確認してください。[/dim]")
        raise typer.Exit(code=1)

    rprint(f"🚚 [bold cyan]{dataset_id}[/bold cyan] をダウンロードします...")

    output_dir.mkdir(exist_ok=True, parents=True)

    _download_bootstrap_config(str(manifest.bootstrap_package_url), output_dir)
    rprint("  - [green]✓[/green] DVC設定をセットアップしました。")

    dvc_files_to_pull = dataset.dvc_files
    if not dvc_files_to_pull:
        rprint(
            f"[yellow]警告: {dataset_id} にはダウンロード対象のファイルがありません。[/yellow]",
        )
        return

    rprint(f"  - データ本体をダウンロード中 ({len(dvc_files_to_pull)} ファイル)...")
    _run_dvc_pull(dvc_files_to_pull, output_dir)

    rprint(f"[bold green]✅ ダウンロード完了:[/bold green] {output_dir.resolve()}")


def _download_bootstrap_config(url: str, output_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bootstrap_zip_path = tmp_path / "bootstrap.zip"

        rprint("  - DVC設定ファイルを準備中...")
        try:
            with httpx.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=30,
            ) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with bootstrap_zip_path.open("wb") as f, Progress() as progress:
                    task = progress.add_task(
                        "[green]Downloading bootstrap...",
                        total=total_size,
                    )
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

            with zipfile.ZipFile(bootstrap_zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
        except httpx.RequestError as e:
            rprint(
                f"[bold red]エラー: DVC設定ファイルのダウンロードに失敗しました: {e}[/bold red]",
            )
            raise typer.Exit(code=1) from e
        except httpx.HTTPStatusError as e:
            rprint(
                f"[bold red]エラー: DVC設定ファイルのダウンロードに失敗しました (HTTP {e.response.status_code})[/bold red]",
            )


def _run_dvc_pull(dvc_files: list[str], output_dir: Path) -> None:
    try:
        command = ["dvc", "pull", *dvc_files, "--force"]
        subprocess.run(  # noqa: S603
            command,
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        rprint("[bold red]エラー: 'dvc'コマンドが見つかりませんでした。[/bold red]")
        rprint("[dim]この機能を利用するにはDVCのインストールが必要です。[/dim]")
        rprint("[dim]  pip install 'dvc[gdrive]'[/dim]")
        raise typer.Exit(code=1) from None
    except subprocess.CalledProcessError as e:
        rprint("[bold red]DVC pull の実行中にエラーが発生しました。[/bold red]")
        rprint(f"[dim]{e.stderr}[/dim]")
        raise typer.Exit(code=1) from e
    finally:
        # クリーンアップ
        dvc_dir = output_dir / ".dvc"
        if dvc_dir.exists():
            shutil.rmtree(dvc_dir)
