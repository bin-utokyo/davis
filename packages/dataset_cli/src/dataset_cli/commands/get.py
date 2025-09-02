# ./packages/dataset_cli/src/dataset_cli/commands/get.py

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich import print as rprint
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from dataset_cli.schemas.manifest import Manifest
from dataset_cli.utils.api import get_latest_manifest
from dataset_cli.utils.config import load_user_config


def get_dataset(
    dataset_id: Annotated[
        str,
        typer.Argument(
            help="`davis list`で表示されるデータセットID (ディレクトリも指定可)",
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
    """指定されたIDまたはディレクトリパスに一致するデータセットとドキュメントをダウンロードします。"""
    config = load_user_config()
    folder_id, client_id, client_secret = _validate_config(config)
    manifest = _load_manifest_safe()

    dvc_files_to_pull, pdf_urls_to_download = _collect_targets(
        dataset_id,
        manifest,
        output_dir,
    )

    rprint(
        f"🚚 [bold cyan]{dataset_id}[/bold cyan] 以下のデータセット ({len(dvc_files_to_pull)}データファイル, {len(pdf_urls_to_download)}ドキュメント) をダウンロードします...",
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    _download_with_dvc(
        dvc_files_to_pull,
        manifest,
        folder_id,
        client_id,
        client_secret,
        output_dir,
    )
    _download_pdfs(pdf_urls_to_download)
    _cleanup_dvc_files(output_dir)

    rprint(
        f"\n[bold green]✅ 全ての処理が完了しました:[/bold green] {output_dir.resolve()}",
    )


def _validate_config(config: dict) -> tuple[str, str, str]:
    """ユーザー設定を検証して必要なキーを返す。"""
    folder_id = config.get("gdrive_folder_id")
    client_id = config.get("gdrive_client_id")
    client_secret = config.get("gdrive_client_secret")

    if not folder_id or not client_id or not client_secret:
        rprint("[bold red]エラー: CLIの設定が不完全です。[/bold red]")
        rprint(
            "[dim]'davis setup' を実行して、初回セットアップを完了してください。[/dim]",
        )
        raise typer.Exit(code=1)

    return folder_id, client_id, client_secret


def _load_manifest_safe() -> Manifest:
    """マニフェストを取得。失敗したらエラーメッセージを出力して終了。"""
    try:
        return get_latest_manifest()
    except Exception as e:
        rprint(
            "[bold red]エラー: データセットの目録(manifest)の取得に失敗しました。[/bold red]",
        )
        rprint(f"[dim]{e}[/dim]")
        raise typer.Exit(code=1) from e


def _collect_targets(
    dataset_id: str,
    manifest: Manifest,
    output_dir: Path,
) -> tuple[list[str], list[tuple[str, Path]]]:
    """対象の DVC ファイルと PDF URL を収集。"""
    dvc_files_to_pull: list[str] = []
    pdf_urls_to_download: list[tuple[str, Path]] = []
    found = False

    for ds_id, ds_info in manifest.datasets.items():
        if ds_id == dataset_id or ds_id.startswith(f"{dataset_id}/"):
            dvc_files_to_pull.extend(ds_info.dvc_files)
            for filename, urls in ds_info.pdf_urls.items():
                pdf_base = Path(ds_info.dvc_files[0]).parent / Path(filename).stem
                pdf_urls_to_download.append(
                    (str(urls.ja), output_dir / f"{pdf_base}.ja.pdf"),
                )
                pdf_urls_to_download.append(
                    (str(urls.en), output_dir / f"{pdf_base}.en.pdf"),
                )
            found = True

    if not found:
        rprint(
            f"[bold red]エラー: データセット '{dataset_id}' は見つかりません。[/bold red]",
        )
        raise typer.Exit(code=1)

    return dvc_files_to_pull, pdf_urls_to_download


def _download_with_dvc(  # noqa: PLR0913
    dvc_files: list[str],
    manifest: Manifest,
    folder_id: str,
    client_id: str,
    client_secret: str,
    output_dir: Path,
) -> None:
    """DVC を使ってデータをダウンロード。"""
    if not dvc_files:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bootstrap_url = str(manifest.bootstrap_package_url)
        bootstrap_zip_path = tmp_path / "bootstrap.zip"

        try:
            with httpx.stream(
                "GET",
                bootstrap_url,
                follow_redirects=True,
                timeout=30,
            ) as response:
                response.raise_for_status()
                with bootstrap_zip_path.open("wb") as f:
                    f.writelines(response.iter_bytes())

            with zipfile.ZipFile(bootstrap_zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_path)
        except httpx.HTTPStatusError as e:
            rprint(
                f"[bold red]HTTPエラー: ブートストラップパッケージのダウンロードに失敗しました (HTTP {e.response.status_code})[/bold red]",
            )
            local_bootstrap = Path("dist/dvc-bootstrap.zip")
            if local_bootstrap.exists():
                rprint(
                    f"[yellow]ローカルの '{local_bootstrap}' を使用します。[/yellow]",
                )
                shutil.copy(local_bootstrap, bootstrap_zip_path)
                with zipfile.ZipFile(bootstrap_zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_path)
            else:
                raise typer.Exit(code=1) from e
        except Exception as e:
            rprint(
                f"[bold red]ブートストラップパッケージの処理中にエラーが発生しました: {e}[/bold red]",
            )
            raise typer.Exit(code=1) from e

        # DVC config を書き込み
        dvc_config_path = tmp_path / ".dvc" / "config"
        dvc_config_path.parent.mkdir(exist_ok=True)
        dvc_config_content = f"""
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://{folder_id}
    gdrive_client_id = {client_id}
    gdrive_client_secret = {client_secret}
"""
        dvc_config_path.write_text(dvc_config_content, encoding="utf-8")

        rprint("  - DVCコマンドを実行し、データをダウンロードします...")
        try:
            subprocess.run(
                ["git", "init"],  # noqa: S607
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
            command = ["dvc", "pull", *dvc_files, "--force"]
            subprocess.run(  # noqa: S603
                command,
                cwd=tmp_path,
                check=True,
                text=True,
                encoding="utf-8",
                capture_output=False,
            )

            # ダウンロードした data/ を移動
            data_dir_in_tmp = tmp_path / "data"
            if data_dir_in_tmp.exists():
                shutil.copytree(data_dir_in_tmp, output_dir, dirs_exist_ok=True)

        except FileNotFoundError as e:
            rprint(
                "[bold red]エラー: 'dvc' または 'git' コマンドが見つかりませんでした。[/bold red]",
            )
            rprint(
                "[dim]これらのコマンドがインストールされ、PATHが通っていることを確認してください。[/dim]",
            )
            raise typer.Exit(code=1) from e
        except subprocess.CalledProcessError as e:
            rprint("[bold red]DVC pull の実行中にエラーが発生しました。[/bold red]")
            rprint(
                "[dim]Google Driveの認証に失敗したか、ファイルにアクセス権がない可能性があります。[/dim]",
            )
            rprint(
                "[dim]ブラウザが開いて認証を求められた場合は、許可してください。[/dim]",
            )
            raise typer.Exit(code=1) from e

    rprint(f"\n[bold green]✅ ダウンロード完了:[/bold green] {output_dir.resolve()}")


def _download_pdfs(pdf_urls_to_download: list[tuple[str, Path]]) -> None:
    """PDF のダウンロード処理。"""
    if not pdf_urls_to_download:
        return

    rprint("  - ドキュメント(PDF)をダウンロード中...")
    with (
        httpx.Client(follow_redirects=True, timeout=30) as client,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress,
    ):
        task = progress.add_task(
            "[cyan]Downloading PDFs...",
            total=len(pdf_urls_to_download),
        )
        for url, dest_path in pdf_urls_to_download:
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                response = client.get(str(url))
                response.raise_for_status()
                with dest_path.open("wb") as f:
                    f.write(response.content)
                progress.update(task, advance=1)
            except httpx.RequestError:
                rprint(f"[yellow]警告: PDFのダウンロードに失敗しました: {url}[/yellow]")
            except httpx.HTTPStatusError as e:
                rprint(
                    f"[yellow]警告: HTTPエラー {e.response.status_code} - {e.response.reason_phrase} でダウンロードに失敗しました: {url}[/yellow]",
                )


def _cleanup_dvc_files(output_dir: Path) -> None:
    """不要な .dvc ファイルを削除。"""
    rprint("  - クリーンアップ中...")
    deleted_count = 0
    for dvc_file_path in output_dir.rglob("*.dvc"):
        if dvc_file_path.is_file():
            dvc_file_path.unlink()
            deleted_count += 1
    rprint(f"  - [green]✓[/green] {deleted_count}個の.dvcファイルを削除しました。")
