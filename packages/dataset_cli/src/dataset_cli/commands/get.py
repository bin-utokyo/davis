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
from dataset_cli.utils.dvc import DVCClient, DVCError


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


def _download_with_dvc(  # noqa: PLR0913, PLR0915
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
        config = load_user_config()
        git_path_fallback = shutil.which("git") or "git"
        git_executable_path = config.get("git_executable_path", git_path_fallback)
        try:
            # Gitリポジトリを初期化
            subprocess.run(  # noqa: S603
                [git_executable_path, "init"],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # DVCClientを初期化し、dvc pullを実行
            dvc_client = DVCClient(repo_path=tmp_path)
            dvc_client.pull(targets=dvc_files, force=True)

            # ダウンロードしたファイルを個別にコピー
            rprint("  - ファイルを最終的な出力先にコピーしています...")
            for dvc_file_rel_path_str in dvc_files:
                data_file_rel_path = Path(dvc_file_rel_path_str.removesuffix(".dvc"))

                src_path = tmp_path / data_file_rel_path
                dest_path = output_dir / data_file_rel_path

                if src_path.exists():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src_path, dest_path)
                else:
                    rprint(
                        f"[yellow]警告: dvc pull後にファイルが見つかりませんでした: {src_path}[/yellow]",
                    )

        except FileNotFoundError as e:
            # git initが失敗した場合
            rprint(
                "[bold red]エラー: 'git' コマンドが見つかりませんでした。[/bold red]",
            )
            rprint(
                "[dim]Gitがインストールされ、PATHが通っていることを確認してください。[/dim]",
            )
            raise typer.Exit(code=1) from e
        except DVCError as e:
            rprint("[bold red]DVC pull の実行中にエラーが発生しました。[/bold red]")
            rprint(f"[dim]{e}[/dim]")
            rprint(
                "[dim]Google Driveの認証に失敗したか、ファイルにアクセス権がない可能性があります。[/dim]",
            )
            rprint(
                "[dim]ブラウザが開いて認証を求められた場合は、許可してください。[/dim]",
            )
            raise typer.Exit(code=1) from e
        except subprocess.CalledProcessError as e:
            # git initが失敗した場合
            rprint("[bold red]Gitの初期化中にエラーが発生しました。[/bold red]")
            rprint(f"[dim]{e.stderr}[/dim]")
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
