# ./packages/dataset_cli/src/dataset_cli/commands/setup.py

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich import print as rprint

from dataset_cli.utils.config import CONFIG_FILE, load_user_config, save_user_config
from dataset_cli.utils.i18n import _


def setup_davis() -> None:  # noqa: C901, PLR0912, PLR0915
    """
    Davis CLI の初回セットアップを対話的に行います。
    Google DriveのフォルダIDと、ご自身の認証情報を設定します。
    """
    rprint(
        _("[bold]Davisデータセットツールの初回セットアップを開始します。[/bold]"),
    )

    rprint(_("\n[bold]依存関係をチェックしています...[/bold]"))

    # Check for git
    if not (git_path := shutil.which("git")):
        rprint(_("[bold red]エラー: 'git' コマンドが見つかりません。[/bold red]"))
        rprint(_("Gitをインストールし、システムのPATHに登録してください。"))
        rprint(_("Gitは https://git-scm.com/downloads からダウンロードできます。"))
        if platform.system() == "Darwin":
            rprint(
                _(
                    "macOSをご利用の場合は、Xcodeコマンドラインツールに含まれています。また、Homebrewをお使いの場合は `brew install git` でインストールできます。",
                ),
            )
        elif platform.system() == "Windows":
            rprint(
                _(
                    "Windowsをご利用の場合は、Git for Windows (https://gitforwindows.org/) をインストールしてください。",
                ),
            )
        raise typer.Abort

    rprint(_("  - [green]✅ git はインストール済みです。[/green]"))
    rprint(
        _("    [dim]Gitのパス: {git_path}[/dim]").format(
            git_path=Path(git_path).resolve(),
        ),
    )

    # Check for dvc
    dvc_path = shutil.which("dvc")
    if not dvc_path:
        # If not in PATH, check next to the python executable
        py_executable_dir = Path(sys.executable).parent
        dvc_in_env = py_executable_dir / "dvc"
        if dvc_in_env.is_file() and os.access(dvc_in_env, os.X_OK):
            dvc_path = str(dvc_in_env)

    if not dvc_path:
        rprint(_("[bold red]エラー: 'dvc' コマンドが見つかりません。[/bold red]"))
        rprint(
            _(
                "PATHまたは現在のPython環境で 'dvc' 実行可能ファイルが見つかりませんでした。",
            ),
        )
        # if uv is installed, suggest using uv
        if shutil.which("uv"):
            install_cmd = "uv tool install dvc[gdrive]"
            execute_auto_install = typer.confirm(
                _(
                    "uvが見つかりました。`{install_cmd}` を実行してdvcをインストールしますか？",
                ).format(install_cmd=install_cmd),
                default=True,
            )
            if execute_auto_install:
                rprint(
                    _("[dim]実行中: {install_cmd}[/dim]").format(
                        install_cmd=install_cmd,
                    ),
                )
                subprocess.run(  # noqa: S602
                    install_cmd,
                    shell=True,
                    check=False,
                )
                dvc_path = shutil.which("dvc")
        if not dvc_path:
            rprint(
                _("dvcをインストールし、システムのPATHに登録してください。"),
            )
            # Recommend installing via pip if uv is not available
            rprint(
                _(
                    "`uv tool install dvc[gdrive]` または `pip install dvc[gdrive]` でインストールできます。",
                ),
            )
            # Provide platform-specific installation hints
            if platform.system() == "Darwin":
                rprint(
                    _(
                        "Homebrewをお使いの場合は `brew install dvc` でもインストールできます。",
                    ),
                )
                rprint(_("https://dvc.org/doc/install/macos も参照してください。"))
            elif platform.system() == "Windows":
                rprint(
                    _(
                        "Chocolateyをお使いの場合は `choco install dvc` でもインストールできます。",
                    ),
                )
                rprint(_("https://dvc.org/doc/install/windows も参照してください。"))
        raise typer.Abort

    dvc_abs_path = str(Path(dvc_path).resolve())
    rprint(_("  - [green]✅ dvc はインストール済みです。[/green]"))
    rprint(
        _("    [dim]DVCのパス: {dvc_abs_path}[/dim]").format(
            dvc_abs_path=dvc_abs_path,
        ),
    )
    rprint(
        _(
            "[dim]このツールはDVCを利用してGoogle Driveからデータをダウンロードします。[/dim]",
        ),
    )
    rprint(
        _(
            "[dim]設定情報はすべてローカルPCに保存され、外部に送信されることはありません。[/dim]",
        ),
    )

    current_config = load_user_config()

    # 既存の設定があれば表示
    if current_config:
        rprint(_("\n[bold]現在の設定:[/bold]"))
        for key, value in current_config.items():
            # シークレットは一部をマスクして表示
            display_value = (
                f"{value[:4]}...{value[-4:]}"
                if "secret" in key and len(value) > 8  # noqa: PLR2004
                else value
            )
            rprint(f"  - [cyan]{key}[/cyan]: [white]{display_value}[/white]")
        if not typer.confirm(_("\n設定を上書きしますか？")):
            rprint(_("セットアップを中止しました。"))
            raise typer.Abort

    rprint(
        _(
            "\n[bold]管理者から提供された案内に記載されている情報を入力してください:[/bold]",
        ),
    )

    folder_id = typer.prompt(_("  - Google Drive フォルダID"))
    client_id = typer.prompt("  - Google Cloud Client ID")
    client_secret = typer.prompt("  - Google Cloud Client Secret", hide_input=True)

    new_config = {
        "gdrive_folder_id": folder_id,
        "gdrive_client_id": client_id,
        "gdrive_client_secret": client_secret,
        "git_executable_path": str(Path(git_path).resolve()),
        "dvc_executable_path": dvc_abs_path,
    }
    save_user_config(new_config)

    rprint(
        _("\n[bold green]✅ 設定を {config_file} に保存しました。[/bold green]").format(
            config_file=CONFIG_FILE,
        ),
    )
    rprint(_("これで `davis get <DATASET_ID>` コマンドが使用できます。"))
