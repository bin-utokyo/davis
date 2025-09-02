# ./packages/dataset_cli/src/dataset_cli/commands/setup.py

import typer
from rich import print as rprint

from dataset_cli.utils.config import CONFIG_FILE, load_user_config, save_user_config


def setup_davis() -> None:
    """
    Davis CLI の初回セットアップを対話的に行います。
    Google DriveのフォルダIDと、ご自身の認証情報を設定します。
    """
    rprint(
        "[bold]Davisデータセットツールの初回セットアップを開始します。[/bold]",
    )
    rprint(
        "[dim]このツールはDVCを利用してGoogle Driveからデータをダウンロードします。[/dim]",
    )
    rprint(
        "[dim]設定情報はすべてローカルPCに保存され、外部に送信されることはありません。[/dim]",
    )

    current_config = load_user_config()

    # 既存の設定があれば表示
    if current_config:
        rprint("\n[bold]現在の設定:[/bold]")
        for key, value in current_config.items():
            # シークレットは一部をマスクして表示
            display_value = (
                f"{value[:4]}...{value[-4:]}"
                if "secret" in key and len(value) > 8  # noqa: PLR2004
                else value
            )
            rprint(f"  - [cyan]{key}[/cyan]: {display_value}")
        if not typer.confirm("\n設定を上書きしますか？"):
            rprint("セットアップを中止しました。")
            raise typer.Abort

    rprint(
        "\n[bold]管理者から提供された案内に記載されている情報を入力してください:[/bold]",
    )

    folder_id = typer.prompt("  - Google Drive フォルダID")
    client_id = typer.prompt("  - Google Cloud Client ID")
    client_secret = typer.prompt("  - Google Cloud Client Secret", hide_input=True)

    new_config = {
        "gdrive_folder_id": folder_id,
        "gdrive_client_id": client_id,
        "gdrive_client_secret": client_secret,
    }
    save_user_config(new_config)

    rprint(f"\n[bold green]✅ 設定を {CONFIG_FILE} に保存しました。[/bold green]")
    rprint("これで `davis get <DATASET_ID>` コマンドが使用できます。")
