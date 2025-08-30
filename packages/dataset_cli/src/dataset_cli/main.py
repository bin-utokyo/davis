# ./src/dataset_cli/src/dataset_cli/main.py

import typer

from .commands import data, get, info, setup
from .commands.admin import release

app = typer.Typer(
    name="davis",
    help="データセット配布用CLIツール",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

# --- エンドユーザー向けコマンド ---
app.command("setup", help="このツールの初回セットアップを行います。")(
    setup.setup_davis,
)
app.command("get", help="データセットをダウンロードします。")(get.get_dataset)
app.command("list", help="利用可能なデータセットの一覧を表示します。")(
    info.list_datasets,
)
app.command("info", help="データセットの詳細情報を表示します。")(info.show_info)


# --- 管理者向けコマンド ---
manage_app = typer.Typer(
    name="manage",
    help="データセット管理者向けのコマンド群 (要認証)",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


manage_app.command(
    "generate-manifest",
    help="manifest.json を自動生成します。",
)(release.generate_manifest)
manage_app.command(
    "create-bootstrap",
    help="dvc-bootstrap.zip を作成します。",
)(release.create_bootstrap)

manage_app.add_typer(data.app, name="data", help="データ検証・整形コマンド")


app.add_typer(manage_app)


if __name__ == "__main__":
    app()
