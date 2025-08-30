import typer

from .commands import get, info, setup
from .commands.admin import local, release

app = typer.Typer(
    name="davis",
    help="データセット配布用CLIツール",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

# --- エンドユーザー向けコマンド ---
app.command("setup", help="このツールの初回セットアップを行います。")(setup.setup_davis)
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

# Add the local commands to the manage app
manage_app.add_typer(
    local.app,
    name="local",
    help="データセットのローカル管理（検証・PDF生成・同期）",
)

app.add_typer(manage_app)

if __name__ == "__main__":
    app()
