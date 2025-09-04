import contextlib
import gettext
import locale
from pathlib import Path
from typing import Literal, cast

import typer

from dataset_cli.commands import config
from dataset_cli.utils.config import load_user_config

from .commands import get, info, setup
from .commands.admin import local, release
from .utils.i18n import _


def init_translation() -> None:
    """言語コードを指定して翻訳を初期化する"""
    user_config = load_user_config()
    lang = cast("Literal['ja', 'en']", user_config.get("lang", "ja"))

    locale_dir = (Path(__file__).parent.parent.parent / "locales").resolve()
    assert locale_dir.exists(), f"Locale directory does not exist: {locale_dir}"

    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, "")

    translator = gettext.translation(
        "messages",
        locale_dir,
        languages=[lang],
        fallback=True,
    )

    user_config = load_user_config()
    _ = translator.gettext


app = typer.Typer(
    name="davis",
    help=_("データセット配布用CLIツール"),
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

# --- エンドユーザー向けコマンド ---
app.command("setup", help=_("このツールの初回セットアップを行います。"))(
    setup.setup_davis,
)
app.command("get", help=_("データセットをダウンロードします。"))(get.get_dataset)
app.command("list", help=_("利用可能なデータセットの一覧を表示します。"))(
    info.list_datasets,
)
app.command("info", help=_("データセットの詳細情報を表示します。"))(info.show_info)

app.add_typer(
    config.app,
    name="config",
    help=_("CLIの設定を管理します。"),
)

# --- 管理者向けコマンド ---
manage_app = typer.Typer(
    name="manage",
    help=_("データセット管理者向けのコマンド群 (要認証)"),
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

manage_app.command(
    "generate-manifest",
    help=_("manifest.json を自動生成します。"),
)(release.generate_manifest)
manage_app.command(
    "create-bootstrap",
    help=_("dvc-bootstrap.zip を作成します。"),
)(release.create_bootstrap)

# Add the local commands to the manage app
manage_app.add_typer(
    local.app,
    name="local",
    help=_("データセットのローカル管理（検証・PDF生成・同期）"),
)

app.add_typer(manage_app)


if __name__ == "__main__":
    app()
