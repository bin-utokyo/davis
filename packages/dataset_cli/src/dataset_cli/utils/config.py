import configparser
from functools import lru_cache
from pathlib import Path

import typer
from rich import print as rprint

# アプリケーション設定を保存するディレクトリを定義
APP_NAME = "davis"
CONFIG_DIR = Path(typer.get_app_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.ini"


class ConfigError(Exception):
    pass


REPOSITORY_NAME = "bin-utokyo/davis"


@lru_cache(maxsize=1)
def get_repo_url() -> str:
    """リポジトリのURLを取得する。"""
    return f"https://github.com/{REPOSITORY_NAME}"


def save_user_config(config_data: dict[str, str]) -> None:
    """ユーザー設定を保存"""
    from dataset_cli.utils.i18n import _  # noqa: PLC0415

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config = configparser.ConfigParser()
        config["default"] = config_data
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            config.write(f)
    except OSError as e:
        rprint(
            _(
                "[bold red]エラー: 設定ファイルの書き込みに失敗しました: {config_file}[/bold red]",
            ).format(config_file=CONFIG_FILE),
        )
        rprint(f"[dim]{e}[/dim]")
        raise typer.Exit(code=1) from e


def load_user_config() -> dict[str, str]:
    """ユーザー設定を読み込み"""
    if not CONFIG_FILE.exists():
        return {}

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE, encoding="utf-8")
        if "default" in config:
            return dict(config["default"])
    except (OSError, configparser.Error) as e:
        rprint(
            (
                f"[bold red]エラー: 設定ファイルの読み込みに失敗しました: {CONFIG_FILE}[/bold red]",
            ),
        )
        rprint(
            f"[bold red]Error: Failed to read config file: {CONFIG_FILE}[/bold red]",
        )
        rprint(f"[dim]{e}[/dim]")
        return {}
    return {}
