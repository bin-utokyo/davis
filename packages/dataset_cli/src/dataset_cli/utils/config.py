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
    """
    ユーザー設定を指定された辞書データで上書き保存します。

    Args:
        config_data (Dict[str, str]): 保存する設定データ。
    """
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config = configparser.ConfigParser()
        config["default"] = config_data
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            config.write(f)
    except OSError as e:
        rprint(
            f"[bold red]エラー: 設定ファイルの書き込みに失敗しました: {CONFIG_FILE}[/bold red]",
        )
        rprint(f"[dim]{e}[/dim]")
        raise typer.Exit(code=1) from e


def load_user_config() -> dict[str, str]:
    """
    ユーザー設定をファイルから読み込みます。

    Returns:
        Dict[str, str]: 読み込まれた設定データ。存在しない場合は空の辞書。
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE, encoding="utf-8")
        if "default" in config:
            return dict(config["default"])

    except (OSError, configparser.Error) as e:
        rprint(
            f"[bold red]エラー: 設定ファイルの読み込みに失敗しました: {CONFIG_FILE}[/bold red]",
        )
        rprint(f"[dim]{e}[/dim]")
        return {}
    return {}
