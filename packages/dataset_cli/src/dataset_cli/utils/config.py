import configparser
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

# アプリケーション設定を保存するディレクトリを定義
APP_NAME = "davis"
CONFIG_DIR = Path(typer.get_app_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.ini"


class ConfigError(Exception):
    pass


@lru_cache(maxsize=1)
def find_repo_root() -> Path:
    """現在のファイルから親を遡ってGitリポジトリのルート (.gitが存在する場所) を探す"""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / ".git").is_dir():
            return parent
    msg = "Gitリポジトリのルートが見つかりませんでした。"
    raise FileNotFoundError(msg)


@lru_cache(maxsize=1)
def load_root_pyproject_toml() -> dict:
    """リポジトリルートのpyproject.tomlを読み込んで辞書として返す。"""
    try:
        root_path = find_repo_root()
        pyproject_path = root_path / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        msg = "ルートのpyproject.tomlが見つかりません。"
        raise ConfigError(msg) from e
    except tomllib.TOMLDecodeError as e:
        msg = "ルートのpyproject.tomlの解析に失敗しました。"
        raise ConfigError(msg) from e


# CLIのどこからでもこの関数を呼び出せる
def get_monorepo_config_value(section: str, key: str) -> Any:  # noqa: ANN401
    """ルートのpyproject.tomlから特定の設定値を取得する。"""
    config = load_root_pyproject_toml()
    return config.get(section, {}).get(key)


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
