import tomllib  # Python 3.11+ では tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any


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
