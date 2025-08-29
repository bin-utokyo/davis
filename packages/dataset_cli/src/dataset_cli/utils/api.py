# ./src/dataset_cli/src/dataset_cli/utils/api.py
from functools import lru_cache

import httpx
from rich import print as rprint

from dataset_cli.schemas.manifest import Manifest
from dataset_cli.utils.config import get_monorepo_config_value


@lru_cache(maxsize=1)
def get_repo_url() -> str:
    """
    インストールされたパッケージのメタデータからリポジトリURLを動的に取得する。
    """
    repo_url = get_monorepo_config_value("project", "urls").get("Repository")
    if repo_url:
        return str(repo_url)
    msg = "pyproject.tomlに'repository' URLが見つかりません。"
    raise KeyError(msg)


def get_latest_manifest() -> Manifest:
    """GitHubの最新リリースからmanifest.jsonを取得してパースする。"""
    repo_url = get_repo_url()
    # "github.com" を "api.github.com/repos" に置換してAPIエンドポイントを構築
    api_base_url = repo_url.replace("github.com", "api.github.com/repos")
    latest_release_api_url = f"{api_base_url}/releases/latest"

    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
            rprint(f"[dim]Fetching release info from: {latest_release_api_url}[/dim]")
            response = client.get(latest_release_api_url)
            response.raise_for_status()
            release_data = response.json()

            for asset in release_data.get("assets", []):
                if asset["name"] == "manifest.json":
                    manifest_url = asset["browser_download_url"]
                    rprint(f"[dim]Fetching manifest from: {manifest_url}[/dim]")
                    manifest_response = client.get(manifest_url)
                    manifest_response.raise_for_status()
                    return Manifest.model_validate(manifest_response.json())

        msg = "最新リリースに 'manifest.json' が見つかりません。"
        raise FileNotFoundError(msg)
    except httpx.ConnectError:
        rprint("[bold red]ネットワークエラー: GitHub APIに接続できません。[/bold red]")
        raise
