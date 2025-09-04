# ./src/dataset_cli/src/dataset_cli/utils/api.py
import json
from pathlib import Path

import httpx
from rich import print as rprint

from dataset_cli.schemas.manifest import Manifest
from dataset_cli.utils.config import get_repo_url
from dataset_cli.utils.i18n import _


def get_latest_manifest() -> Manifest:
    """GitHubの最新リリースからmanifest.jsonを取得してパースする。"""
    local_manifest = Path("dist/manifest.json")
    if local_manifest.exists():
        rprint(_("[yellow]ローカルの 'dist/manifest.json' を使用します。[/yellow]"))
        content = json.loads(
            local_manifest.read_text(encoding="utf-8"),
        )
        return Manifest.model_validate(content)

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

        msg = _("最新リリースに 'manifest.json' が見つかりません。")
        raise FileNotFoundError(msg)
    except httpx.ConnectError:
        rprint(
            _("[bold red]ネットワークエラー: GitHub APIに接続できません。[/bold red]"),
        )
        raise
    except httpx.HTTPStatusError as e:
        rprint(
            _(
                "[bold red]HTTPエラー: {status_code} - {reason_phrase}[/bold red]",
            ).format(
                status_code=e.response.status_code,
                reason_phrase=e.response.reason_phrase,
            ),
        )
        manifest_json = Path("dist/manifest.json")
        if manifest_json.exists():
            rprint(_("[yellow]ローカルの 'dist/manifest.json' を使用します。[/yellow]"))
            content = json.loads(
                manifest_json.read_text(encoding="utf-8"),
            )
            return Manifest.model_validate(content)
        msg = _("GitHub APIからmanifest.jsonを取得できません。")
        raise httpx.HTTPStatusError(
            msg,
            request=e.request,
            response=e.response,
        ) from e
