from pathlib import Path

import git

from dataset_cli.utils.i18n import _


def get_git_repo() -> git.Repo:
    """Gitリポジトリを取得するヘルパー関数。

    Returns
    -------
    git.Repo
        カレントディレクトリのGitリポジトリオブジェクト。

    Raises
    ------
    RuntimeError
        カレントディレクトリがGitリポジトリではない場合。
    """

    repo_path = Path.cwd()
    if not (repo_path / ".git").exists():
        msg = _("このディレクトリはGitリポジトリではありません。")
        raise RuntimeError(msg)

    return git.Repo(
        repo_path,
    )


repo = get_git_repo()
