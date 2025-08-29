# ./src/dataset_cli/src/dataset_cli/utils/dvc.py

import subprocess
from collections.abc import Sequence
from pathlib import Path

import typer
from rich import print as rprint


class DVCError(Exception):
    """DVCコマンドの実行に失敗した際に送出されるカスタム例外。"""

    def __init__(
        self,
        message: str,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return (
            f"{super().__str__()}\n"
            f"Return Code: {self.return_code}\n"
            f"--- STDOUT ---\n{self.stdout}\n"
            f"--- STDERR ---\n{self.stderr}"
        )


class DVCClient:
    """
    DVCコマンドをラップし、Pythonから実行するためのクライアントクラス。

    Attributes
    ----------
        repo_path (Path): DVCリポジトリのルートパス。
    """

    def __init__(self, repo_path: Path = Path()) -> None:
        """
        DVCClientを初期化します。

        Args:
            repo_path (Path, optional): DVCリポジトリのパス。
                                        デフォルトはカレントディレクトリ。
        """
        self.repo_path = repo_path.resolve()
        if not (self.repo_path / ".dvc").exists():
            rprint(
                f"[yellow]警告: 指定されたパスに.dvcディレクトリが見つかりません: {self.repo_path}. DVCリポジトリを初期化してください。[/yellow]",
            )

    def _run_command(
        self,
        command: list[str],
        *,
        stream_output: bool = True,
    ) -> tuple[str, str]:
        """
        指定されたDVCコマンドを実行する内部ヘルパーメソッド。

        Args:
            command (list[str]): 'dvc'に続くコマンドと引数のリスト。
            stream_output (bool): Trueの場合、出力をリアルタイムでコンソールに表示する。

        Returns:
            tuple[str, str]: (標準出力, 標準エラー) のタプル。

        Raises:
            DVCError: コマンドの実行が失敗した場合。
        """
        full_command = ["dvc", *command]
        rprint(f"[cyan]実行中: {' '.join(full_command)}[/cyan]")

        try:
            process = subprocess.run(  # noqa: S603
                full_command,
                cwd=self.repo_path,
                check=True,
                capture_output=True,  # Always capture output
                text=True,
                encoding="utf-8",
            )
            stdout = process.stdout
            stderr = process.stderr

            if stream_output:
                if stdout:
                    rprint(f"[stdout] {stdout.strip()}")
                if stderr:
                    rprint(f"[stderr][red]{stderr.strip()}[/red]")

            return stdout, stderr  # noqa: TRY300

        except FileNotFoundError as e:
            rprint(
                "[red]エラー: 'dvc'コマンドが見つかりませんでした。DVCはインストールされていますか？[/red]",
            )
            raise typer.Exit(code=127) from e
        except subprocess.CalledProcessError as e:
            msg = f"DVCコマンド '{' '.join(full_command)}' の実行に失敗しました。"
            raise DVCError(
                msg,
                e.returncode,
                e.stdout or "",
                e.stderr or "",
            ) from e

    def url(self, target: str) -> str:
        """
        DVCで管理されているファイルの公開URLを取得します。(要認証)

        Args:
            target (str): 対象ファイルのパス。

        Returns:
            str: 公開URL。
        """
        stdout, _ = self._run_command(["url", target], stream_output=False)
        return stdout.strip()

    def add(self, targets: Sequence[str]) -> None:
        """
        `dvc add` を実行して、ファイルをDVCの管理下に置きます。

        Args:
            targets (Sequence[str]): 対象のファイルまたはディレクトリのリスト。
        """
        self._run_command(["add", *targets], stream_output=False)

    def commit(
        self,
        targets: Sequence[str] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """
        `dvc commit` を実行して、.dvcファイルの変更を記録します。

        Args:
            targets (Optional[Sequence[str]], optional): コミット対象。指定しない場合は全て。
            force (bool, optional): Trueの場合、`--force`フラグを付与します。
        """
        cmd = ["commit"]
        if force:
            cmd.append("--force")
        if targets:
            cmd.extend(targets)
        self._run_command(cmd, stream_output=False)

    def push(
        self,
        targets: Sequence[str] | None = None,
        remote: str | None = None,
    ) -> None:
        """
        `dvc push` を実行して、データをリモートストレージにアップロードします。

        Args:
            targets (Optional[Sequence[str]]): プッシュ対象。Noneの場合は全て。
            remote (Optional[str]): 使用するリモートストレージの名前。
        """
        cmd = ["push"]
        if remote:
            cmd.extend(["-r", remote])
        if targets:
            cmd.extend(targets)
        self._run_command(cmd, stream_output=False)

    def pull(
        self,
        targets: Sequence[str] | None = None,
        remote: str | None = None,
    ) -> None:
        """
        `dvc pull` を実行して、データをリモートストレージからダウンロードします。

        Args:
            targets (Optional[Sequence[str]]): プル対象。Noneの場合は全て。
            remote (Optional[str]): 使用するリモートストレージの名前。
        """
        cmd = ["pull"]
        if remote:
            cmd.extend(["-r", remote])
        if targets:
            cmd.extend(targets)
        self._run_command(cmd, stream_output=False)

    def status(self, *, json_output: bool = False) -> str:
        """
        `dvc status` を実行して、リポジトリの状態を確認します。

        Args:
            json_output (bool): Trueの場合、`--json`フラグを付けてJSON文字列を返す。

        Returns:
            str: `dvc status`の標準出力。
        """
        cmd = ["status"]
        if json_output:
            cmd.append("--json")
        stdout, _ = self._run_command(cmd, stream_output=False)
        return stdout
