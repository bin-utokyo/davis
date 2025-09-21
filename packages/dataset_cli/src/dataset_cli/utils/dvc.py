import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import typer
from rich import print as rprint

from dataset_cli.utils.i18n import _

from .config import load_user_config


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
        return f"{super().__str__()}\nReturn Code: {self.return_code}\n--- STDOUT ---\n{self.stdout}\n--- STDERR ---\n{self.stderr}"


class DVCClient:
    """
    DVCコマンドをラップし、Pythonから実行するためのクライアントクラス。

    Attributes
    ----------
        repo_path (Path): DVCリポジトリのルートパス。
    """

    def __init__(self, repo_path: str | Path = ".") -> None:
        """
        DVCClientを初期化します。

        Args:
            repo_path (str | Path, optional): DVCリポジトリのパス。
                                              デフォルトはカレントディレクトリ。
        """
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".dvc").exists():
            rprint(
                _(
                    "[yellow]警告: 指定されたパスに.dvcディレクトリが見つかりません: {repo_path}[/yellow]",
                ).format(repo_path=self.repo_path),
            )
            execute_dvc_init = typer.confirm(
                _("このディレクトリで `dvc init` を実行しますか？"),
                default=True,
            )
            if execute_dvc_init:
                self._run_command(["init"])
                rprint(_("[green]DVCリポジトリを初期化しました。[/green]"))

    def _run_command(  # noqa: C901
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
        config = load_user_config()
        dvc_path_fallback = shutil.which("dvc") or "dvc"
        dvc_executable_path = config.get("dvc_executable_path", dvc_path_fallback)

        full_command = [dvc_executable_path, *command]
        rprint(
            _("[cyan]実行中: {full_command}[/cyan]").format(
                full_command=" ".join(full_command),
            ),
        )

        try:
            # Popenを使用してサブプロセスを開始し、出力をリアルタイムでストリーミング
            process = subprocess.Popen(  # noqa: S603
                full_command,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            # 標準出力を読み取る
            if process.stdout:
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        stdout_lines.append(line)
                        if stream_output:
                            rprint(f"[stdout] {line}")

            # 標準エラーを読み取る
            if process.stderr:
                stderr_output = process.stderr.read()
                if stderr_output:
                    stderr_lines = stderr_output.strip().splitlines()
                    if stream_output:
                        for line in stderr_lines:
                            rprint(f"[stderr][red]{line}[/red]")

            return_code = process.wait()
            stdout = "\n".join(stdout_lines)
            stderr = "\n".join(stderr_lines)

            if return_code != 0:
                msg = _(
                    "DVCコマンド '{full_command}' の実行に失敗しました。",
                ).format(full_command=" ".join(full_command))
                raise DVCError(  # noqa: TRY301
                    msg,
                    return_code,
                    stdout,
                    stderr,
                )

        except FileNotFoundError as e:
            rprint(
                _(
                    "[red]エラー: 'dvc'コマンドが見つかりませんでした。DVCはインストールされていますか？[/red]",
                ),
            )
            raise typer.Exit(code=127) from e
        except DVCError as e:
            # DVCErrorを再度送出して、呼び出し元でキャッチできるようにする
            msg = _("DVCコマンドの実行に失敗しました: {e}").format(e=e)
            raise DVCError(
                msg,
                e.return_code,
                e.stdout,
                e.stderr,
            ) from e

        return stdout, stderr

    def add(self, targets: str | Sequence[str]) -> None:
        """
        `dvc add` を実行して、ファイルやディレクトリをDVCの管理下に置きます。

        Args:
            targets (str | Sequence[str]): 対象のファイルまたはディレクトリ。
        """
        target_list = [targets] if isinstance(targets, str) else list(targets)
        self._run_command(["add", *target_list])

    def commit(
        self,
        targets: str | Sequence[str] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """
        `dvc commit` を実行して、.dvcファイルの変更を記録します。

        Args:
            targets (Optional[str | Sequence[str]], optional): コミット対象。指定しない場合は全て。
            force (bool, optional): Trueの場合、`--force`フラグを付与します。
        """
        cmd = ["commit"]
        if force:
            cmd.append("--force")
        if targets:
            target_list = [targets] if isinstance(targets, str) else list(targets)
            cmd.extend(target_list)
        self._run_command(cmd)

    def push(
        self,
        targets: str | Sequence[str] | None = None,
        remote: str | None = None,
    ) -> None:
        """
        `dvc push` を実行して、データをリモートストレージにアップロードします。

        Args:
            targets (Optional[str | Sequence[str]], optional): プッシュ対象。指定しない場合は全て。
            remote (Optional[str], optional): 使用するリモートストレージの名前。
        """
        cmd = ["push"]
        if remote:
            cmd.extend(["-r", remote])
        if targets:
            target_list = [targets] if isinstance(targets, str) else list(targets)
            cmd.extend(target_list)
        self._run_command(cmd)

    def pull(
        self,
        targets: str | Sequence[str] | None = None,
        remote: str | None = None,
        jobs: int | None = None,
        *,
        force: bool = False,
    ) -> None:
        """
        `dvc pull` を実行して、データをリモートストレージからダウンロードします。

        Args:
            targets (Optional[str | Sequence[str]], optional): プル対象。指定しない場合は全て。
            remote (Optional[str], optional): 使用するリモートストレージの名前。
            jobs (Optional[int], optional): 並列ダウンロードの数。
            force (bool, optional): Trueの場合、`--force`フラグを付与します。
        """
        cmd = ["pull"]
        if force:
            cmd.append("--force")
        if remote:
            cmd.extend(["-r", remote])
        if jobs is not None and jobs > 1:
            cmd.extend(["-j", str(jobs)])
        if targets:
            target_list = [targets] if isinstance(targets, str) else list(targets)
            cmd.extend(target_list)
        self._run_command(cmd)

    def status(self, *args: str) -> str:
        """
        `dvc status` を実行して、リポジトリの状態を確認します。

        Args:
            *args: `dvc status` に渡す追加の引数。

        Returns:
            str: `dvc status`の標準出力。
        """
        cmd = ["status", *args]
        stdout, _ = self._run_command(cmd, stream_output=False)
        return stdout

    def remote_modify(
        self,
        remote: str,
        option: str,
        value: str,
        *,
        local: bool = False,
    ) -> None:
        """
        `dvc remote modify` を実行して、リモートの設定を変更します。

        Args:
            remote (str): 変更するリモートの名前。
            option (str): 変更するオプション。
            value (str): 設定する値。
            local (bool, optional): Trueの場合、`--local`フラグを付与します。
        """
        cmd = ["remote", "modify"]
        if local:
            cmd.append("--local")
        cmd.extend([remote, option, value])
        self._run_command(cmd, stream_output=False)
