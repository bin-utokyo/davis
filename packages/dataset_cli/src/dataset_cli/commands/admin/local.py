import datetime
import itertools as it
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml
from git import GitCommandError, InvalidGitRepositoryError, Repo
from polars.exceptions import PolarsError
from pydantic import ValidationError
from rich import print as rprint
from rich.progress import track

from dataset_cli.schemas.dataset_config import (
    ColumnConfig,
    DatasetConfig,
    LocalizedStr,
)
from dataset_cli.schemas.polars import get_polars_data_type_name
from dataset_cli.utils.dvc import DVCClient
from dataset_cli.utils.parser import infer_file_type
from dataset_cli.utils.validate import (
    detect_encoding,
    read_data_with_schema,
    validate_paths,
)

app = typer.Typer(no_args_is_help=True, help="データセットのローカル管理コマンド")


DVC_EXCLUDE_SUFFIXES = {".dvc", ".schema.yaml", ".pdf", ".DS_Store", ".gitignore"}
VALIDATION_TARGET_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
PDF_TARGET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _get_and_validate_repo() -> Repo:
    try:
        repo = Repo(search_parent_directories=True)
    except InvalidGitRepositoryError as e:
        rprint("[bold red]エラー: ここはGitリポジトリではありません。[/bold red]")
        raise typer.Exit(code=1) from e
    return repo


def _get_dvc_deleted_files(dvc_client: DVCClient) -> list[str]:
    try:
        result = dvc_client.status("--quiet", "--json")
        if not result:
            return []
        status = json.loads(result)
        return [item["path"] for item in status.get("deleted", [])]
    except json.JSONDecodeError:
        return []


def _is_excluded(path: Path) -> bool:
    return any(str(path).endswith(suffix) for suffix in DVC_EXCLUDE_SUFFIXES)


def _detect_changes(repo: Repo, repo_path: Path, dvc_client: DVCClient) -> list[Path]:
    rprint("\n[bold]Step 1: データセットの変更を検出します...[/bold]")
    deleted_files = _get_dvc_deleted_files(dvc_client)
    if deleted_files:
        rprint(f"  [red]削除を検出 (DVC):[/red] {', '.join(deleted_files)}")
        dvc_files_to_remove = [f"{f}.dvc" for f in deleted_files]
        if dvc_files_to_remove:
            repo.index.remove(dvc_files_to_remove, working_tree=True, r=True)
            rprint(
                f"  [red]対応する.dvcファイルを削除:[/red] {', '.join(dvc_files_to_remove)}",
            )

    data_dir = repo_path / "data"
    all_data_files = (
        [
            p.relative_to(repo_path)
            for p in data_dir.rglob("*")
            if p.is_file() and not _is_excluded(p)
        ]
        if data_dir.is_dir()
        else []
    )

    if not all_data_files:
        rprint(
            "  [dim]data/ ディレクトリ内にDVCで管理する対象ファイルが見つかりません。[/dim]",
        )
    else:
        rprint(
            f"  [cyan]DVC管理候補 (スキャン):[/cyan] {len(all_data_files)}個のファイル",
        )
    return all_data_files


def _dvc_add_and_validate(  # noqa: C901
    repo_path: Path,
    files: list[Path],
    dvc_client: DVCClient,
) -> None:
    rprint("\n[bold]Step 2: DVCへの追加とスキーマ検証を開始します...[/bold]")
    if not files:
        return

    files_to_add_str = [str(p) for p in files]
    dvc_client.add(files_to_add_str)
    rprint("  [cyan]✓[/cyan] `dvc add` を実行し、DVCの追跡情報を更新しました。")

    files_for_validation_check = [
        f for f in files if f.suffix in VALIDATION_TARGET_EXTENSIONS
    ]
    validation_targets: set[Path] = set()
    processed_dirs: set[Path] = set()

    for file_path_rel in files_for_validation_check:
        file_path = repo_path / file_path_rel
        file_schema = file_path.with_suffix(file_path.suffix + ".schema.yaml")
        if file_schema.exists():
            validation_targets.add(file_path_rel)
            continue
        dir_path = file_path.parent
        if dir_path in processed_dirs:
            continue
        dir_schema = dir_path / "schema.yaml"
        if dir_schema.exists():
            validation_targets.add(dir_path.relative_to(repo_path))
            processed_dirs.add(dir_path)

    if not validation_targets:
        return

    rprint(f"  [cyan]検証ターゲット:[/cyan] {[str(p) for p in validation_targets]}")
    error_occurred: list[Path] = []
    for target_path in validation_targets:
        try:
            validate((repo_path / target_path).as_posix())
        except typer.Exit:
            error_occurred.append(target_path)
    if error_occurred:
        rprint(
            "\n[bold red]エラー: 1つ以上のファイルが検証に失敗しました。同期を中断します。[/bold red]",
        )
        for p in error_occurred:
            rprint(f"  [red]✗ 検証失敗:[/red] {p}")
        raise typer.Exit(code=1)
    rprint("[green]✅ 検証対象はすべて検証を通過しました。[/green]")


def _generate_pdfs_and_stage(repo: Repo) -> None:
    rprint("\n[bold]Step 3: PDF生成と最終ステージングを行います...[/bold]")
    generate_pdf()
    repo.git.add(all=True)
    rprint("  [green]✓[/green] 全ての変更をGitにステージングしました。")


def _commit_and_push(
    repo: Repo,
    message: str,
    dvc_client: DVCClient,
) -> None:
    rprint("\n[bold]Step 4: コミットとプッシュを行います...[/bold]")
    if not repo.index.diff(repo.head.commit):
        rprint(
            "\n[green]✅ コミットすべき変更はありませんでした。処理を終了します。[/green]",
        )
        return

    final_commit_message = message or typer.prompt(
        "コミットメッセージを入力してください",
        default="Update dataset",
    )
    repo.index.commit(final_commit_message)
    rprint(f"[green]✅ Gitにコミットしました: '{final_commit_message}'[/green]")

    if typer.confirm("\nDVC と Git の変更をリモートにプッシュしますか？", default=True):
        dvc_client.push()
        current_branch = repo.active_branch
        if current_branch.tracking_branch() is None:
            rprint(
                "  [dim]上流ブランチが設定されていません。'--set-upstream'をつけてプッシュします...[/dim]",
            )
            repo.git.push("--set-upstream", "origin", current_branch.name)
        else:
            repo.remotes.origin.push()
        rprint("[green]🚀 ✅ DVCとGitのリモート同期が完了しました！[/green]")
    else:
        rprint("[yellow]プッシュはスキップされました。[/yellow]")


@app.command(
    name="sync",
    help="データセットの変更を検証し、DVCとGitにコミット・プッシュします。",
)
def sync_dataset(
    commit_message: Annotated[
        str,
        typer.Option("--message", "-m", help="コミットメッセージ。"),
    ] = "",
) -> None:
    repo = _get_and_validate_repo()
    repo_path = Path(repo.working_dir)
    dvc_client = DVCClient(repo_path=repo_path)
    try:
        changed_files = _detect_changes(repo, repo_path, dvc_client)
        _dvc_add_and_validate(repo_path, changed_files, dvc_client)
        _generate_pdfs_and_stage(repo)
        _commit_and_push(repo, commit_message, dvc_client)
    except (
        GitCommandError,
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
    ) as e:
        rprint(f"[bold red]Git/DVCコマンドエラー: {e}[/bold red]")


# --- validate command --- #
@app.command(
    name="validate",
    help="指定されたファイルまたはディレクトリをスキーマに照らして検証します。",
)
def validate(
    file_or_dir_path: Annotated[
        str,
        typer.Argument(help="検証するファイルまたはディレクトリのパス"),
    ],
    schema_path_str: Annotated[
        str | None,
        typer.Option("--schema", "-s", help="検証に使用するスキーマファイルのパス"),
    ] = None,
    encoding: Annotated[
        str | None,
        typer.Option(help="ファイルエンコーディング（例: utf-8, cp932 など）"),
    ] = None,
) -> None:
    path_obj = Path(file_or_dir_path)
    if not path_obj.exists():
        rprint(f"[red]指定されたパスが存在しません: {file_or_dir_path}[/red]")
        raise typer.Exit(code=1)

    if path_obj.is_file():
        _validate_file(file_or_dir_path, schema_path_str, encoding)
    elif path_obj.is_dir():
        _validate_directory(file_or_dir_path, schema_path_str, encoding)
    else:
        rprint(f"[red]不明なパスのタイプです: {file_or_dir_path}[/red]")
        raise typer.Exit(code=1)


def _validate_file(
    file_path: str,
    schema_path_str: str | None,
    encoding: str | None,
) -> None:
    schema_path = (
        Path(schema_path_str) if schema_path_str else Path(file_path + ".schema.yaml")
    )
    validate_paths(file_path, schema_path)
    read_data_with_schema(file_path, schema_path, encoding=encoding)
    rprint("[green]ファイルはスキーマに適合しています。[/green]")


def _validate_directory(
    dir_path: str,
    schema_path_str: str | None,
    encoding: str | None,
) -> None:
    dir_path_obj = Path(dir_path)
    file_paths = list(
        it.chain.from_iterable(
            dir_path_obj.glob(ext) for ext in VALIDATION_TARGET_EXTENSIONS
        ),
    )

    if not file_paths:
        rprint("[yellow]検証対象のファイルが見つかりませんでした。[/yellow]")
        return

    schema_path = (
        Path(schema_path_str) if schema_path_str else dir_path_obj / "schema.yaml"
    )
    if not schema_path.exists():
        rprint(f"[red]スキーマファイルが見つかりません: {schema_path}[/red]")
        raise typer.Exit(code=1)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                read_data_with_schema,
                f.as_posix(),
                schema_path,
                encoding,
                write_hash=False,
                quiet=True,
            ): f
            for f in file_paths
        }
        for future in track(
            as_completed(futures),
            total=len(futures),
            description="検証中",
        ):
            try:
                future.result()
            except (ValidationError, typer.Exit) as e:
                rprint(f"[red]エラー: {e}[/red]")


# --- infer-schema command --- #
@app.command(
    name="infer-schema",
    help="データファイルからスキーマ定義(.schema.yaml)を対話的に生成します。",
)
def infer_schema(
    file_path: Annotated[str, typer.Argument(help="スキーマを推測するファイルのパス")],
    encoding: Annotated[
        str | None,
        typer.Option(help="ファイルエンコーディング（例: utf-8, cp932 など）"),
    ] = None,
) -> None:
    path = Path(file_path)
    if not path.exists():
        rprint(f"[red]ファイルが見つかりません: {file_path}[/red]")
        raise typer.Exit(code=1)

    file_type = infer_file_type(file_path)
    if not file_type:
        rprint("[red]ファイルタイプを推測できませんでした。[/red]")
        raise typer.Exit(code=1)

    enc = encoding or detect_encoding(file_path)

    try:
        if file_type == "text/csv":
            input_dataframe = pl.read_csv(file_path, try_parse_dates=True, encoding=enc)
        elif file_type in (
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ):
            input_dataframe = pl.read_excel(file_path)
        elif file_type == "application/json":
            input_dataframe = pl.read_json(file_path)
        elif file_type == "application/x-parquet":
            input_dataframe = pl.read_parquet(file_path)
        else:
            msg = f"サポートされていないファイルタイプ: {file_type}"
            raise NotImplementedError(msg)

        jst_year = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9)),
        ).year
        schema = DatasetConfig(
            name=LocalizedStr(
                ja=typer.prompt("データセットの名称（日本語）"),
                en=typer.prompt("データセットの名称（英語）"),
            ),
            description=LocalizedStr(
                ja=typer.prompt("データセットの説明（日本語）"),
                en=typer.prompt("データセットの説明（英語）"),
            ),
            license_=LocalizedStr(
                ja=typer.prompt(
                    "データセットのライセンス（日本語）",
                    default=f"{jst_year}年度の行動モデル夏の学校のための利用のみ許可します。",
                ),
                en=typer.prompt(
                    "データセットのライセンス（英語）",
                    default=f"Use is permitted only for the {jst_year} Summer Course on Behavior Modeling in Transportation Networks.",
                ),
            ),
            city=LocalizedStr(
                ja=typer.prompt("データセットの都市（日本語）"),
                en=typer.prompt("データセットの都市（英語）"),
            ),
            year=typer.prompt("データセットの年", type=int),
            columns=[
                ColumnConfig(
                    name=name,
                    type_=get_polars_data_type_name(dtype),
                    description=None,
                )
                for name, dtype in zip(
                    input_dataframe.columns,
                    input_dataframe.schema.values(),
                    strict=True,
                )
            ],
            hash_=None,
        )

        schema_path = path.with_suffix(path.suffix + ".schema.yaml")
        if schema_path.exists() and not typer.confirm(
            "既存のスキーマファイルを上書きしますか？",
            default=False,
        ):
            rprint("[red]スキーマの推測を中止しました。[/red]")
            raise typer.Exit(code=1)

        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with schema_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                schema.model_dump(),
                f,
                indent=4,
                allow_unicode=True,
                sort_keys=False,
            )

        rprint(f"[green]スキーマを {schema_path} に保存しました。[/green]")

    except (ValueError, PolarsError) as e:
        rprint(f"[red]スキーマの推測に失敗しました: {e}[/red]")
        raise typer.Exit(code=1) from e


# --- generate-pdf command --- #
@app.command(
    name="generate-pdf",
    help="指定されたファイル/ディレクトリ、またはデータセット内の全対象ファイルからPDFを生成します。",
)
def generate_pdf(  # noqa: C901, PLR0912
    paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help="PDFを生成するファイルまたはディレクトリのパス。省略した場合は 'data' ディレクトリ内の全対象ファイル。",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
) -> None:
    from dataset_cli.utils.pdf import create_readme_pdf  # noqa: PLC0415

    target_files: list[Path] = []
    if not paths:
        data_dir = Path("data")
        if not data_dir.is_dir():
            rprint("[yellow]警告: 'data' ディレクトリが見つかりません。[/yellow]")
            raise typer.Exit(0)
        target_files = [
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix in PDF_TARGET_EXTENSIONS
        ]
    else:
        for path in paths:
            if path.is_file():
                if path.suffix in PDF_TARGET_EXTENSIONS:
                    target_files.append(path)
                else:
                    rprint(f"[yellow]警告: サポートされていないファイルタイプです: {path}[/yellow]")
            elif path.is_dir():
                target_files.extend(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix in PDF_TARGET_EXTENSIONS
                )

    # 重複を除去し、ソートする
    files_for_pdf = sorted(set(target_files))

    if not files_for_pdf:
        rprint("  [dim]PDF生成対象のファイルが見つかりませんでした。[/dim]")
        return

    rprint(f"  [cyan]対象ファイル数:[/cyan] {len(files_for_pdf)}個")
    success_count = 0
    failure_count = 0

    # Use Path.cwd() to make relative paths in output consistent
    cwd = Path.cwd()

    for file_path in files_for_pdf:
        try:
            for lang in ("ja", "en"):
                create_readme_pdf(file_path=file_path, lang=lang)
            # Use try-except block for robust relative path calculation
            try:
                display_path = file_path.relative_to(cwd)
            except ValueError:
                display_path = file_path
            rprint(f"  [green]✓[/green] 生成完了: {display_path}")
            success_count += 1
        except (typer.Exit, FileNotFoundError, ValidationError) as e:
            try:
                display_path = file_path.relative_to(cwd)
            except ValueError:
                display_path = file_path
            rprint(
                f"[bold red]  ✗[/bold red] 生成失敗: {display_path}",
            )
            rprint(f"    [dim]エラー詳細: {e}[/dim]")
            failure_count += 1

    rprint("\n[bold]完了[/bold]")
    rprint(f"  [green]成功:[/green] {success_count} ファイル ({success_count * 2} PDF)")
    if failure_count > 0:
        rprint(f"  [red]失敗:[/red] {failure_count} ファイル")
