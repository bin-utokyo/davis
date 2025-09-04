import typer
from rich import print as rprint

from dataset_cli.utils.config import load_user_config, save_user_config
from dataset_cli.utils.i18n import _

app = typer.Typer(no_args_is_help=True, help=_("設定管理コマンド"))


@app.command("lang", help=_("表示言語を設定します。"))
def set_language(
    lang: str = typer.Argument(..., help=_("設定する言語コード (例: ja, en)")),
) -> None:
    current_config = load_user_config()
    current_config["lang"] = lang
    save_user_config(current_config)
    rprint(
        _(
            "[green]言語設定を '{lang}' に変更しました。次回起動時から適用されます。[/green]",
        ).format(lang=lang),
    )


@app.command("show", help=_("現在の設定を表示します。"))
def show_config() -> None:
    config = load_user_config()
    rprint(_("[bold]現在の設定:[/bold]"))
    for key, value in config.items():
        display_value = (
            f"{value[:4]}...{value[-4:]}"
            if "secret" in key and len(value) > 8  # noqa: PLR2004
            else value
        )
        rprint(f"  - [cyan]{key}[/cyan]: {display_value}")
