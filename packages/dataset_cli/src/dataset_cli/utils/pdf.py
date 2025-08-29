import datetime
from importlib import resources
from pathlib import Path

import jinja2
import typer
from rich import print as pprint
from weasyprint import CSS, HTML # type: ignore[import-untyped]

from dataset_cli.schemas.dataset_config import DatasetConfig
from dataset_cli.utils.parser import parse_yaml_and_validate
from dataset_cli.utils.validate import validate_file_hash


def create_readme_pdf(
    file_path: Path,
    lang: str = "ja",
) -> None:
    """
    YAMLデータに基づいてREADMEのPDFを生成する。
    HTMLテンプレートとCSSは外部ファイルから読み込む。
    """
    # 1. YAMLデータをパース
    config_path = Path(file_path.as_posix() + ".schema.yaml")

    data = parse_yaml_and_validate(Path(config_path), DatasetConfig)
    if data.hash_ is None:
        pprint(
            "[red]スキーマファイルにハッシュが設定されていません。先に `davis data validate-file` を実行してください。[/red]",
        )
        raise typer.Exit(1)

    if not validate_file_hash(
        file_path,
        data.hash_,
    ):
        pprint(
            "[red]ファイルのハッシュがスキーマと一致しません。先に `davis data validate-file` を実行してください。[/red]",
        )
        raise typer.Exit(1)

    output_path = Path(file_path).parent / (file_path.name + f".{lang}.pdf")

    # 2. Jinja2テンプレート環境をセットアップし、外部ファイルを読み込む
    templates_path = resources.files("dataset_cli.templates")
    template_html_content = templates_path.joinpath("readme.html").read_text(
        encoding="utf-8",
    )
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(), autoescape=True,
    )  # S701
    template = env.from_string(template_html_content)

    # 3. CSSも同様に、パスではなく中身を直接読み込む
    css_content = templates_path.joinpath("readme.css").read_text(encoding="utf-8")
    css = CSS(string=css_content)

    now = datetime.datetime.now(
        tz=datetime.timezone(datetime.timedelta(hours=9)),
    )

    # 3. HTMLをレンダリング
    context = {
        "data": data,
        "lang": lang,
        "last_updated": now,
    }
    rendered_html = template.render(context)

    # 4. WeasyPrintでPDFを生成

    HTML(string=rendered_html).write_pdf(output_path, stylesheets=[css])

    pprint(f"[green]README PDFを {output_path} に保存しました。[/green]")
