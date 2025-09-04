import mimetypes
from pathlib import Path

import yaml
from pydantic import BaseModel

from dataset_cli.utils.i18n import _


def parse_yaml_and_validate[T: BaseModel](
    yaml_path: str | Path,
    pydantic_model_class: type[T],
) -> T:
    """
    指定されたYAMLファイルをパースし、指定されたPydanticモデルで検証します。

    Args
    ----
    yaml_path: str | Path
        検証するYAMLファイルのパス。
    pydantic_model_class: type[T]
        検証に使用するPydanticモデルのクラス。TはBaseModelを継承した型でなければならない。

    Returns
    -------
    T: 検証されたPydanticモデルのインスタンス。

    Raises
    ------
    FileNotFoundError: 指定されたYAMLファイルが存在しない場合。
    ValueError: 指定されたパスがYAMLファイルではない場合。
    RuntimeError: YAMLの読み込みに失敗した場合。
    """
    path: Path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path

    if not path.exists():
        msg = _("YAMLファイルが見つかりません: {yaml_path}").format(yaml_path=yaml_path)
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = _("指定されたパスはファイルではありません: {yaml_path}").format(
            yaml_path=yaml_path,
        )
        raise FileNotFoundError(msg)
    if path.suffix.lower() not in (".yaml", ".yml"):
        msg = _("指定されたパスはYAMLファイルではありません: {yaml_path}").format(
            yaml_path=yaml_path,
        )
        raise ValueError(msg)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = _("YAMLの読み込みに失敗しました: {e}").format(e=e)
        raise RuntimeError(msg) from e

    return pydantic_model_class(**data)


def infer_file_type(file_path: str | Path) -> str | None:
    """
    指定されたファイルのタイプを推測します。

    Args
    ----
    file_path: str | Path
        ファイルのパス。

    Returns
    -------
    str: 推測されたファイルタイプ（例: "csv", "json", "yaml"）。
    """  # noqa: RUF002
    file_type, _ = mimetypes.guess_type(file_path)
    return file_type
