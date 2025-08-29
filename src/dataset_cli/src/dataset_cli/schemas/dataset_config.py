from pydantic import BaseModel, Field

from .polars import PolarsDataType


class LocalizedStr(BaseModel):
    """ローカライズされた文字列を表すモデル。"""

    ja: str = Field(
        ...,
        description="日本語の文字列",
    )
    en: str = Field(
        ...,
        description="英語の文字列",
    )


class ColumnConfig(BaseModel):
    """データセットの列設定を表すモデル。"""

    name: str = Field(
        ...,
        title="列名",
        description="英語での列名を指定することを推奨します。",
        examples=[
            "participant_id",
        ],
    )
    description: LocalizedStr | None = Field(
        None,
        title="列の説明",
        examples=[
            {
                "ja": "参加者の一意の識別子",
                "en": "Unique identifier for the participant",
            },
        ],
    )
    type_: PolarsDataType = Field(
        ...,
        title="列のデータ型",
        examples=[
            "String",
            "Int64",
            {"name": "List", "inner": "Int32"},
            {"name": "Array", "inner": "Float32", "width": 10},
            {
                "name": "Struct",
                "fields": [
                    {"name": "id", "dtype": "Int64"},
                    {"name": "name", "dtype": "String"},
                ],
            },
        ],
    )


class DatasetConfig(BaseModel):
    """データセットの設定を表すモデル。"""

    name: LocalizedStr = Field(
        ...,
        title="データセット名",
        examples=[
            {
                "ja": "夏の学校",
                "en": "Summer School",
            },
        ],
    )
    license_: LocalizedStr | None = Field(
        ...,
        title="データセットのライセンス",
        examples=[
            {
                "ja": "CC BY 4.0",
                "en": "CC BY 4.0",
            },
        ],
    )
    description: LocalizedStr | None = Field(
        None,
        title="データセットの説明",
        examples=[
            {
                "ja": "このデータセットは、行動モデル夏の学校のために収集されたデータです。",
                "en": "This dataset contains data collected for the Summer School on Behavioral Models.",
            },
        ],
    )
    year: int = Field(
        ...,
        title="データセットが収集された年",
        examples=[2023],
    )
    city: LocalizedStr | None = Field(
        None,
        title="データセットが収集された都市",
        examples=[
            {
                "ja": "東京",
                "en": "Tokyo",
            },
        ],
    )
    columns: list[ColumnConfig] = Field(
        ...,
        title="データセットの列設定",
        examples=[
            {
                "name": {"ja": "参加者ID", "en": "Participant ID"},
                "description": {
                    "ja": "参加者の一意の識別子",
                    "en": "Unique identifier for the participant",
                },
                "type_": "String",
            },
            {
                "name": {"ja": "年齢", "en": "Age"},
                "type_": {"name": "Int64"},
            },
        ],
    )
    hash_: str | None = Field(
        None,
        title="データセットのハッシュ値",
        description="手動で設定する必要はありません。自動的に計算されます。",
    )


DatasetConfig.model_rebuild()
