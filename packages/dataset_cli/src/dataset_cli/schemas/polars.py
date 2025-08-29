"""Polars のデータ型を Pydantic モデルで定義するモジュール。

このモジュールでは、Polars のシンプルなデータ型と複合データ型を Pydantic モデルとして定義します。

See Also
--------
https://docs.pola.rs/api/python/dev/reference/datatypes.html
"""

from typing import (
    Literal,
)  # Use List for type hinting lists in older Python versions if needed, otherwise list

import polars as pl
from pydantic import BaseModel, Field, ValidationError

# Polarsのシンプルなデータ型をLiteralで列挙
PolarsSimpleDataTypeNameLiteral = Literal[
    "Decimal",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Date",
    "Datetime",
    "Duration",
    "Time",
    "String",
    # "Utf8",  # Alias for String in Polars
    # "Binary",  # Cannot be used in CSV files
    "Boolean",
]


class PolarsSimpleDataTypeLiteral(BaseModel):
    """Polarsのシンプルなデータ型を表すモデル。"""

    name: PolarsSimpleDataTypeNameLiteral = Field(
        ...,
        description="Polarsのシンプルなデータ型名",
        examples=[
            "String",
            "Int64",
            "Float32",
            "Boolean",
        ],
    )


class PolarsEnumDataTypeLiteral(BaseModel):
    """Polarsの列挙型データ型を表すモデル。"""

    name: Literal["Enum"] = Field(
        "Enum",
    )
    options: list[str] = Field(
        ...,
        description="列挙型の要素のリスト",
        examples=[
            ["option1", "option2", "option3"],
        ],
    )


class PolarsListDataTypeLiteral(BaseModel):
    """Polarsのリストデータ型を表すモデル。"""

    name: Literal["List"] = Field(
        "List",
    )
    inner: PolarsSimpleDataTypeLiteral = Field(
        ...,
        description="リストの要素のデータ型",
    )


class PolarsArrayDataTypeLiteral(BaseModel):
    """Polarsの配列データ型を表すモデル。"""

    name: Literal["Array"] = Field(
        "Array",
    )
    inner: PolarsSimpleDataTypeLiteral = Field(
        ...,
        description="配列の要素のデータ型",
    )
    width: int = Field(
        ...,
        description="配列の幅（要素数）",
    )


PolarsDataType = (
    PolarsSimpleDataTypeLiteral
    | PolarsEnumDataTypeLiteral
    | PolarsListDataTypeLiteral
    | PolarsArrayDataTypeLiteral
)


def get_polars_data_type(
    columns: PolarsDataType,
) -> pl.DataType:
    """Polarsのデータ型を使用してPolarsのデータ型オブジェクトを構築する関数。

    Parameters
    ----------
    columns : list[PolarsDataType]
        Polarsのデータ型のリスト。

    Returns
    -------
    pl.DataType
        Polarsのデータ型オブジェクト。

    Raises
    ------
    ValueError
        指定されたPolarsのデータ型がサポートされていない場合。
    """
    if isinstance(columns, PolarsSimpleDataTypeLiteral):
        mapped_type: dict[str, pl.DataType] = {
            "String": pl.Utf8(),
            "Boolean": pl.Boolean(),
            "Decimal": pl.Decimal(),
            "Float32": pl.Float32(),
            "Float64": pl.Float64(),
            "Int8": pl.Int8(),
            "Int16": pl.Int16(),
            "Int32": pl.Int32(),
            "Int64": pl.Int64(),
            "Int128": pl.Int128(),
            "UInt8": pl.UInt8(),
            "UInt16": pl.UInt16(),
            "UInt32": pl.UInt32(),
            "UInt64": pl.UInt64(),
            "Date": pl.Date(),
            "Datetime": pl.Datetime(time_zone="Asia/Tokyo"),
            "Duration": pl.Duration(),
            "Time": pl.Time(),
        }
        return mapped_type[columns.name]

    if isinstance(columns, PolarsEnumDataTypeLiteral):
        return pl.Enum(
            columns.options,
        )
    if isinstance(columns, PolarsListDataTypeLiteral):
        return pl.List(get_polars_data_type(columns.inner))
    if isinstance(columns, PolarsArrayDataTypeLiteral):
        return pl.Array(
            get_polars_data_type(columns.inner),
            width=columns.width,
        )

    msg = f"Unsupported Polars data type: {columns}"
    raise ValueError(msg)


def get_polars_data_type_name(
    type_: pl.DataType,
) -> PolarsDataType:
    """Polarsのデータ型をPydanticモデルで表現する関数。
    Parameters
    ----------
    type_ : pl.DataType
        Polarsのデータ型オブジェクト。
    Returns
    -------
    PolarsDataType
        Polarsのデータ型を表すPydanticモデル。
    Raises
    ------
    ValueError
        指定されたPolarsのデータ型がサポートされていない場合。
    TypeError
        ListやArrayの内部データ型がPolarsのDataTypeでない場合。
    """

    try:
        type_name = type(type_).__name__
        validated_data = PolarsSimpleDataTypeLiteral.model_validate({"name": type_name})
        return PolarsSimpleDataTypeLiteral(name=validated_data.name)
    except ValidationError:
        pass

    # Enum 型
    if isinstance(type_, pl.Enum):
        categories: pl.Series = type_.categories
        return PolarsEnumDataTypeLiteral(name="Enum", options=categories.to_list())

    # List 型
    if isinstance(type_, pl.List):
        raw_inner = type_.inner
        if not isinstance(raw_inner, pl.DataType):
            msg = f"List inner type must be a Polars DataType: {raw_inner}"
            raise TypeError(msg)
        inner = get_polars_data_type_name(raw_inner)
        if not isinstance(inner, PolarsSimpleDataTypeLiteral):
            msg = f"List inner type must be simple: {inner}"
            raise TypeError(msg)
        return PolarsListDataTypeLiteral(name="List", inner=inner)

    # Array 型
    if isinstance(type_, pl.Array):
        raw_inner = type_.inner
        if not isinstance(raw_inner, pl.DataType):
            msg = f"Array inner type must be a Polars DataType: {raw_inner}"
            raise TypeError(msg)
        inner = get_polars_data_type_name(raw_inner)
        if not isinstance(inner, PolarsSimpleDataTypeLiteral):
            msg = f"Array inner type must be simple: {inner}"
            raise TypeError(msg)
        return PolarsArrayDataTypeLiteral(name="Array", inner=inner, width=type_.width)

    msg = f"Unsupported Polars data type: {type_}"
    raise ValueError(msg)
