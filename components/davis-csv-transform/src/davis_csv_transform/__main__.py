from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    output_directory = Path(request["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        if request["operation"] != "transform":
            raise ValueError(f"unsupported operation: {request['operation']}")
        source = request["inputs"]["table"]
        fieldnames, rows, encoding, delimiter = read_csv(
            Path(source["resolved"]["path"]), source["source"].get("read")
        )
        input_rows = len(rows)
        input_columns = len(fieldnames)
        join_summaries: list[dict[str, Any]] = []
        for join in request["config"].get("joins", []):
            input_name = join["input"]
            if input_name == "table":
                raise ValueError("join input must not be the base table")
            lookup = request["inputs"].get(input_name)
            if lookup is None:
                raise ValueError(f"join input was not provided: {input_name}")
            lookup_fields, lookup_rows, lookup_encoding, lookup_delimiter = read_csv(
                Path(lookup["resolved"]["path"]), lookup["source"].get("read")
            )
            fieldnames, rows, join_summary = join_tables(
                fieldnames, rows, lookup_fields, lookup_rows, join
            )
            join_summary.update(
                {
                    "input": input_name,
                    "source_encoding": lookup_encoding,
                    "source_delimiter": lookup_delimiter,
                }
            )
            join_summaries.append(join_summary)

        output_fields = apply_calculations(fieldnames, rows, request["config"])
        table_path = output_directory / "transformed.csv"
        with table_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=output_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "input_rows": input_rows,
            "input_columns": input_columns,
            "output_rows": len(rows),
            "output_columns": len(output_fields),
            "calculated_columns": [
                calculation["output"]
                for calculation in request["config"].get("calculations", [])
            ],
            "joins": join_summaries,
            "source_encoding": encoding,
            "source_delimiter": delimiter,
        }
        write_json(output_directory / "transformation-summary.json", summary)
        write_json(
            output_directory / "run-result.json",
            {
                "api_version": "davis.result/v1alpha1",
                "run_id": request["run_id"],
                "status": "succeeded",
                "artifacts": {
                    "transformed_table": descriptor("transformed.csv", "text/csv"),
                    "transformation_summary": descriptor(
                        "transformation-summary.json", "application/json"
                    ),
                },
                "extensions": {},
            },
        )
    except Exception as error:
        write_json(
            output_directory / "run-result.json",
            {
                "api_version": "davis.result/v1alpha1",
                "run_id": request.get("run_id", "unknown"),
                "status": "failed",
                "artifacts": {},
                "extensions": {},
                "error": {
                    "code": "DAVIS_CSV_TRANSFORM_FAILED",
                    "message": str(error),
                },
            },
        )
        raise


def read_csv(
    path: Path, options: dict[str, Any] | None
) -> tuple[list[str], list[dict[str, str]], str, str]:
    options = options or {}
    requested_encoding = (options.get("encoding") or "auto").lower()
    encodings = (
        [requested_encoding]
        if requested_encoding != "auto"
        else ["utf-8-sig", "utf-8", "cp932"]
    )
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue
        requested_delimiter = options.get("delimiter") or "auto"
        if requested_delimiter == "auto":
            try:
                delimiter = csv.Sniffer().sniff(text[:8192]).delimiter
            except csv.Error:
                delimiter = ","
        else:
            delimiter = "\t" if requested_delimiter == "\\t" else requested_delimiter
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row")
        return list(reader.fieldnames), list(reader), encoding, delimiter
    raise ValueError(f"could not decode CSV {path}: {last_error}")


def apply_calculations(
    fieldnames: list[str], rows: list[dict[str, str]], config: dict[str, Any]
) -> list[str]:
    output_fields = list(fieldnames)
    for calculation in config.get("calculations", []):
        output = calculation["output"]
        if output in output_fields and not calculation.get("replace", False):
            raise ValueError(f"output column already exists: {output}")
        for term in calculation["terms"]:
            if ("column" in term) == ("constant" in term):
                raise ValueError("each term must contain exactly one of column or constant")
            if "column" in term and term["column"] not in output_fields:
                raise ValueError(f"column was not found: {term['column']}")
        for row_number, row in enumerate(rows, start=2):
            value = 0.0
            for term in calculation["terms"]:
                source = term.get("constant")
                if "column" in term:
                    source = parse_number(row.get(term["column"], ""), term["column"], row_number)
                value += float(term.get("coefficient", 1.0)) * float(source)
            if not math.isfinite(value):
                raise ValueError(f"calculation produced a non-finite value at row {row_number}")
            row[output] = format(value, ".15g")
        if output not in output_fields:
            output_fields.append(output)
    return output_fields


def join_tables(
    left_fields: list[str],
    left_rows: list[dict[str, str]],
    right_fields: list[str],
    right_rows: list[dict[str, str]],
    specification: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    left_on = key_columns(specification["left_on"], "left_on")
    right_on = key_columns(specification["right_on"], "right_on")
    if len(left_on) != len(right_on):
        raise ValueError("left_on and right_on must contain the same number of columns")
    require_columns(left_fields, left_on, "base table")
    require_columns(right_fields, right_on, "join input")

    selected_columns: dict[str, str] = specification["columns"]
    require_columns(right_fields, list(selected_columns.values()), "join input")
    collisions = [column for column in selected_columns if column in left_fields]
    if collisions:
        raise ValueError(f"join output column already exists: {collisions[0]}")

    relationship = specification.get("relationship", "many_to_one")
    how = specification.get("how", "left")
    if relationship not in {"many_to_one", "one_to_one"}:
        raise ValueError(f"unsupported join relationship: {relationship}")
    if how not in {"left", "inner"}:
        raise ValueError(f"unsupported join type: {how}")
    allow_unmatched = specification.get("allow_unmatched", False)
    right_index: dict[tuple[str, ...], dict[str, str]] = {}
    for row_number, row in enumerate(right_rows, start=2):
        key = row_key(row, right_on, "join input", row_number)
        if key in right_index:
            raise ValueError(
                f"join input contains a duplicate key at row {row_number}: {key!r}"
            )
        right_index[key] = row

    seen_left: set[tuple[str, ...]] = set()
    output_rows: list[dict[str, str]] = []
    unmatched = 0
    for row_number, row in enumerate(left_rows, start=2):
        key = row_key(row, left_on, "base table", row_number)
        if relationship == "one_to_one" and key in seen_left:
            raise ValueError(
                f"base table contains a duplicate key at row {row_number}: {key!r}"
            )
        seen_left.add(key)
        matched = right_index.get(key)
        if matched is None:
            unmatched += 1
            if not allow_unmatched:
                raise ValueError(f"join input has no row for base key: {key!r}")
            if how == "inner":
                continue
            joined = dict(row)
            joined.update({output: "" for output in selected_columns})
        else:
            joined = dict(row)
            joined.update(
                {
                    output: matched[source]
                    for output, source in selected_columns.items()
                }
            )
        output_rows.append(joined)

    return (
        [*left_fields, *selected_columns],
        output_rows,
        {
            "how": how,
            "relationship": relationship,
            "base_rows": len(left_rows),
            "lookup_rows": len(right_rows),
            "output_rows": len(output_rows),
            "unmatched_rows": unmatched,
            "left_on": left_on,
            "right_on": right_on,
            "imported_columns": list(selected_columns),
        },
    )


def key_columns(value: str | list[str], name: str) -> list[str]:
    columns = [value] if isinstance(value, str) else value
    if not columns or any(not column for column in columns):
        raise ValueError(f"{name} must contain at least one column")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} must not contain duplicate columns")
    return columns


def require_columns(fieldnames: list[str], columns: list[str], table: str) -> None:
    for column in columns:
        if column not in fieldnames:
            raise ValueError(f"column was not found in {table}: {column}")


def row_key(
    row: dict[str, str], columns: list[str], table: str, row_number: int
) -> tuple[str, ...]:
    key = tuple(row.get(column, "") for column in columns)
    if any(value == "" for value in key):
        raise ValueError(f"{table} contains an empty join key at row {row_number}")
    return key


def parse_number(value: str, column: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"column {column!r} contains a non-numeric value at row {row_number}: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"column {column!r} is non-finite at row {row_number}")
    return number


def descriptor(path: str, media_type: str) -> dict[str, str]:
    return {"path": path, "media_type": media_type}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
