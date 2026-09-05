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
        output_fields = apply_calculations(fieldnames, rows, request["config"])
        table_path = output_directory / "transformed.csv"
        with table_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=output_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "input_rows": len(rows),
            "input_columns": len(fieldnames),
            "output_columns": len(output_fields),
            "calculated_columns": [
                calculation["output"] for calculation in request["config"]["calculations"]
            ],
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
    for calculation in config["calculations"]:
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
