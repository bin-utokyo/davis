from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


@dataclass
class PreparedData:
    frame: pd.DataFrame
    case_column: str
    alternative_column: str
    chosen_column: str
    available: np.ndarray
    chosen: np.ndarray
    weights: np.ndarray
    design: np.ndarray
    parameter_names: list[str]
    groups: list[np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    output_directory = Path(request["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        prepared = prepare(request)
        operation = request["operation"]
        if operation == "validate":
            result = validation_result(request, prepared, output_directory)
        elif operation == "estimate":
            result = estimate(request, prepared, output_directory)
        else:
            raise ValueError(f"unsupported operation: {operation}")
        write_json(output_directory / "run-result.json", result)
    except Exception as error:
        failed = {
            "api_version": "davis.result/v1alpha1",
            "run_id": request.get("run_id", "unknown"),
            "status": "failed",
            "artifacts": {},
            "extensions": {},
            "error": {
                "code": "DAVIS_MNL_FAILED",
                "message": str(error),
            },
        }
        write_json(output_directory / "run-result.json", failed)
        raise


def prepare(request: dict[str, Any]) -> PreparedData:
    choice_input = request["inputs"]["choice_data"]
    source = choice_input["resolved"]
    path = Path(source["path"])
    if source["media_type"] == "text/csv" or path.suffix.lower() == ".csv":
        frame = read_csv(path, choice_input["source"].get("read"))
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported choice-data format: {path}")

    config = request["config"]
    roles = config["roles"]
    required_roles = ["case_id", "alternative_id", "chosen"]
    for role in required_roles:
        if role not in roles:
            raise ValueError(f"missing role: {role}")
    required_columns = {roles[role] for role in required_roles}
    required_columns.update(
        term["column"] for term in config["terms"] if "column" in term
    )
    if "available" in roles:
        required_columns.add(roles["available"])
    if "weight" in roles:
        required_columns.add(roles["weight"])
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"columns were not found: {', '.join(missing)}")

    case_column = roles["case_id"]
    alternative_column = roles["alternative_id"]
    chosen_column = roles["chosen"]
    if frame.duplicated([case_column, alternative_column]).any():
        raise ValueError("(case_id, alternative_id) must be unique")

    chosen = boolean_array(frame[chosen_column], chosen_column)
    available = (
        boolean_array(frame[roles["available"]], roles["available"])
        if "available" in roles
        else np.ones(len(frame), dtype=bool)
    )
    weights = (
        numeric_array(frame[roles["weight"]], roles["weight"])
        if "weight" in roles
        else np.ones(len(frame), dtype=float)
    )
    if np.any(weights < 0):
        raise ValueError("weights must not be negative")
    if np.any(chosen & ~available):
        raise ValueError("a chosen alternative is unavailable")

    parameter_names: list[str] = []
    for term in config["terms"]:
        name = term["parameter"]
        if name not in parameter_names:
            parameter_names.append(name)
    if not parameter_names:
        raise ValueError("at least one parameter is required")
    parameter_index = {name: index for index, name in enumerate(parameter_names)}
    design = np.zeros((len(frame), len(parameter_names)), dtype=float)
    alternatives = frame[alternative_column].astype(str).to_numpy()
    for term in config["terms"]:
        if ("column" in term) == ("constant" in term):
            raise ValueError("each term must contain exactly one of column or constant")
        values = (
            numeric_array(frame[term["column"]], term["column"])
            if "column" in term
            else np.full(len(frame), float(term["constant"]))
        )
        if term.get("alternatives"):
            allowed = {str(value) for value in term["alternatives"]}
            values = np.where(np.isin(alternatives, list(allowed)), values, 0.0)
        design[:, parameter_index[term["parameter"]]] += values

    groups = [indices for indices in frame.groupby(case_column, sort=False).indices.values()]
    for indices in groups:
        if int(np.sum(chosen[indices])) != 1:
            case = frame.iloc[int(indices[0])][case_column]
            raise ValueError(f"case {case!r} must have exactly one chosen alternative")
        if not np.any(available[indices]):
            case = frame.iloc[int(indices[0])][case_column]
            raise ValueError(f"case {case!r} has no available alternative")
        if not np.allclose(weights[indices], weights[indices][0]):
            case = frame.iloc[int(indices[0])][case_column]
            raise ValueError(f"case {case!r} has inconsistent weights")

    return PreparedData(
        frame=frame,
        case_column=case_column,
        alternative_column=alternative_column,
        chosen_column=chosen_column,
        available=available,
        chosen=chosen,
        weights=weights,
        design=design,
        parameter_names=parameter_names,
        groups=groups,
    )


def read_csv(path: Path, options: dict[str, Any] | None) -> pd.DataFrame:
    options = options or {}
    requested_encoding = (options.get("encoding") or "auto").lower()
    requested_delimiter = options.get("delimiter") or "auto"
    encodings = (
        [requested_encoding]
        if requested_encoding != "auto"
        else ["utf-8-sig", "utf-8", "cp932"]
    )
    null_values = options.get("null_values") or ["", "NA", "N/A", "null", "NULL"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            arguments: dict[str, Any] = {
                "encoding": encoding,
                "dtype": str,
                "keep_default_na": False,
                "na_values": null_values,
            }
            if requested_delimiter == "auto":
                arguments.update({"sep": None, "engine": "python"})
            else:
                arguments["sep"] = "\t" if requested_delimiter == "\\t" else requested_delimiter
            return pd.read_csv(path, **arguments)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"could not decode CSV {path}: {last_error}")


def validation_result(
    request: dict[str, Any], prepared: PreparedData, output_directory: Path
) -> dict[str, Any]:
    summary = sample_summary(prepared)
    write_json(output_directory / "sample-summary.json", summary)
    return success_result(
        request,
        {
            "sample_summary": descriptor("sample-summary.json", "application/json"),
        },
    )


def estimate(
    request: dict[str, Any], prepared: PreparedData, output_directory: Path
) -> dict[str, Any]:
    config = request["config"]
    parameter_config = config.get("parameters", {})
    initial = np.array(
        [float(parameter_config.get(name, {}).get("initial", 0.0)) for name in prepared.parameter_names]
    )
    estimation = config.get("estimation", {})
    method = {
        "bfgs": "BFGS",
        "l-bfgs-b": "L-BFGS-B",
    }.get(estimation.get("optimizer", "bfgs"), "BFGS")
    options = {"maxiter": int(estimation.get("max_iterations", 500))}
    result = minimize(
        lambda parameters: negative_log_likelihood(parameters, prepared),
        initial,
        method=method,
        tol=float(estimation.get("tolerance", 1.0e-8)),
        options=options,
    )
    estimates = np.asarray(result.x, dtype=float)
    covariance = covariance_matrix(result, len(estimates))
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    statistics = np.divide(
        estimates,
        standard_errors,
        out=np.full_like(estimates, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2 * norm.sf(np.abs(statistics))
    parameters = pd.DataFrame(
        {
            "name": prepared.parameter_names,
            "estimate": estimates,
            "std_error": standard_errors,
            "statistic": statistics,
            "p_value": p_values,
        }
    )
    parameters.to_csv(output_directory / "parameters.csv", index=False)
    pd.DataFrame(
        covariance,
        index=prepared.parameter_names,
        columns=prepared.parameter_names,
    ).rename_axis("parameter").to_csv(output_directory / "covariance.csv")

    probabilities = predict_probabilities(estimates, prepared)
    predictions = prepared.frame[
        [prepared.case_column, prepared.alternative_column, prepared.chosen_column]
    ].copy()
    predictions["probability"] = probabilities
    predictions.to_csv(output_directory / "predictions.csv", index=False)

    final_ll = -negative_log_likelihood(estimates, prepared)
    null_ll = null_log_likelihood(prepared)
    parameter_count = len(estimates)
    case_count = len(prepared.groups)
    metrics = {
        "n_cases": case_count,
        "n_rows": len(prepared.frame),
        "n_parameters": parameter_count,
        "log_likelihood_null": null_ll,
        "log_likelihood_final": final_ll,
        "rho_squared": 1 - final_ll / null_ll,
        "adjusted_rho_squared": 1 - (final_ll - parameter_count) / null_ll,
        "aic": -2 * final_ll + 2 * parameter_count,
        "bic": -2 * final_ll + math.log(case_count) * parameter_count,
        "converged": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
    }
    write_json(output_directory / "metrics.json", metrics)
    write_json(output_directory / "sample-summary.json", sample_summary(prepared))
    return success_result(
        request,
        {
            "parameters": descriptor("parameters.csv", "text/csv"),
            "covariance": descriptor("covariance.csv", "text/csv"),
            "metrics": descriptor("metrics.json", "application/json"),
            "predictions": descriptor("predictions.csv", "text/csv"),
            "sample_summary": descriptor("sample-summary.json", "application/json"),
        },
    )


def negative_log_likelihood(parameters: np.ndarray, data: PreparedData) -> float:
    utilities = data.design @ parameters
    total = 0.0
    for indices in data.groups:
        available = data.available[indices]
        available_utilities = utilities[indices][available]
        chosen_position = int(np.flatnonzero(data.chosen[indices])[0])
        chosen_utility = utilities[int(indices[chosen_position])]
        weight = float(data.weights[int(indices[0])])
        total += weight * (chosen_utility - logsumexp(available_utilities))
    return -total


def null_log_likelihood(data: PreparedData) -> float:
    total = 0.0
    for indices in data.groups:
        available_count = int(np.sum(data.available[indices]))
        total -= float(data.weights[int(indices[0])]) * math.log(available_count)
    return total


def predict_probabilities(parameters: np.ndarray, data: PreparedData) -> np.ndarray:
    utilities = data.design @ parameters
    probabilities = np.zeros(len(data.frame), dtype=float)
    for indices in data.groups:
        available = data.available[indices]
        values = utilities[indices][available]
        values = np.exp(values - logsumexp(values))
        probabilities[np.asarray(indices)[available]] = values
    return probabilities


def covariance_matrix(result: Any, size: int) -> np.ndarray:
    inverse = getattr(result, "hess_inv", None)
    if inverse is None:
        return np.full((size, size), np.nan)
    if hasattr(inverse, "todense"):
        inverse = inverse.todense()
    array = np.asarray(inverse, dtype=float)
    return array if array.shape == (size, size) else np.full((size, size), np.nan)


def boolean_array(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"1": True, "0": False, "true": True, "false": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"column {name!r} contains non-boolean values: {invalid[:5]}")
    return normalized.map(mapping).to_numpy(dtype=bool)


def numeric_array(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {name!r} contains missing or non-finite numeric values")
    return values


def sample_summary(data: PreparedData) -> dict[str, Any]:
    alternatives = data.frame[data.alternative_column].astype(str)
    return {
        "n_cases": len(data.groups),
        "n_rows": len(data.frame),
        "alternatives": sorted(alternatives.unique().tolist()),
        "chosen_counts": {
            str(key): int(value)
            for key, value in alternatives[data.chosen].value_counts().items()
        },
        "unavailable_rows": int(np.sum(~data.available)),
    }


def descriptor(path: str, media_type: str) -> dict[str, Any]:
    return {"path": path, "media_type": media_type}


def success_result(
    request: dict[str, Any], artifacts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "api_version": "davis.result/v1alpha1",
        "run_id": request["run_id"],
        "status": "succeeded",
        "artifacts": artifacts,
        "extensions": {},
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
