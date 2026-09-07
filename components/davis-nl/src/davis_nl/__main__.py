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
    groups: list[np.ndarray]
    available: np.ndarray
    chosen: np.ndarray
    design: np.ndarray
    utility_parameters: list[str]
    alternative_column: str
    case_column: str
    chosen_column: str
    nest_names: list[str]
    nest_for_row: np.ndarray
    fixed_upper_scale_mus: dict[int, float]
    estimated_scale_nests: list[int]
    initial_upper_scale_mus: list[float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    request = json.loads(parser.parse_args().request.read_text(encoding="utf-8"))
    output = Path(request["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        data = prepare(request)
        if request["operation"] == "validate":
            write_json(output / "sample-summary.json", summary(data))
            artifacts = {
                "sample_summary": artifact("sample-summary.json", "application/json")
            }
        elif request["operation"] == "estimate":
            artifacts = estimate(request, data, output)
        else:
            raise ValueError(f"unsupported operation: {request['operation']}")
        write_json(output / "run-result.json", result(request, artifacts))
    except Exception as error:
        write_json(
            output / "run-result.json",
            {
                "api_version": "davis.result/v1alpha1",
                "run_id": request.get("run_id", "unknown"),
                "status": "failed",
                "artifacts": {},
                "extensions": {},
                "error": {"code": "DAVIS_NL_FAILED", "message": str(error)},
            },
        )
        raise


def prepare(request: dict[str, Any]) -> PreparedData:
    source = request["inputs"]["choice_data"]["resolved"]
    path = Path(source["path"])
    frame = (
        pd.read_csv(path, dtype=str, keep_default_na=False)
        if path.suffix.lower() == ".csv"
        else pd.read_parquet(path)
    )
    config = request["config"]
    roles = config["roles"]
    case_column, alternative_column, chosen_column = (
        roles[key] for key in ("case_id", "alternative_id", "chosen")
    )
    required = {case_column, alternative_column, chosen_column}
    required.update(term["column"] for term in config["terms"] if "column" in term)
    if "available" in roles:
        required.add(roles["available"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"columns were not found: {', '.join(missing)}")
    if frame.duplicated([case_column, alternative_column]).any():
        raise ValueError("(case_id, alternative_id) must be unique")

    alternatives = frame[alternative_column].astype(str).to_numpy()
    nest_names: list[str] = []
    alternative_to_nest: dict[str, int] = {}
    fixed_upper_scale_mus: dict[int, float] = {}
    estimated_scale_nests: list[int] = []
    initial_upper_scale_mus: list[float] = []
    for nest_index, nest in enumerate(config["nests"]):
        name = nest["name"]
        if name in nest_names:
            raise ValueError(f"duplicate nest name: {name}")
        nest_names.append(name)
        for alternative in map(str, nest["alternatives"]):
            if alternative in alternative_to_nest:
                raise ValueError(
                    f"alternative belongs to multiple nests: {alternative}"
                )
            alternative_to_nest[alternative] = nest_index
        upper_scale_mu = nest.get("scale_mu_d", {})
        if "fixed" in upper_scale_mu:
            fixed_upper_scale_mus[nest_index] = float(upper_scale_mu["fixed"])
        else:
            estimated_scale_nests.append(nest_index)
            initial_upper_scale_mus.append(float(upper_scale_mu.get("initial", 0.8)))
    unknown = sorted(set(alternatives) - set(alternative_to_nest))
    unused = sorted(set(alternative_to_nest) - set(alternatives))
    if unknown or unused:
        raise ValueError(
            f"nests must partition observed alternatives; missing={unknown}, unused={unused}"
        )
    nest_for_row = np.array(
        [alternative_to_nest[value] for value in alternatives], dtype=int
    )

    chosen = boolean_array(frame[chosen_column], chosen_column)
    available = (
        boolean_array(frame[roles["available"]], roles["available"])
        if "available" in roles
        else np.ones(len(frame), dtype=bool)
    )
    if np.any(chosen & ~available):
        raise ValueError("a chosen alternative is unavailable")
    utility_parameters: list[str] = []
    for term in config["terms"]:
        if term["parameter"] not in utility_parameters:
            utility_parameters.append(term["parameter"])
    design = np.zeros((len(frame), len(utility_parameters)))
    parameter_index = {name: index for index, name in enumerate(utility_parameters)}
    for term in config["terms"]:
        if ("column" in term) == ("constant" in term):
            raise ValueError("each term must contain exactly one of column or constant")
        values = (
            numeric_array(frame[term["column"]], term["column"])
            if "column" in term
            else np.full(len(frame), float(term["constant"]))
        )
        if term.get("alternatives"):
            values = np.where(
                np.isin(alternatives, list(map(str, term["alternatives"]))), values, 0.0
            )
        design[:, parameter_index[term["parameter"]]] += values
    groups = list(frame.groupby(case_column, sort=False).indices.values())
    for indices in groups:
        if np.sum(chosen[indices]) != 1:
            raise ValueError(
                f"case {frame.iloc[int(indices[0])][case_column]!r} must have exactly one chosen alternative"
            )
        if not np.any(available[indices]):
            raise ValueError(
                f"case {frame.iloc[int(indices[0])][case_column]!r} has no available alternative"
            )
    return PreparedData(
        frame,
        groups,
        available,
        chosen,
        design,
        utility_parameters,
        alternative_column,
        case_column,
        chosen_column,
        nest_names,
        nest_for_row,
        fixed_upper_scale_mus,
        estimated_scale_nests,
        initial_upper_scale_mus,
    )


def probabilities(parameters: np.ndarray, data: PreparedData) -> np.ndarray:
    """Calculate two-level nested-logit probabilities.

    Following the course PDF, the lowest-level scale mu is fixed to one. Each
    upper nest has scale mu_d in (0, 1]. The within-nest probability therefore
    uses utility directly, and mu_d multiplies the inclusive value at the upper
    level.
    """
    utility_count = len(data.utility_parameters)
    utilities = data.design @ parameters[:utility_count]
    upper_scale_mus = dict(data.fixed_upper_scale_mus)
    upper_scale_mus.update(
        {
            nest: float(parameters[utility_count + index])
            for index, nest in enumerate(data.estimated_scale_nests)
        }
    )
    output = np.zeros(len(data.frame))
    for indices in data.groups:
        available_indices = np.asarray(indices)[data.available[indices]]
        log_conditionals: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        nest_values: list[float] = []
        nest_order: list[int] = []
        for nest in np.unique(data.nest_for_row[available_indices]):
            rows = available_indices[data.nest_for_row[available_indices] == nest]
            upper_scale_mu = upper_scale_mus[int(nest)]
            denominator = logsumexp(utilities[rows])
            log_conditionals[int(nest)] = (
                rows,
                utilities[rows] - denominator,
                denominator,
            )
            nest_order.append(int(nest))
            nest_values.append(upper_scale_mu * denominator)
        log_nest_denominator = logsumexp(nest_values)
        for nest, nest_value in zip(nest_order, nest_values, strict=True):
            rows, conditional, _ = log_conditionals[nest]
            output[rows] = np.exp(conditional + nest_value - log_nest_denominator)
    return output


def negative_log_likelihood(parameters: np.ndarray, data: PreparedData) -> float:
    predicted = probabilities(parameters, data)
    chosen_probabilities = predicted[data.chosen]
    if np.any(chosen_probabilities <= 0) or not np.all(
        np.isfinite(chosen_probabilities)
    ):
        return 1.0e100
    return -float(np.sum(np.log(chosen_probabilities)))


def estimate(
    request: dict[str, Any], data: PreparedData, output: Path
) -> dict[str, dict[str, str]]:
    # Utility coefficients are unconstrained. Following the course PDF, the
    # lowest-level scale is fixed to one and upper nest scales satisfy
    # 0 < mu_d <= 1.
    initial = np.concatenate(
        [
            np.zeros(len(data.utility_parameters)),
            np.asarray(data.initial_upper_scale_mus),
        ]
    )
    bounds = [(None, None)] * len(data.utility_parameters) + [(0.05, 1.0)] * len(
        data.initial_upper_scale_mus
    )
    options = request["config"].get("estimation", {})
    fitted = minimize(
        negative_log_likelihood,
        initial,
        args=(data,),
        method="L-BFGS-B",
        bounds=bounds,
        tol=float(options.get("tolerance", 1.0e-8)),
        options={"maxiter": int(options.get("max_iterations", 500))},
    )
    covariance = covariance_matrix(
        lambda parameters: negative_log_likelihood(parameters, data), fitted.x
    )
    boundary = parameters_at_bounds(fitted.x, bounds)
    covariance[boundary, :] = np.nan
    covariance[:, boundary] = np.nan
    if not fitted.success:
        covariance[:] = np.nan
    standard_errors = standard_errors_from_covariance(covariance)
    inference_status = (
        summarize_inference(standard_errors, boundary)
        if fitted.success
        else "unavailable_not_converged"
    )
    t_values = np.divide(
        fitted.x,
        standard_errors,
        out=np.full_like(fitted.x, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2 * norm.sf(np.abs(t_values))
    estimated_scales = {
        nest: float(fitted.x[len(data.utility_parameters) + index])
        for index, nest in enumerate(data.estimated_scale_nests)
    }
    scale_values = {**data.fixed_upper_scale_mus, **estimated_scales}
    names = data.utility_parameters + [
        f"scale_mu_d:{name}" for name in data.nest_names
    ]
    kinds = ["utility"] * len(data.utility_parameters) + ["scale_mu_d"] * len(
        data.nest_names
    )
    estimates = list(fitted.x[: len(data.utility_parameters)]) + [
        scale_values[index] for index in range(len(data.nest_names))
    ]
    fixed = [False] * len(data.utility_parameters) + [
        index in data.fixed_upper_scale_mus for index in range(len(data.nest_names))
    ]
    free_index_by_nest = {
        nest: len(data.utility_parameters) + index
        for index, nest in enumerate(data.estimated_scale_nests)
    }
    output_standard_errors = list(standard_errors[: len(data.utility_parameters)]) + [
        standard_errors[free_index_by_nest[index]]
        if index in free_index_by_nest
        else np.nan
        for index in range(len(data.nest_names))
    ]
    output_t_values = list(t_values[: len(data.utility_parameters)]) + [
        t_values[free_index_by_nest[index]]
        if index in free_index_by_nest
        else np.nan
        for index in range(len(data.nest_names))
    ]
    output_p_values = list(p_values[: len(data.utility_parameters)]) + [
        p_values[free_index_by_nest[index]]
        if index in free_index_by_nest
        else np.nan
        for index in range(len(data.nest_names))
    ]
    pd.DataFrame(
        {
            "name": names,
            "kind": kinds,
            "estimate": estimates,
            "std_error": output_standard_errors,
            "t_value": output_t_values,
            "p_value": output_p_values,
            "significance": [significance_mark(value) for value in output_p_values],
            "fixed": fixed,
        }
    ).to_csv(output / "parameters.csv", index=False)
    free_names = data.utility_parameters + [
        f"scale_mu_d:{data.nest_names[nest]}" for nest in data.estimated_scale_nests
    ]
    pd.DataFrame(covariance, index=free_names, columns=free_names).rename_axis(
        "parameter"
    ).to_csv(output / "covariance.csv")
    predicted = probabilities(fitted.x, data)
    predictions = data.frame[
        [data.case_column, data.alternative_column, data.chosen_column]
    ].copy()
    predictions["nest"] = [data.nest_names[index] for index in data.nest_for_row]
    predictions["probability"] = predicted
    predictions.to_csv(output / "predictions.csv", index=False)
    final_ll = -negative_log_likelihood(fitted.x, data)
    null_ll = null_log_likelihood(data)
    parameter_count = len(fitted.x)
    rho_squared, adjusted_rho_squared = likelihood_ratios(
        final_ll, null_ll, parameter_count
    )
    metrics = {
        "n_cases": len(data.groups),
        "n_rows": len(data.frame),
        "n_parameters": parameter_count,
        "null_model": "uniform_available_alternatives",
        "inference_distribution": "asymptotic_normal",
        "inference_status": inference_status,
        "log_likelihood_null": null_ll,
        "log_likelihood_final": final_ll,
        "rho_squared": rho_squared,
        "adjusted_rho_squared": adjusted_rho_squared,
        "aic": -2 * final_ll + 2 * parameter_count,
        "bic": -2 * final_ll + math.log(len(data.groups)) * parameter_count,
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "message": str(fitted.message),
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "sample-summary.json", summary(data))
    return {
        "parameters": artifact("parameters.csv", "text/csv"),
        "covariance": artifact("covariance.csv", "text/csv"),
        "metrics": artifact("metrics.json", "application/json"),
        "predictions": artifact("predictions.csv", "text/csv"),
        "sample_summary": artifact("sample-summary.json", "application/json"),
    }


def null_log_likelihood(data: PreparedData) -> float:
    return -sum(math.log(int(np.sum(data.available[indices]))) for indices in data.groups)


def likelihood_ratios(
    final_ll: float, null_ll: float, parameter_count: int
) -> tuple[float | None, float | None]:
    if null_ll == 0:
        return None, None
    return 1 - final_ll / null_ll, 1 - (final_ll - parameter_count) / null_ll


def covariance_matrix(objective: Any, estimates: np.ndarray) -> np.ndarray:
    hessian = finite_difference_hessian(objective, estimates)
    if not np.all(np.isfinite(hessian)):
        return np.full(hessian.shape, np.nan)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues <= 0) or np.linalg.cond(hessian) > 1.0e12:
        return np.full(hessian.shape, np.nan)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return np.full(hessian.shape, np.nan)
    return (covariance + covariance.T) / 2


def finite_difference_hessian(objective: Any, point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    size = len(point)
    steps = 1.0e-4 * np.maximum(1.0, np.abs(point))
    hessian = np.empty((size, size), dtype=float)
    base = float(objective(point))
    if not np.isfinite(base) or base >= 1.0e99:
        return np.full((size, size), np.nan)
    for first in range(size):
        delta_first = np.zeros(size)
        delta_first[first] = steps[first]
        forward = float(objective(point + delta_first))
        backward = float(objective(point - delta_first))
        hessian[first, first] = (
            forward - 2.0 * base + backward
        ) / steps[first] ** 2
        for second in range(first):
            delta_second = np.zeros(size)
            delta_second[second] = steps[second]
            mixed = (
                float(objective(point + delta_first + delta_second))
                - float(objective(point + delta_first - delta_second))
                - float(objective(point - delta_first + delta_second))
                + float(objective(point - delta_first - delta_second))
            ) / (4.0 * steps[first] * steps[second])
            hessian[first, second] = mixed
            hessian[second, first] = mixed
    return hessian


def standard_errors_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    standard_errors = np.full(len(diagonal), np.nan)
    valid = np.isfinite(diagonal) & (diagonal > 0)
    standard_errors[valid] = np.sqrt(diagonal[valid])
    return standard_errors


def parameters_at_bounds(
    estimates: np.ndarray, bounds: list[tuple[float | None, float | None]]
) -> np.ndarray:
    result = np.zeros(len(estimates), dtype=bool)
    for index, (estimate, (lower, upper)) in enumerate(
        zip(estimates, bounds, strict=True)
    ):
        tolerance = 1.0e-6 * max(1.0, abs(float(estimate)))
        result[index] = (lower is not None and estimate <= lower + tolerance) or (
            upper is not None and estimate >= upper - tolerance
        )
    return result


def significance_mark(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "†"
    return ""


def summarize_inference(
    standard_errors: np.ndarray, boundary: np.ndarray
) -> str:
    valid = np.isfinite(standard_errors)
    if np.all(valid) and not np.any(boundary):
        return "available"
    if np.any(valid):
        return "partial_boundary_or_singular_information"
    return "unavailable_boundary_or_singular_information"


def boolean_array(series: pd.Series, name: str) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"1": True, "0": False, "true": True, "false": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"column {name!r} contains non-boolean values: {invalid[:5]}")
    return normalized.map(mapping).to_numpy(dtype=bool)


def numeric_array(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"column {name!r} contains missing or non-finite numeric values"
        )
    return values


def summary(data: PreparedData) -> dict[str, Any]:
    alternatives = data.frame[data.alternative_column].astype(str)
    return {
        "n_cases": len(data.groups),
        "n_rows": len(data.frame),
        "alternatives": sorted(map(str, alternatives.unique())),
        "nests": {
            name: sorted(map(str, alternatives[data.nest_for_row == index].unique()))
            for index, name in enumerate(data.nest_names)
        },
    }


def artifact(path: str, media_type: str) -> dict[str, str]:
    return {"path": path, "media_type": media_type}


def result(
    request: dict[str, Any], artifacts: dict[str, dict[str, str]]
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
