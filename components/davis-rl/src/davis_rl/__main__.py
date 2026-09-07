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
from scipy.stats import norm


@dataclass
class PreparedData:
    network: pd.DataFrame
    observations: pd.DataFrame
    nodes: list[str]
    from_index: np.ndarray
    to_index: np.ndarray
    design: np.ndarray
    parameter_names: list[str]
    observed_link_indices: np.ndarray
    observed_destinations: np.ndarray
    trips: list[np.ndarray]
    network_roles: dict[str, str]
    observation_roles: dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    request = json.loads(parser.parse_args().request.read_text(encoding="utf-8"))
    output = Path(request["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        data = prepare(request)
        if request["operation"] == "validate":
            validate_initial_parameters(request, data)
            write_json(output / "sample-summary.json", sample_summary(data))
            artifacts = {
                "sample_summary": artifact("sample-summary.json", "application/json")
            }
        elif request["operation"] == "estimate":
            artifacts = estimate(request, data, output)
        else:
            raise ValueError(f"unsupported operation: {request['operation']}")
        write_json(output / "run-result.json", success_result(request, artifacts))
    except Exception as error:
        write_json(
            output / "run-result.json",
            {
                "api_version": "davis.result/v1alpha1",
                "run_id": request.get("run_id", "unknown"),
                "status": "failed",
                "artifacts": {},
                "extensions": {},
                "error": {"code": "DAVIS_RL_FAILED", "message": str(error)},
            },
        )
        raise


def prepare(request: dict[str, Any]) -> PreparedData:
    network = read_table(request["inputs"]["network"]["resolved"])
    observations = read_table(request["inputs"]["observations"]["resolved"])
    config = request["config"]
    network_roles = config["network_roles"]
    observation_roles = config["observation_roles"]

    network_columns = {
        network_roles["link_id"],
        network_roles["from_node"],
        network_roles["to_node"],
        *(term["column"] for term in config["terms"]),
    }
    observation_columns = {
        observation_roles["trip_id"],
        observation_roles["step"],
        observation_roles["link_id"],
        observation_roles["destination"],
    }
    require_columns(network, network_columns, "network")
    require_columns(observations, observation_columns, "observations")

    link_column = network_roles["link_id"]
    from_column = network_roles["from_node"]
    to_column = network_roles["to_node"]
    network[link_column] = network[link_column].astype(str)
    network[from_column] = network[from_column].astype(str)
    network[to_column] = network[to_column].astype(str)
    if network[link_column].duplicated().any():
        raise ValueError("network link_id values must be unique")

    nodes = sorted(set(network[from_column]) | set(network[to_column]))
    node_index = {node: index for index, node in enumerate(nodes)}
    from_index = network[from_column].map(node_index).to_numpy(dtype=int)
    to_index = network[to_column].map(node_index).to_numpy(dtype=int)

    parameter_names: list[str] = []
    for term in config["terms"]:
        if term["parameter"] not in parameter_names:
            parameter_names.append(term["parameter"])
    parameter_index = {name: index for index, name in enumerate(parameter_names)}
    design = np.zeros((len(network), len(parameter_names)), dtype=float)
    for term in config["terms"]:
        values = numeric_array(network[term["column"]], term["column"])
        design[:, parameter_index[term["parameter"]]] += values * float(
            term.get("coefficient", 1.0)
        )

    trip_column = observation_roles["trip_id"]
    step_column = observation_roles["step"]
    observed_link_column = observation_roles["link_id"]
    destination_column = observation_roles["destination"]
    observations[trip_column] = observations[trip_column].astype(str)
    observations[observed_link_column] = observations[observed_link_column].astype(str)
    observations[destination_column] = observations[destination_column].astype(str)
    observations["__davis_step"] = pd.to_numeric(
        observations[step_column], errors="coerce"
    )
    if observations["__davis_step"].isna().any():
        raise ValueError("observation step values must be numeric")
    observations.sort_values([trip_column, "__davis_step"], inplace=True)
    observations.reset_index(drop=True, inplace=True)
    if observations.duplicated([trip_column, "__davis_step"]).any():
        raise ValueError("(trip_id, step) must be unique")

    link_index = {link: index for index, link in enumerate(network[link_column])}
    unknown_links = sorted(set(observations[observed_link_column]) - set(link_index))
    if unknown_links:
        raise ValueError(f"observed links were not found in network: {unknown_links}")
    unknown_destinations = sorted(set(observations[destination_column]) - set(nodes))
    if unknown_destinations:
        raise ValueError(
            f"observed destinations were not found in network: {unknown_destinations}"
        )
    observed_link_indices = (
        observations[observed_link_column].map(link_index).to_numpy()
    )
    observed_destinations = observations[destination_column].to_numpy(dtype=str)
    trips = list(observations.groupby(trip_column, sort=False).indices.values())
    validate_observed_paths(
        observations,
        trips,
        observed_link_indices,
        observed_destinations,
        from_index,
        to_index,
        nodes,
        trip_column,
    )
    return PreparedData(
        network=network,
        observations=observations,
        nodes=nodes,
        from_index=from_index,
        to_index=to_index,
        design=design,
        parameter_names=parameter_names,
        observed_link_indices=observed_link_indices,
        observed_destinations=observed_destinations,
        trips=trips,
        network_roles=network_roles,
        observation_roles=observation_roles,
    )


def read_table(resolved: dict[str, Any]) -> pd.DataFrame:
    path = Path(resolved["path"])
    if resolved.get("media_type") == "text/csv" or path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {path}")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} columns were not found: {', '.join(missing)}")


def validate_observed_paths(
    observations: pd.DataFrame,
    trips: list[np.ndarray],
    observed_links: np.ndarray,
    destinations: np.ndarray,
    from_index: np.ndarray,
    to_index: np.ndarray,
    nodes: list[str],
    trip_column: str,
) -> None:
    for rows in trips:
        trip = observations.iloc[int(rows[0])][trip_column]
        trip_destinations = set(destinations[rows])
        if len(trip_destinations) != 1:
            raise ValueError(f"trip {trip!r} has multiple destinations")
        links = observed_links[rows]
        for previous, current in zip(links, links[1:], strict=False):
            if to_index[previous] != from_index[current]:
                raise ValueError(f"trip {trip!r} contains disconnected links")
        destination = next(iter(trip_destinations))
        destination_index = nodes.index(destination)
        if any(to_index[link] == destination_index for link in links[:-1]):
            raise ValueError(
                f"trip {trip!r} reaches destination {destination!r} before its final link"
            )
        if nodes[to_index[links[-1]]] != destination:
            raise ValueError(
                f"trip {trip!r} does not end at destination {destination!r}"
            )


def solve_value_function(
    parameters: np.ndarray, destination: str, data: PreparedData
) -> np.ndarray:
    """Solve the exponentiated Bellman equation for one destination.

    With scale normalized to one, z_i = exp(V_i) and
    z_i = sum_a exp(v_a) z_j for links a=(i,j), while z_destination=1.
    This linear form avoids enumerating complete paths.
    """
    node_count = len(data.nodes)
    destination_index = data.nodes.index(destination)
    utilities = data.design @ parameters
    if np.any(utilities > 700):
        raise ValueError("link utilities overflowed while solving the value function")
    matrix = np.eye(node_count)
    target = np.zeros(node_count)
    target[destination_index] = 1.0
    for link, (from_node, to_node) in enumerate(
        zip(data.from_index, data.to_index, strict=True)
    ):
        if from_node != destination_index:
            matrix[from_node, to_node] -= math.exp(float(utilities[link]))
    matrix[destination_index, :] = 0.0
    matrix[destination_index, destination_index] = 1.0
    try:
        values = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"value function has no unique solution for destination {destination!r}"
        ) from error
    if not np.all(np.isfinite(values)) or np.any(values < -1.0e-10):
        raise ValueError(
            f"value function is invalid for destination {destination!r}; "
            "check cycles and utility parameter bounds"
        )
    return np.maximum(values, 0.0)


def link_probabilities(
    parameters: np.ndarray, destination: str, data: PreparedData
) -> np.ndarray:
    values = solve_value_function(parameters, destination, data)
    utilities = data.design @ parameters
    probabilities = np.zeros(len(data.network))
    destination_index = data.nodes.index(destination)
    for link, (from_node, to_node) in enumerate(
        zip(data.from_index, data.to_index, strict=True)
    ):
        if from_node != destination_index and values[from_node] > 0:
            probabilities[link] = (
                math.exp(float(utilities[link])) * values[to_node] / values[from_node]
            )
    return probabilities


def negative_log_likelihood(parameters: np.ndarray, data: PreparedData) -> float:
    total = 0.0
    try:
        for destination in sorted(set(data.observed_destinations)):
            probabilities = link_probabilities(parameters, destination, data)
            selected_rows = np.flatnonzero(data.observed_destinations == destination)
            chosen_probabilities = probabilities[
                data.observed_link_indices[selected_rows]
            ]
            if np.any(chosen_probabilities <= 0) or not np.all(
                np.isfinite(chosen_probabilities)
            ):
                return 1.0e100
            total += float(np.sum(np.log(chosen_probabilities)))
    except ValueError:
        return 1.0e100
    return -total


def parameter_setup(
    request: dict[str, Any], data: PreparedData
) -> tuple[np.ndarray, list]:
    declarations = request["config"].get("parameters", {})
    initial = np.array(
        [
            float(declarations.get(name, {}).get("initial", 0.0))
            for name in data.parameter_names
        ]
    )
    bounds = [
        (
            declarations.get(name, {}).get("lower"),
            declarations.get(name, {}).get("upper"),
        )
        for name in data.parameter_names
    ]
    return initial, bounds


def validate_initial_parameters(request: dict[str, Any], data: PreparedData) -> None:
    initial, _ = parameter_setup(request, data)
    value = negative_log_likelihood(initial, data)
    if not np.isfinite(value) or value >= 1.0e99:
        raise ValueError(
            "initial parameters do not produce valid value functions and path probabilities"
        )


def estimate(
    request: dict[str, Any], data: PreparedData, output: Path
) -> dict[str, dict[str, str]]:
    initial, bounds = parameter_setup(request, data)
    validate_initial_parameters(request, data)
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
    if not np.isfinite(fitted.fun) or fitted.fun >= 1.0e99:
        raise ValueError("estimation did not find valid recursive-logit parameters")
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
    pd.DataFrame(
        {
            "name": data.parameter_names,
            "estimate": fitted.x,
            "std_error": standard_errors,
            "t_value": t_values,
            "p_value": p_values,
            "significance": [significance_mark(value) for value in p_values],
        }
    ).to_csv(output / "parameters.csv", index=False)
    pd.DataFrame(
        covariance, index=data.parameter_names, columns=data.parameter_names
    ).rename_axis("parameter").to_csv(output / "covariance.csv")

    predictions = data.observations.copy()
    predictions["from_node"] = [
        data.nodes[data.from_index[link]] for link in data.observed_link_indices
    ]
    predictions["to_node"] = [
        data.nodes[data.to_index[link]] for link in data.observed_link_indices
    ]
    predictions["probability"] = 0.0
    for destination in sorted(set(data.observed_destinations)):
        probabilities = link_probabilities(fitted.x, destination, data)
        rows = np.flatnonzero(data.observed_destinations == destination)
        predictions.loc[rows, "probability"] = probabilities[
            data.observed_link_indices[rows]
        ]
    predictions.drop(columns=["__davis_step"]).to_csv(
        output / "predictions.csv", index=False
    )

    final_ll = -float(fitted.fun)
    null_ll = null_log_likelihood(data)
    parameter_count = len(fitted.x)
    observation_count = len(data.observations)
    trip_count = len(data.trips)
    rho_squared, adjusted_rho_squared = likelihood_ratios(
        final_ll, null_ll, parameter_count
    )
    metrics = {
        "n_trips": trip_count,
        "n_observed_links": observation_count,
        "n_parameters": parameter_count,
        "null_model": "uniform_feasible_outgoing_links",
        "inference_distribution": "asymptotic_normal",
        "inference_status": inference_status,
        "log_likelihood_null": null_ll,
        "log_likelihood_final": final_ll,
        "rho_squared": rho_squared,
        "adjusted_rho_squared": adjusted_rho_squared,
        "aic": -2 * final_ll + 2 * parameter_count,
        "bic": -2 * final_ll + math.log(trip_count) * parameter_count,
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "message": str(fitted.message),
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "sample-summary.json", sample_summary(data))
    return {
        "parameters": artifact("parameters.csv", "text/csv"),
        "covariance": artifact("covariance.csv", "text/csv"),
        "metrics": artifact("metrics.json", "application/json"),
        "predictions": artifact("predictions.csv", "text/csv"),
        "sample_summary": artifact("sample-summary.json", "application/json"),
    }


def null_log_likelihood(data: PreparedData) -> float:
    """Log likelihood for locally uniform, destination-feasible link choices."""
    reachable_by_destination = {
        destination: nodes_reaching_destination(destination, data)
        for destination in sorted(set(data.observed_destinations))
    }
    total = 0.0
    for link, destination in zip(
        data.observed_link_indices, data.observed_destinations, strict=True
    ):
        from_node = data.from_index[link]
        reachable = reachable_by_destination[str(destination)]
        feasible = (data.from_index == from_node) & np.isin(
            data.to_index, list(reachable)
        )
        count = int(np.sum(feasible))
        if count == 0 or not feasible[link]:
            raise ValueError(
                f"observed link is not feasible under null model for destination {destination!r}"
            )
        total -= math.log(count)
    return total


def likelihood_ratios(
    final_ll: float, null_ll: float, parameter_count: int
) -> tuple[float | None, float | None]:
    if null_ll == 0:
        return None, None
    return 1 - final_ll / null_ll, 1 - (final_ll - parameter_count) / null_ll


def nodes_reaching_destination(destination: str, data: PreparedData) -> set[int]:
    destination_index = data.nodes.index(destination)
    reachable = {destination_index}
    changed = True
    while changed:
        changed = False
        for from_node, to_node in zip(data.from_index, data.to_index, strict=True):
            if int(to_node) in reachable and int(from_node) not in reachable:
                reachable.add(int(from_node))
                changed = True
    return reachable


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
        result[index] = (lower is not None and estimate <= float(lower) + tolerance) or (
            upper is not None and estimate >= float(upper) - tolerance
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


def numeric_array(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"column {name!r} contains missing or non-finite numeric values"
        )
    return values


def sample_summary(data: PreparedData) -> dict[str, Any]:
    return {
        "n_nodes": len(data.nodes),
        "n_links": len(data.network),
        "n_trips": len(data.trips),
        "n_observed_links": len(data.observations),
        "destinations": sorted(set(map(str, data.observed_destinations))),
    }


def artifact(path: str, media_type: str) -> dict[str, str]:
    return {"path": path, "media_type": media_type}


def success_result(
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
