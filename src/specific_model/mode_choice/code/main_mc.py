from typing import Any, Optional
import os
import sys
import re
import ast

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from definition import Los, Trip
from abc_mc import ModeChoiceModel
from model import MNL


def get_model(model_name: str) -> ModeChoiceModel:
    if model_name == "MNL":
        return MNL()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def estimate(input_dir: str, output_dir: Optional[str], model_name: str = "MNL") -> None:
    """
    Estimate the mode choice model parameters.

    Args:
        input_dir (str): The directory containing input CSV files.
        output_dir (Optional[str]): The directory to save output files.
        model_name (str, optional): The name of the mode choice model to use. Defaults to "MNL".

    Returns:
        None
    """
    df_los = pd.read_csv(os.path.join(input_dir, "los.csv"), engine="pyarrow")
    df_trip = pd.read_csv(os.path.join(input_dir, "trip.csv"), engine="pyarrow")

    los_dict = {
        (row["OZone"], row["DZone"]): Los.from_dict(row)
        for row in df_los.to_dict(orient="records")
    }

    df_trip = df_trip.astype({
        "TripID": int,
        "DepartureTime": "datetime64[ns]",
        "ArrivalTime": "datetime64[ns]",
        "OZone": int,
        "DZone": int,
        "Mode": int
    })

    model = get_model(model_name)

    trips = [
        Trip(
            trip_id=row["TripID"],
            dep_time=row["DepartureTime"],
            arr_time=row["ArrivalTime"],
            o_zone=row["OZone"],
            d_zone=row["DZone"],
            model=model,  # dependency injection
            mode=row["Mode"]
        )
        for row in df_trip.to_records(index=False)
    ]

    # function to compute ll
    def compute_minus_ll(params: np.ndarray) -> float:
        ll = 0.0
        for trip in trips:
            los = trip.get_los(los_dict)
            if los is not None:
                ll += trip.calculate_log_likelihood(los, params)
        return -ll

    def compute_hessian(params: np.ndarray) -> np.ndarray:
        h = 10 ** -4  # 数値微分用の微小量
        n = len(params)
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                e_i, e_j = np.zeros(n), np.zeros(n)
                e_i[i] = 1
                e_j[j] = 1

                res[i][j] = (-compute_minus_ll(params + h * e_i + h * e_j)
                            + compute_minus_ll(params + h * e_i - h * e_j)
                            + compute_minus_ll(params - h * e_i + h * e_j)
                            - compute_minus_ll(params - h * e_i - h * e_j)) / (4 * h * h)
        return res

    def tval(x: np.ndarray) -> np.ndarray:
        return x / np.sqrt(-np.diag(np.linalg.inv(compute_hessian(x))))
    
    # Estimate
    x0 = np.zeros(len(los_dict[list(los_dict.keys())[0]].attribute_names))
    res = minimize(compute_minus_ll, x0, method="Nelder-Mead")
    t_val = tval(res.x)
    LL0 = -compute_minus_ll(x0)
    LL = -compute_minus_ll(res.x)
    rho2 = 1 - LL / LL0
    adj_rho2 = 1 - (LL - len(res.x)) / LL0
    aic = -2 * LL + 2 * len(res.x)

    result_str = f"""
    sample number = {len(trips)}
        variables = {', '.join(los_dict[list(los_dict.keys())[0]].attribute_names)}
        parameter = {', '.join(map(str, res.x))}
          t value = {', '.join(map(str, t_val))}
               L0 = {LL0}
               LL = {LL}
             rho2 = {rho2}
    adjusted rho2 = {adj_rho2}
              AIC = {aic}
    """
    print(result_str)

    if output_dir is not None:
        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write(result_str)


def simulate(input_dir: str, output_dir: str, model_name: str = "MNL"):
    df_los = pd.read_csv(os.path.join(input_dir, "los.csv"), engine="pyarrow")
    df_trip = pd.read_csv(os.path.join(input_dir, "trip.csv"), engine="pyarrow")

    los_dict = {
        (row["OZone"], row["DZone"]): Los.from_dict(row)
        for row in df_los.to_dict(orient="records")
    }

    df_trip = df_trip.astype({
        "TripID": int,
        "DepartureTime": "datetime64[ns]",
        "ArrivalTime": "datetime64[ns]",
        "OZone": int,
        "DZone": int
    })

    model = get_model(model_name)

    # Create Trip objects (Mode is not used because we are simulating)
    trips = [
        Trip(
            trip_id=row["TripID"],
            dep_time=row["DepartureTime"],
            arr_time=row["ArrivalTime"],
            o_zone=row["OZone"],
            d_zone=row["DZone"],
            model=model  # dependency injection
        )
        for row in df_trip.to_records(index=False)
    ]

    # Read parameter
    param_path = os.path.join(input_dir, "result.txt")
    with open(param_path, encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"parameter\s*=\s*(\[[^\]]+\])", text)
    if match:
        param_list = ast.literal_eval(match.group(1))  # list[float]
    else:
        raise ValueError("Failed to extract parameters from input/result.txt")
    params = np.array(param_list, dtype=np.float32)

    # Simulate mode choice for each trip
    modes = []
    for trip in trips:
        los = trip.get_los(los_dict)
        if los is not None:
            mode = trip.choose_mode(los, params)
            modes.append(mode)

    df_trip["Mode"] = modes
    df_trip.to_csv(os.path.join(output_dir, "trip_simulated.csv"), index=False)

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 4:
        print("Usage: python main_mc.py <estimation_mode> <input_dir> <output_dir> [<model_name>]")
        exit(1)

    estimation_mode = argv[1]
    input_dir = argv[2]
    output_dir = argv[3]
    model_name = argv[4] if len(argv) > 4 else "MNL"
    if estimation_mode == "true":
        estimate(input_dir, output_dir, model_name)
    elif estimation_mode == "false":
        simulate(input_dir, output_dir, model_name)
    else:
        raise NotImplementedError(f"Estimation mode '{estimation_mode}' is not supported. Supported modes are: true, false")