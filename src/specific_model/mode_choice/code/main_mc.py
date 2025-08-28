from typing import Any, Optional
import os
import sys
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from definition import Los, Trip


def estimate(input_dir: str, output_dir: Optional[str]) -> None:
    df_los = pd.read_csv(os.path.join(input_dir, "los.csv"), engine="pyarrow")
    df_trip = pd.read_csv(os.path.join(input_dir, "trip.csv"), engine="pyarrow")
    print([row
        for row in df_los.to_dict(orient="records")
    ])

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

    trips = [
        Trip(
            trip_id=row["TripID"],
            dep_time=row["DepartureTime"],
            arr_time=row["ArrivalTime"],
            o_zone=row["OZone"],
            d_zone=row["DZone"],
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
        print(res)
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
        variables = {los_dict[list(los_dict.keys())[0]].attribute_names}
        parameter = {res.x}
          t value = {t_val}
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


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 4:
        print("Usage: python main_mc.py <estimation_mode> <input_dir> <output_dir>")
        print(argv)
        exit(1)

    estimation_mode = bool(argv[1])
    input_dir = argv[2]
    output_dir = argv[3]
    if estimation_mode:
        estimate(input_dir, output_dir)
    else:
        raise NotImplementedError(f"Estimation mode '{estimation_mode}' is not supported.")