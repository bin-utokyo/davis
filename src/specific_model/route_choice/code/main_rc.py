
import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import minimize

import re
import ast

from .abc import RouteChoiceModel
from .model import RL
from .definition import Network, LinkTransition

def get_model(model_name: str, network: Network, estimate_discount: bool = True, beta: float = 0.9) -> RouteChoiceModel:
    if model_name == "RL":
        return RL(network, estimate_discount=estimate_discount, beta=beta)
    raise ValueError(f"Unknown model name: {model_name}")

def estimate(input_dir: str, output_dir: str, model_name: str = "RL") -> None:
    df_link = pd.read_csv(os.path.join(input_dir, "link.csv"), engine="pyarrow")
    df_node = pd.read_csv(os.path.join(input_dir, "node.csv"), engine="pyarrow")
    df_transition = pd.read_csv(os.path.join(input_dir, "transition.csv"), engine="pyarrow")

    network = Network(df_node, df_link)

    model = get_model(model_name, network)

    transition_list = [LinkTransition.from_dict(row, network, model) for row in df_transition.to_dict(orient="records")]

    # function to compute ll
    def compute_minus_ll(params: np.ndarray) -> float:
        ll = 0.0
        for transition in transition_list:
            ll += transition.calculate_log_likelihood(params)
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
    x0 = np.zeros(model.get_param_size())
    res = minimize(compute_minus_ll, x0, method="Nelder-Mead")

    t_val = tval(res.x)
    LL0 = -compute_minus_ll(x0)
    LL = -compute_minus_ll(res.x)
    rho2 = 1 - LL / LL0
    adj_rho2 = 1 - (LL - len(res.x)) / LL0
    aic = -2 * LL + 2 * len(res.x)

    result_str = f"""
    sample number = {len(transition_list)}
        variables = {network.f_name}
        parameter = {res.x}
          t value = {t_val}
               L0 = {LL0}
               LL = {LL}
             rho2 = {rho2}
    adjusted rho2 = {adj_rho2}
              AIC = {aic}
         discount = {model.get_beta(res.x) if hasattr(model, 'get_beta') else 'N/A'}
    """
    print(result_str)

    if output_dir is not None:
        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write(result_str)


def simulate(input_dir: str, output_dir: str, model_name: str = "RL") -> None:
    df_link = pd.read_csv(os.path.join(input_dir, "link.csv"), engine="pyarrow")
    df_node = pd.read_csv(os.path.join(input_dir, "node.csv"), engine="pyarrow")
    df_transition = pd.read_csv(os.path.join(input_dir, "transition.csv"), engine="pyarrow")

    network = Network(df_node, df_link)

    model = get_model(model_name, network)

    df_transition.drop(columns="NextLinkID", inplace=True)
    transition_list = [LinkTransition.from_dict(row, network, model) for row in df_transition.to_dict(orient="records")]

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
    
    # Simulation
    next_link_ids = []
    for transition in transition_list:
        next_link_id = model.choose_transition(transition, params)
        next_link_ids.append(next_link_id)

    df_transition["NextLinkID"] = next_link_ids
    df_transition.to_csv(os.path.join(output_dir, "transition_simulated.csv"), index=False)

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 4:
        print("Usage: python main_mc.py <estimation_mode> <input_dir> <output_dir> [<model_name>]")
        print(argv)
        exit(1)

    estimation_mode = argv[1]
    input_dir = argv[2]
    output_dir = argv[3]
    model_name = argv[4] if len(argv) > 4 else "RL"

    if estimation_mode == "true":
        estimate(input_dir, output_dir, model_name)
    elif estimation_mode == "false":
        simulate(input_dir, output_dir, model_name)
    else:
        print(f"Unknown estimation mode: {estimation_mode}")
        exit(1)