import numpy as np
import pandas as pd
import os
import sys
import json
from scipy.optimize import minimize

import re
import ast

from abc_rc import RouteChoiceModel
from model import RL
from definition import Network, NetworkIO, LinkTransition, PP
from algorithm import HybridMapmatching, dial_assignment, get_prev_link, get_shared_link, get_heading


def read_csv(file: str) -> pd.DataFrame:
    return pd.read_csv(file, engine="pyarrow")


def get_model(model_name: str, network: Network, estimate_discount: bool = True, beta: float = 0.9) -> RouteChoiceModel:
    if model_name == "RL":
        return RL(network, estimate_discount=estimate_discount, beta=beta)
    raise ValueError(f"Unknown model name: {model_name}")


def read_params(param_path: str) -> np.ndarray:
    with open(param_path, encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"parameter\s*=\s*(\[[^\]]+\])", text)
    if match:
        param_list = ast.literal_eval(match.group(1))  # list[float]
    else:
        raise ValueError("Failed to extract parameters from input/result.txt")
    params = np.array(param_list, dtype=np.float32)
    return params


def get_network_from_osm(input_dir: str, polygon_coord: list[list[float]]) -> None:
    node_file = os.path.join(input_dir, "node.csv")
    link_file = os.path.join(input_dir, "link.csv")

    NetworkIO.get_from_osm(polygon_coord, node_file, link_file)


def mapmatching(input_dir: str, mode: str) -> None:
    node_file = os.path.join(input_dir, "node.csv")
    link_file = os.path.join(input_dir, "link.csv")
    trip_file = os.path.join(input_dir, "trip.csv")
    feeder_file = os.path.join(input_dir, "feeder.csv")
    loc_file = os.path.join(input_dir, "loc.csv")
    output_file = os.path.join(output_dir, "transition.csv")

    # Create network
    df_node = read_csv(node_file)
    df_link = read_csv(link_file)

    network = Network(df_node, df_link)

    # Create PP
    pp = PP.load(input_dir, trip_file, feeder_file, loc_file)

    # Mapmatching
    mapmatching = HybridMapmatching(network, 10.0, [mode], 10.0)
    result = mapmatching.match(pp)

    # Output result
    result.to_csv(output_file, index=False)


def estimate(input_dir: str, output_dir: str, model_name: str = "RL") -> None:
    df_link = read_csv(os.path.join(input_dir, "link.csv"))
    df_node = read_csv(os.path.join(input_dir, "node.csv"))
    df_transition = read_csv(os.path.join(input_dir, "transition.csv"))

    network = Network(df_node, df_link)

    model = get_model(model_name, network)

    transition_list = [LinkTransition.from_dict(row, network, model) for row in df_transition.to_dict(orient="records")]
    transition_list = [t for t in transition_list if t is not None]# remove None values

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
        variables = [{', '.join(map(str, network.f_name))}]
        parameter = [{', '.join(map(str, res.x))}]
          t value = [{', '.join(map(str, t_val))}]
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
    df_link = read_csv(os.path.join(input_dir, "link.csv"))
    df_node = read_csv(os.path.join(input_dir, "node.csv"))
    df_transition = read_csv(os.path.join(input_dir, "transition.csv"))

    network = Network(df_node, df_link)

    model = get_model(model_name, network)

    df_transition.drop(columns="NextLinkID", inplace=True)
    transition_list = [LinkTransition.from_dict(row, network, model) for row in df_transition.to_dict(orient="records")]

    # Read parameter
    param_path = os.path.join(input_dir, "result.txt")
    params = read_params(param_path)
    
    # Simulation
    next_link_ids = []
    for transition in transition_list:
        if transition is not None:
            next_link_id = model.choose_transition(transition, params)
            next_link_ids.append(next_link_id)
        else:
            next_link_ids.append(None)

    df_transition["NextLinkID"] = next_link_ids
    df_transition.to_csv(os.path.join(output_dir, "transition_simulated.csv"), index=False)


def assignment(input_dir: str, output_dir: str) -> None:
    df_link = read_csv(os.path.join(input_dir, "link.csv"))
    df_node = read_csv(os.path.join(input_dir, "node.csv"))
    df_demand = read_csv(os.path.join(input_dir, "demand.csv"))

    network = Network(df_node, df_link)

    model = get_model(model_name, network)

    # Read parameter
    param_path = os.path.join(input_dir, "result.txt")
    params = read_params(param_path)

    # Assignment for each OD pair
    o_node_ids = df_demand["OriginNodeID"].to_numpy()
    d_node_ids = df_demand["DestinationNodeID"].to_numpy()
    demands = df_demand["Demand"].to_numpy()

    ## Prepare network properties
    prev_link = get_prev_link(network)
    shared_link = get_shared_link(network)
    heading = get_heading(network)

    link_flow = np.zeros(network.n_link, dtype=np.float32)
    for i in range(len(df_demand)):
        _, path = network.get_shortest_path(o_node_ids[i], d_node_ids[i])
        if len(path) > 1:
            o_link_id = path[0]
            d_link_id = path[-1]

            link_transition = LinkTransition(i, o_link_id, None, d_node_ids[i], [], model)
            
            P2 = model.calculate_transition_probabilities(link_transition, params)
            u_od = np.zeros((network.n_link, network.n_link), dtype=np.float32)
            u_od[network.link_id2idx[o_link_id], network.link_id2idx[d_link_id]] = demands[i]

            link_flow_tmp = dial_assignment(
                u_od,
                network.link_adj_matrix.toarray().astype(np.float32),
                network.link_dist_matrix.astype(np.float32),
                prev_link,
                P_2=P2,
                shared_links=shared_link,
                headings=heading,
            )
            link_flow += link_flow_tmp

    # Output result
    df_flow = pd.DataFrame({
        "LinkID": network.link_list,
        "LinkFlow": link_flow,
    })
    output_file = os.path.join(output_dir, "link_flow.csv")
    df_flow.to_csv(output_file, index=False)


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 11:
        print("Usage: python main_rc.py <mode> <polygon_coord> <network> <mapmatching> <estimate> <simulate> <assignment> <input_dir> <output_dir> [<model_name>]")
        exit(1)

    mode = argv[1]
    polygon_coord: list[list[float]] = json.loads(argv[2])
    network_mode = argv[3]
    mapmatching_mode = argv[4]
    estimation_mode = argv[5]
    simulation_mode = argv[6]
    assignment_mode = argv[7]
    input_dir = argv[8]
    output_dir = argv[9]
    model_name = argv[10] if len(argv) > 10 else "RL"

    if network_mode == "true":
        get_network_from_osm(input_dir, polygon_coord)
    if mapmatching_mode == "true":
        mapmatching(input_dir, mode)
    if estimation_mode == "true":
        estimate(input_dir, output_dir, model_name)
    if simulation_mode == "true":
        simulate(input_dir, output_dir, model_name)
    if assignment_mode == "true":
        assignment(input_dir, output_dir)
        