import sys
from pathlib import Path

from typing import Optional, cast
import numpy as np
from scipy.sparse import csr_array

sys.path.append(str(Path(__file__).parent.parent))
from definition import Network

__all__ = ["dial_assignment", "get_prev_link", "get_shared_link", "get_heading"]


def dial_assignment(u_od: np.ndarray, adj_mat: np.ndarray, min_costs: np.ndarray, prev_links: np.ndarray, c: Optional[np.ndarray] = None, P_2: Optional[np.ndarray] = None, shared_links: Optional[np.ndarray] = None, up_links: Optional[np.ndarray] = None, headings: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Dial assignment function.

    Args:
        u_od (np.ndarray): Demand from each link. Shape: (link_num, link_num).
        adj_mat (np.ndarray): Adjacency matrix. Shape: (link_num, link_num).
        min_costs (np.ndarray): Minimum cost for each link. Shape: (link_num, link_num). ij component represents cost from i's end to j's end.
        prev_links (np.ndarray): Previous link idx for each link. Shape: (link_num, link_num). ij component represents previous link from i to j.
        c (np.ndarray | None, optional): Cost for each link. Shape: (link_num). Defaults to None.
        P_2 (np.ndarray | None, optional): Transition probability among links. Shape: (link_num, link_num). Defaults to None. ij component represents transition from i to j.
        shared_links (np.ndarray | None, optional): Whether each link pair shares the same end node. Shape: (link_num, link_num). Defaults to None.
        up_links (np.ndarray | None, optional): Upstream link idx for each link. Shape: (link_num). Defaults to None.
        headings (np.ndarray | None, optional): Heading links for each link. Shape: (link_num, link_num). Defaults to None.

    Returns:
        u_l (np.ndarray): Traffic in each link. Shape: (link_num).
    """
    # input check
    link_num = len(u_od)
    if c is None and P_2 is None:
        raise ValueError("c or P_2 are required.")
    if c is not None and c.shape != (link_num,):
        raise ValueError(f"c shape should be (link_num). c.shape: {c.shape}, link_num: {link_num}.")
    if P_2 is not None and P_2.shape != (link_num, link_num):
        raise ValueError(f"P_2 shape should be (link_num, link_num). P_2.shape: {P_2.shape}, link_num: {link_num}.")

    adj_mat = adj_mat.astype(np.bool_)
    np_arrange = np.arange(link_num)

    # get links sharing the same end node
    if shared_links is None:
        shared_links = np.eye(link_num, dtype=np.bool_)
        d_link_max = np.argmax(adj_mat, axis=1)

        shared_links_tmp = d_link_max[:, np.newaxis] == d_link_max[np.newaxis, :]

        is_valid = np.sum(adj_mat, axis=1) > 0
        shared_links[is_valid] = shared_links_tmp[is_valid]


    # get one uplink for each link
    if up_links is None:
        up_links: np.ndarray = np.argmax(adj_mat, axis=0)
        up_links[np.sum(adj_mat, axis=0) == 0] = -1

    # Compute the link likelihood
    if P_2 is None:
        link_likelihood = np.zeros((link_num, link_num), dtype=np.float32)
        min_costs_pad = np.zeros((link_num, link_num + 1), dtype=np.float32)
        min_costs_pad[:, :-1] = min_costs
        min_costs_pad[:, -1] = np.inf

        # link_likelihood[o, l] = exp(min_costs[o, l] - min_costs[o, uplink] - c[l])
        p = np.exp(min_costs - min_costs_pad[:, up_links] - np.expand_dims(cast(np.ndarray, c), axis=0))  # (link_num, link_num)
        mask = (min_costs >= min_costs_pad[:, up_links]) & ~np.isinf(min_costs_pad[:, up_links])
        link_likelihood = np.where(mask, p, np.zeros((link_num, link_num), dtype=np.float32))
    else:
        link_likelihood = np.zeros((link_num, link_num), dtype=np.float32)
        # link_likelihood[o, l] = P_2[l', l] if l is acievable and l' is prev_link of l
        P_2_pad = np.zeros((link_num + 1, link_num), dtype=np.float32)
        P_2_pad[:-1, :] = P_2
        
        link_likelihood = P_2_pad[prev_links, np_arrange]

    link_likelihood[np_arrange[:, np.newaxis], np_arrange[:, np.newaxis]] = 1.0
    
    # for each origin link
    ## compute the link weight
    argsort = np.argsort(min_costs, axis=1)
    weight = np.zeros((link_num, link_num), dtype=np.float32)  # ij component represents weight when origin is i and current link is j
    weight = np.where(min_costs == 0, link_likelihood, np.zeros((link_num, link_num), dtype=np.float32))
    weight = _dial_forward(link_num, argsort, adj_mat, link_likelihood, weight, np_arrange)
    if headings is not None:
        weight[headings == 0] = 0.0

    ## compute link flow
    argsort = np.argsort(min_costs, axis=1)[:, ::-1]
    u_l = np.zeros((link_num, link_num), dtype=np.float32)
    
    u_l = _dial_backward(link_num, argsort, adj_mat, weight, np_arrange, u_od, shared_links, u_l)

    u_l = u_l.sum(axis=0)

    return u_l

def _dial_forward(link_num: int, argsort: np.ndarray, adj_mat: np.ndarray, link_likelihood: np.ndarray, weight: np.ndarray, np_arrange: np.ndarray) -> np.ndarray:
    for d in range(link_num):
        tmp_links = argsort[:, d]
        tmp_uplink_all = adj_mat[:, tmp_links].transpose(1, 0)

        w = link_likelihood[np_arrange, tmp_links] * np.sum(weight * tmp_uplink_all, axis=1)
        mask = tmp_links != np_arrange

        weight[np_arrange, tmp_links] = np.where(mask, w, weight[np_arrange, tmp_links])
    return weight

def _dial_backward(link_num: int, argsort: np.ndarray, adj_mat: np.ndarray, weight: np.ndarray, np_arrange: np.ndarray, u_od: np.ndarray, shared_links: np.ndarray, u_l: np.ndarray) -> np.ndarray:
    for d in range(link_num):
        tmp_links = argsort[:, d]

        deno = np.sum(weight * shared_links[tmp_links, :], axis=1)

        rate = weight[np_arrange, tmp_links] / (deno + (deno == 0.0))

        u_l[np_arrange, tmp_links] = u_od[np_arrange, tmp_links] + (u_l * adj_mat[tmp_links, :]).sum(axis=1) * rate
    return u_l


def get_prev_link(network: Network) -> np.ndarray:
    """
    Link idx of one upstream link of each link.

    Args:
        network (Network): Network object.

    Returns:
        np.ndarray: Array of previous link indices. Shape: (link_num, link_num). ij component represents previous link of j moving from i to j.
    """
    prev_link = network.link_predecessor.copy().astype(np.int64)
    prev_link[prev_link == -9999] = -1
    return prev_link

def get_shared_link(network: Network) -> np.ndarray:
    """
    Whether each link pair shares the same end node.

    Args:
        network (Network): Network object.

    Returns:
        np.ndarray: Array indicating whether each link pair shares the same end node. Shape: (link_num, link_num). ij component is True if link i and link j share the same end node, else False.
    """
    share_dnode_link = np.eye(network.n_link, dtype=bool)
    up_matrix = (network.incidence_matrix == -1)
    for j in range(network.n_node):
        idx = cast(csr_array, up_matrix)[j].nonzero()[1]
        if len(idx) > 0:
            share_dnode_link[idx[:, np.newaxis], idx[np.newaxis, :]] = True
    return share_dnode_link

def get_heading(network: Network) -> np.ndarray:
    """
    Whether each link is heading link i to j.

    Args:
        network (Network): Network object.

    Returns:
        np.ndarray: Array indicating whether each link is leaving from link i to j. Shape: (link_num, link_num). ij component is True if link j is heading link from link i, else False.
    """
    heading = np.eye(network.n_link, dtype=bool)

    o_node_idxs = np.array([network.node_id2idx[nid] for nid in network.link_start])
    d_node_idxs = np.array([network.node_id2idx[nid] for nid in network.link_end])

    o_node_dist = network.dist_matrix[o_node_idxs, :][:, o_node_idxs]
    d_node_dist = network.dist_matrix[o_node_idxs, :][:, d_node_idxs]

    heading = np.logical_or(o_node_dist < d_node_dist, heading)
    return heading
