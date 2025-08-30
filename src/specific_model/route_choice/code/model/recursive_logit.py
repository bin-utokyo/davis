import os
import sys
from logging import getLogger, StreamHandler, Formatter

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, csr_array, vstack, hstack


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from abc_rc import RouteChoiceModel

__all__ = ["RL"]

# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"

logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)


class RL(RouteChoiceModel):
    def __init__(self, network: "Network", estimate_discount: bool = True, beta: float = 0.9):
        self.network = network
        self.estimate_discount = estimate_discount
        self.beta = beta

        self.X = self._get_X()
        self.link_adj_matrix = self._get_expanded_link_adj_matrix()

        self.U_cache: dict[tuple[bytes, float], np.ndarray] = dict()  # key: (params.tobytes(), mu)
        self.M_cache: dict[tuple[bytes, float], csr_matrix] = dict()  # key: (params.tobytes(), mu)
        self.V_cache: dict[tuple[bytes, int, float], np.ndarray] = dict()  # key: (params.tobytes(), d_idx, mu)

    def get_param_size(self) -> int:
        """Get the size of the model parameters.

        Returns:
            int: The size of the model parameters.
        """
        return (len(self.network.f_name) + (1 if self.estimate_discount else 0))

    def is_valid(self, params: np.ndarray) -> bool:
        """Check if the model parameters are valid.

        Args:
            params (np.ndarray): Model parameters.

        Returns:
            bool: True if the parameters are valid, False otherwise.
        """
        return len(params) == self.get_param_size()

    def calculate_transition_probability(self, link_transition: "LinkTransition", params: np.ndarray) -> dict[int, float]:
        """Calculate the transition probability for a given link transition.

        Args:
            link_transition (LinkTransition): The link transition object.
            params (np.ndarray): Model parameters.

        Returns:
            dict[int, float]: A dictionary mapping link IDs to their transition probabilities.
        """
        exp_u = self.exp_util_array(params)
        exp_v = self.value_array(params, link_transition.destination_node_id)
        beta = self.get_beta(params)

        down_link_idxs = np.array([self.network.link_id2idx[lid] for lid in link_transition.down_link_ids])
        link_idx = self.network.link_id2idx[link_transition.link_id]

        probs = exp_u[down_link_idxs] * (exp_v[down_link_idxs] ** beta) / exp_v[link_idx]

        # normalize
        probs = probs / probs.sum()
        return {lid: prob for lid, prob in zip(link_transition.down_link_ids, probs)}

    def exp_util_array(self, params: np.ndarray, mu: float = 1) -> np.ndarray:
        """
        Compute the expected utility array for the model.

        Args:
            params (np.ndarray): Model parameters.
            mu (float, optional): Scaling factor for the utilities. Defaults to 1.

        Returns:
            np.ndarray: Expected utility array. The shape is (n_link + 1,).
        """
        cache_key = (params.tobytes(), mu)
        if cache_key in self.U_cache:
            return self.U_cache[cache_key]

        if self.estimate_discount:
            params = params[:-1]  # last parameter is discount factor

        utils = np.einsum('ij,j->i', self.X, params)  # (n_link + 1,)
        result = np.exp(utils)
        self.U_cache[cache_key] = result
        return result

    def util_matrix(self, params: np.ndarray, mu: float = 1) -> csr_matrix:
        """
        Compute the utility matrix for the model.

        Args:
            params (np.ndarray): Model parameters.
            mu (float, optional): Scaling factor for the utilities. Defaults to 1.

        Returns:
            csr_matrix: Utility matrix of shape (n_link, n_link + 1).
        """
        cache_key = (params.tobytes(), mu)
        if cache_key in self.M_cache:
            return self.M_cache[cache_key]
        else:# clear cache
            self.M_cache.clear()
            self.V_cache.clear()
            self.U_cache.clear()

        if self.estimate_discount:
            params = params[:-1]  # last parameter is discount factor

        utils = np.einsum('ij,j->i', self.X, params)  # (1, n_link + 1)
        utils = np.clip(utils, -30, 30)  # Avoid overflow
        exp_mu_utils = np.exp(mu * utils)

        M = self.link_adj_matrix.tocsc(copy=True)
        for i in range(M.shape[1]):
            M.data[M.indptr[i]:M.indptr[i + 1]] *= exp_mu_utils[i]
        M = M.tocsr()

        self.M_cache[cache_key] = csr_matrix(M)
        return self.M_cache[cache_key]
    
    def get_beta(self, params: np.ndarray) -> float:
        """Get the discount factor (beta) for the model.

        Args:
            params (np.ndarray): Model parameters.

        Returns:
            float: Discount factor (beta).
        """
        if self.estimate_discount:
            return 1 / (1 + np.exp(params[-1]))
        else:
            return self.beta

    def value_array(self, params: np.ndarray, d_node_id: int, mu: float = 1) -> np.ndarray:
        """Compute the value array for the model.

        Args:
            params (np.ndarray): Model parameters.
            d_node_id (int): Destination node ID.
            mu (float, optional): Scaling factor for the utilities. Defaults to 1.

        Returns:
            np.ndarray: Value array. The shape is (n_link + 1,).
        """
        cache_key = (params.tobytes(), d_node_id, mu)
        if cache_key in self.V_cache:
            return self.V_cache[cache_key]

        M = self.util_matrix(params, mu=mu)
        V = np.zeros((self.network.n_link + 1, 1), dtype=np.float32)  # last link is dummy link
        V[-1] = 1

        b = np.zeros((self.network.n_link + 1, 1), dtype=np.float32)
        b[-1] = 1
        b = csr_array(b)

        M_d = M.tolil(copy=True)
        d_idxs = np.array(self.network.up_link_idx[d_node_id])
        M_d[d_idxs, -1] = 1
        M_d = M_d.tocsr()

        beta = self.get_beta(params)

        # 価値反復による価値関数の更新
        tms = 1
        dl_z = 100.
        while dl_z > 0.1 or tms <= (np.sqrt(self.network.n_link) * 2):
            V_pre = V.copy()
            V_new = M_d @ (csr_array(V ** beta)) + b
            V = V_new.toarray()
            dl_z = np.linalg.norm(V_new - V_pre, axis=0)
            tms += 1
            if tms > 100000:
                logger.warning("Value iteration did not converge within 100000 iterations.")
                break
            if np.isnan(V).any():
                raise ValueError("NaN encountered in value iteration. Check model parameters.")

        self.V_cache[cache_key] = V.flatten()
        return self.V_cache[cache_key]

    def _get_X(self) -> np.ndarray:
        """
        Get the design matrix X for the model.

        Returns:
            np.ndarray: Design matrix of shape (n_link + 1, n_features)
        """
        X = np.zeros((self.network.n_link + 1, len(self.network.f_name)), dtype=np.float32) # last link is dummy link
        for i, val in enumerate(self.network.attr.values()):
            if np.isnan(val).any():
                logger.warning("NaN values found in link attributes. They will be treated as 0.")
                val = np.nan_to_num(val, nan=0)
            X[:-1, i] = val
        return X

    def _get_expanded_link_adj_matrix(self) -> csr_array:
        """
        Get the expanded link adjacency matrix for the model.

        Returns:
            csr_array: Expanded link adjacency matrix of shape (n_link + 1, n_link + 1)
        """
        adj = self.network.link_adj_matrix.tolil()
        link_adj_matrix = hstack([adj, lil_matrix((adj.shape[0], 1))])
        link_adj_matrix = vstack([link_adj_matrix, lil_matrix((1, link_adj_matrix.shape[1]))])
        return link_adj_matrix.tocsr()

from definition import Network, LinkTransition