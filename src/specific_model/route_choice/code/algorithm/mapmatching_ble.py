from pathlib import Path
import sys

from typing import Any, Optional, cast
from abc import ABC, abstractmethod
import os
from logging import getLogger, StreamHandler, Formatter
import tqdm
import numpy as np
import pandas as pd
import dask.dataframe as dd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from abc_rc import RouteChoiceModel
from definition import BLE, BLENetwork, LinkTransition
from algorithm import Viterbi

__all__ = ["BLEMapmatchingABC", "BLEHybridMapmatching"]

# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"
logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)


class BLEMapmatchingABC(ABC):
    """Abstract base class for BLE map matching algorithms.

    Subclasses implement concrete matching strategies (e.g., greedy DP,
    HMM/Viterbi). The interface standardizes how a preprocessed data container
    (`BLE`) is converted into a sequence of link transitions.
    """
    @abstractmethod
    def match(self, ble: BLE, overlap_threshold: int = 5, allow_circle: bool = False, max_trip: int | None = None) -> pd.DataFrame:
        """Run map matching over all trips/feeders in a `BLE`.

        Args:
            ble (BLE): Preprocessed BLE signal data container. 
            overlap_threshold (int): Window (in data index steps) to collapse
                repeated visits to the same link. If a link reappears within
                this window, intermediate links are collapsed into that link.
            allow_circle (bool): If False, removes circular subpaths by
                collapsing loops detected at repeated start nodes.
            max_trip (int | None): Limit on number of trips to process. If
                None, all trips in `ble` are processed.

        Returns:
            pd.DataFrame: Link transitions for all matched feeders with columns:
                - `TripID`: trip identifier
                - `LinkID`: current link id
                - `NextLinkID`: next link id
                - `DestinationNodeID`: destination (last) link id in the path
        """
        pass


class BLEHybridMapmatching(BLEMapmatchingABC):
    """Hybrid matcher combining segmentation with HMM/Viterbi.
    """
    def __init__(self, model: Optional[RouteChoiceModel] = None, params: Optional[np.ndarray] = None):
        """
        Initialize HybridMapmatching.
        """
        self.network = None
        self.model = model
        self.prior_transition_prob = None

        if model is not None and params is not None:
            self.link_transition_prob_function = lambda network, emission_prob: BLEHybridMapmatching.link_transition_probability_by_model(model, params, network, emission_prob)
        else:
            self.link_transition_prob_function = None

    def set_model(self, model: RouteChoiceModel, params: np.ndarray) -> None:
        """
        Set the route choice model and parameters for mapmatching.

        Args:
            model (RouteChoiceModel): Route choice model
            params (np.ndarray): Model parameters
        """
        self.model = model
        self.params = params

        self.link_transition_prob_function = lambda network, emission_prob: BLEHybridMapmatching.link_transition_probability_by_model(model, params, network, emission_prob) * 0.99 + (self.prior_transition_prob if self.prior_transition_prob is not None else 1 / network.n_link) * 0.01

    def match(self, ble: BLE, overlap_threshold: int = 5, allow_circle: bool = False, max_trip: int | None = None) -> pd.DataFrame:
        """
        Run hybrid map matching over all trips/feeders in a `BLE`.

        Args:
            ble (BLE): Preprocessed BLE signal data container. 
            overlap_threshold (int): Window (in data index steps) to collapse
                repeated visits to the same link. If a link reappears within
                this window, intermediate links are collapsed into that link.
            allow_circle (bool): If False, removes circular subpaths by
                collapsing loops detected at repeated start nodes.
            max_trip (int | None): Limit on number of trips to process. If
                None, all trips in `ble` are processed.

        Returns:
            pd.DataFrame: Link transitions for all matched feeders with columns:
                - `TripID`: trip identifier
                - `LinkID`: current link id
                - `NextLinkID`: next link id
                - `DestinationNodeID`: destination (last) link id in the path
        """
        self.network = ble.network
        self.prior_transition_prob = BLEHybridMapmatching.get_prior_transition_prob(ble.network)

        result = []
        if max_trip is None:
            max_trip = len(ble.record)

        for i in tqdm.tqdm(range(max_trip), desc="Trips"):
            trip = ble.record[i]
            emission_prob = ble.get_emission_probability(trip)
            if self.link_transition_prob_function is not None:
                transition_prob = self.link_transition_prob_function(ble.network, emission_prob)
            else:
                transition_prob = None
            path = self.match_one_feeder(trip, emission_prob, transition_prob, overlap_threshold, allow_circle)

            if path is not None:
                kab_path = self.path2kab(path)
                tmp_result = [[i+1, *kab_path[j]] for j in range(len(kab_path))]  # [feeder_id, tmp_link_id, next_link_id, last_link_id]
                result.extend(tmp_result)
        if len(result) == 0:
            logger.warning("Mapmatching: no result is found.")
            return pd.DataFrame(columns=["TripID", "LinkID", "NextLinkID", "DestinationNodeID", "DestinationLinkID"])

        df_result = pd.DataFrame(result, columns=["TripID", "LinkID", "NextLinkID", "DestinationLinkID"])
        df_result["DestinationNodeID"] = df_result["DestinationLinkID"].apply(lambda x: ble.network.link_end[ble.network.link_id2idx[x]])
        logger.info(f"HybridMapmatching: {len(result)} link transitions are obtained.")
        return df_result

    def match_one_feeder(self, trip: dd.DataFrame, emission_prob: np.ndarray, transition_prob: Optional[np.ndarray], overlap_threshold: int = 5, allow_circle: bool = False) -> list[int] | None:
        """
        Perform mapmatching for one feeder.

        Args:
            trip (dd.DataFrame): Trip data for mapmatching
            emission_prob (np.ndarray): Emission probabilities for the trip data
            transition_prob (np.ndarray): Transition probabilities for the trip data
            overlap_threshold (int): threshold to remove overlapped links
            allow_circle (bool): whether to allow circle path

        Returns:
            path (List[int]): list of link ids
        """
        if self.network is None:
            raise ValueError("Network is not set for mapmatching.")
        if transition_prob is None:
            if self.prior_transition_prob is None:
                self.prior_transition_prob = BLEHybridMapmatching.get_prior_transition_prob(self.network)
            transition_prob = self.prior_transition_prob
        path = self._perform_viterbi(trip, emission_prob, transition_prob)
        if len(path) == 0:
            return None

        # complete the path by shortest path
        path_complete: list[int] = list()
        for i in range(1, len(path)):
            # csr_matrix has getrow(), not get_row()
            if path[i] != path[i-1] and path[i] not in self.network.link_adj_matrix.getrow(path[i-1]).indices:
                inter_path = self.network.get_shortest_path(self.network.link_end[path[i-1]], self.network.link_start[path[i]])[1]
                inter_path = [self.network.link_id2idx[lid] for lid in inter_path]
                if inter_path is not None:
                    path_complete = path_complete + inter_path
            else:
                path_complete.append(path[i-1])
        path_complete.append(path[-1])
        path = np.array(path_complete)
        logger.debug(f"Initial path: {path}")
        
        # remove links when the same link is passed twice within overlap_thresh sec
        passed_links: dict[int, int] = dict()  # {link_id: prev_k}
        for k in range(len(path)):
            lid = path[k]
            if lid in passed_links:
                prev_k = passed_links[lid]
                if k - prev_k < overlap_threshold:
                    for tmp_k in range(prev_k, k):
                        if path[tmp_k] in passed_links:
                            passed_links.pop(path[tmp_k])  # removing the link from the passed_links is ok, because the link can be added to the path afterwards (not passed or passed long enough time later)
                    path[prev_k:k] = lid
                passed_links[lid] = prev_k
            else:
                passed_links[lid] = k

        # remove circle path
        if not allow_circle:
            onode = self.network.link_start[path[0]]
            passed_nodes = {onode: 0}  # {node_id: k}
            for k in range(1, len(path)):
                tmp_onode = self.network.link_start[path[k]]
                if tmp_onode in passed_nodes:
                    prev_k = passed_nodes[tmp_onode]
                    for tmp_k in range(prev_k, k):
                        passed_onode = self.network.link_start[path[tmp_k]]
                        if passed_onode in passed_nodes:
                            passed_nodes.pop(passed_onode)
                    path[prev_k:k] = path[k]
                    passed_nodes[tmp_onode] = prev_k
                else:
                    passed_nodes[tmp_onode] = k

        # remove overlapped links
        path = [path[i] for i in range(len(path) - 1) if path[i] != path[i + 1]] + (
            [path[-1]] if len(path) > 0 else [])
        if len(path) > 1:
            path_no_u_turn = [path[0]]
            for i in range(1, len(path)):
                if len(path_no_u_turn) == 0:
                    path_no_u_turn.append(path[i])
                elif {self.network.link_start[path_no_u_turn[-1]], self.network.link_end[path_no_u_turn[-1]]} != {self.network.link_start[path[i]], self.network.link_end[path[i]]}:  # set of od nodes are different
                    path_no_u_turn.append(path[i])
            path = path_no_u_turn

        # convert link idxs to link ids
        path = [self.network.link_list[lidx] for lidx in path]
        return path

    def _perform_viterbi(self, trip: dd.DataFrame, emission_prob: np.ndarray, transition_prob: np.ndarray) -> list[int]:
        """
        Perform Viterbi algorithm for the given feeder and near links.

        Args:
            trip (dd.DataFrame): Trip data for mapmatching
            emission_prob (np.ndarray): Emission probabilities for the trip data. Shape: (trip length, link_num)
            transition_prob (np.ndarray): Transition probabilities for the trip data. Shape: (link_num, link_num)

        Returns:
            path (List[int]): list of link ids
        """
        near_link_idxs = emission_prob.sum(axis=0) > 0.0
        if np.sum(near_link_idxs) == 0:
            logger.warning("Mapmatching: no near links are found for the given trip.")
            return []
        viterbi_idx2link_idx = {i: j for i, j in enumerate(np.where(near_link_idxs)[0])}

        # set transition_prob
        transition_prob = transition_prob[near_link_idxs, :][:, near_link_idxs]
        prop_sum = np.sum(transition_prob, axis=1, keepdims=True)
        prop_sum[prop_sum == 0] = 1.0
        transition_prob = transition_prob / prop_sum

        # perform forward function by one by
        start_prob = emission_prob[0, near_link_idxs]

        viterbi = Viterbi(near_link_idxs.sum())
        viterbi.initialize(start_prob)
        for i in range(1, len(trip)):
            viterbi.forward(transition_prob, emission_prob[i, near_link_idxs])

        path = viterbi.get_path()[0]
        path = [viterbi_idx2link_idx[i] for i in path]

        return path
        
    @staticmethod
    def get_prior_transition_prob(network: BLENetwork) ->  np.ndarray:
        """
        Get prior transition probability for the given network.

        Args:
            network (Network): Network object

        Returns:
            np.ndarray: prior transition probabilities for all modes
        """
        transition_prob = np.full((network.n_link, network.n_link), 0.0001, dtype=float)
        lid2idx = {lid: i for i, lid in enumerate(network.link_list)}

        for link_idx, link_id in enumerate(network.link_list):
            link_center = network.link_center[link_idx]
            for down_link_idx in network.link_adj_matrix.getrow(link_idx).indices:
                eug_dist = np.linalg.norm(link_center - network.link_center[down_link_idx])
                path_dist = network.get_shortest_path(network.link_start[link_idx], network.link_end[down_link_idx])[0]
                if path_dist == 0.0:
                    transition_prob[link_idx, down_link_idx] = 50.0
                elif path_dist is None:
                    transition_prob[link_idx, down_link_idx] = eug_dist / 10000.
                else:
                    transition_prob[link_idx, down_link_idx] = eug_dist / path_dist
            transition_prob[link_idx, link_idx] = 100.0

        # normalize
        row_sums = transition_prob.sum(axis=1, keepdims=True)
        transition_prob = transition_prob / (row_sums + (row_sums == 0.0))
        return transition_prob
    
    @staticmethod
    def path2kab(path: list[int]) -> list[tuple[int, int, int]]:
        """
        Convert a sequence of link ids into (k, a, b) tuples.

        Args:
            path (list[int]): Ordered link ids representing a matched path.

        Returns:
            list[tuple[int, int, int]]: For each transition i -> i+1, returns
            a tuple `(k, a, b)` where `k` is the current link id, `a` the next link id, and `b` the destination link id of the path.
        """
        kad_path = []
        for i in range(len(path) - 1):
            kad_path.append((path[i], path[i+1], path[-1]))
        return kad_path
    
    @staticmethod
    def link_transition_probability_by_model(model: RouteChoiceModel, params: np.ndarray, network: BLENetwork, transition_prob: np.ndarray) -> np.ndarray:
        """
        Calculate the link transition probabilities for the given network.

        Args:
            model (RouteChoiceModel): Route choice model
            params (np.ndarray): Model parameters
            network (Network): Network object
            transition_prob (np.ndarray): Transition probabilities for all links

        Returns:
            np.ndarray: Link transition probabilities for all links
        """
        o_link_idx = np.argmax(transition_prob[0, :])
        d_link_idx = np.argmax(transition_prob[-1, :])
        link_transition = LinkTransition(
            trip_id = 1,
            link_id = network.link_list[o_link_idx],
            next_link_id = None,
            destination_node_id = network.link_end[d_link_idx],
            down_link_ids = [],
            model = model,
        )
        link_transition_prob = model.calculate_transition_probabilities(link_transition, params)

        # normalize
        link_transition_prob = link_transition_prob / (link_transition_prob.sum() + (link_transition_prob.sum() == 0.0))

        return link_transition_prob

