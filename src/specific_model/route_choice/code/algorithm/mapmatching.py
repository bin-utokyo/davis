from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Optional, cast
from abc import ABC, abstractmethod
import os
from logging import getLogger, StreamHandler, Formatter
import tqdm
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import shapely
from shapely import Point, MultiPoint, LineString, STRtree

sys.path.append(str(Path(__file__).resolve().parent.parent))
from definition import PP, Feeder, Network
from algorithm import heron_vertex, Viterbi

__all__ = ["MapmatchingABC", "HybridMapmatching"]

# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"
logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)


class MapmatchingABC(ABC):
    """Abstract base class for map matching algorithms.

    Subclasses implement concrete matching strategies (e.g., greedy DP,
    HMM/Viterbi). The interface standardizes how a preprocessed trajectory
    (`PP`) is converted into a sequence of link transitions.
    """
    @abstractmethod
    def match(self, pp: PP, overlap_threshold: int = 5, allow_circle: bool = False, max_trip: int | None = None) -> pd.DataFrame:
        """Run map matching over all trips/feeders in a `PP`.

        Args:
            pp (PP): Preprocessed trajectory container. Each trip may contain
                multiple `Feeder`s; each feeder is matched independently.
            overlap_threshold (int): Window (in GPS index steps) to collapse
                repeated visits to the same link. If a link reappears within
                this window, intermediate links are collapsed into that link.
            allow_circle (bool): If False, removes circular subpaths by
                collapsing loops detected at repeated start nodes.
            max_trip (int | None): Limit on number of trips to process. If
                None, all trips in `pp` are processed.

        Returns:
            pd.DataFrame: Link transitions for all matched feeders with columns:
                - `TripID`: trip identifier
                - `LinkID`: current link id
                - `NextLinkID`: next link id
                - `DestinationNodeID`: destination (last) link id in the path
                - `Purpose`: trip purpose carried from the source data
        """
        pass


class HybridMapmatching(MapmatchingABC):
    """Hybrid matcher combining segmentation with HMM/Viterbi.

    The feeder is segmented when a GPS point has a single unambiguous nearby
    link candidate. Each segment is matched via Viterbi on the reduced set of
    candidates, improving robustness and performance for long traces with
    intermittent unambiguous points.
    """
    def __init__(self, network: Network, buffer_size: float, modes: list[str], gps_error: float = 10.0):
        """
        Initialize HybridMapmatching.
        Args:
            network (Network): Network obj to perform mapmatching
            buffer_size (float): buffer size for mapmatching in meters
            modes (list[str]): list of mode codes (pp) to use for mapmatching
            gps_error (float): GPS error in meters
        """
        self.network = network
        self.buffer_size = buffer_size
        self.modes = modes

        self.linkidxs: list[list[int] | None] = list()  # index of links in network that can be passed by each mode
        self.trees: list[STRtree | None] = list()  # STRtree for each mode
        self._set_trees()  # set linkidxs and trees

        self.buffers: list[Optional[shapely.geometry.Polygon]] = list()  # buffer for each mode
        self._set_buffers()
        self.gps_error = gps_error

        # set transition probabilities for HMM
        self.transition_prob: np.ndarray = self.get_prior_transition_prob(self.network)

    def match(self, pp: PP, overlap_threshold: int = 5, allow_circle: bool = False, max_trip: int | None = None) -> pd.DataFrame:
        """Run map matching across trips/feeders and emit kab-form transitions.

        This iterates trips in `pp`, matches each eligible feeder, converts
        the matched link path into (k,a,b) tuples via :meth:`path2kab`, and
        returns a consolidated DataFrame of transitions.

        Args:
            pp (PP): Preprocessed trajectory container.
            overlap_threshold (int): Window to collapse repeated visits to the
                same link. See :meth:`match_one_feeder` for details.
            allow_circle (bool): If False, loops are removed. See
                :meth:`match_one_feeder`.
            max_trip (int | None): Optional limit on number of trips.

        Returns:
            pd.DataFrame: DataFrame with columns [TripID, LinkID, NextLinkID, DestinationLinkID, DestinationNodeID, Purpose]. One row per link transition.
        """

        result = []
        if max_trip is None:
            max_trip = len(pp.trips)
        for trip in tqdm.tqdm(pp.trips[:max_trip], desc="Trips"):
            for feeder in trip.feeders:
                if feeder.transport_mode in self.modes:
                    path = self.match_one_feeder(feeder, overlap_threshold, allow_circle)
                    if path is None:
                        logger.warning(f"Mapmatching: no path is found for feeder {feeder.id} (mode: {feeder.transport_mode})")
                        continue
                    kab_path = self.path2kab(path)
                    if len(kab_path) == 0:
                        logger.warning(f"Mapmatching: no kab path is found for feeder {feeder.id} (mode: {feeder.transport_mode})")
                        continue
                    tmp_result = [[feeder.id, *kab_path[i], trip.purpose] for i in range(len(kab_path))]  # [feeder_id, tmp_link_id, next_link_id, last_link_id, purpose]
                    result.extend(tmp_result)
        if len(result) == 0:
            logger.warning("Mapmatching: no result is found.")
            return pd.DataFrame(columns=["TripID", "LinkID", "NextLinkID", "DestinationNodeID", "DestinationLinkID", "Purpose"])

        df_result = pd.DataFrame(result, columns=["TripID", "LinkID", "NextLinkID", "DestinationLinkID", "Purpose"])
        df_result["DestinationNodeID"] = df_result["DestinationLinkID"].apply(lambda x: self.network.link_end[self.network.link_id2idx[x]])
        logger.info(f"HybridMapmatching: {len(result)} link transitions are obtained.")
        return df_result

    def match_one_feeder(self, feeder: Feeder, overlap_threshold: int = 5, allow_circle: bool = False) -> list[int] | None:
        """
        Perform mapmatching for one feeder.

        Args:
            feeder (Feeder): Feeder data for mapmatching
            overlap_threshold (int): threshold to remove overlapped links
            allow_circle (bool): whether to allow circle path

        Returns:
            path (List[int]): list of link ids
        """
        if feeder.transport_mode not in self.modes:
            return None
        feeder = self._filter_by_buffer(feeder)
        if len(feeder) == 0 or feeder.gps_points is None or feeder.gps_times is None:
            return None

        near_link_subsets = self._get_near_link_subsets(feeder, self.tree)

        path = []  # list of link idxs
        for i, (feeder_subset, near_link_idxs_subset) in enumerate(near_link_subsets):
            path_subset = self._perform_viterbi(feeder_subset, near_link_idxs_subset)

            if len(path_subset) > 0:
                if i < len(near_link_subsets) - 1:
                    path.extend(path_subset[:-1])  # remove the last link to avoid duplication
                else:
                    path.extend(path_subset)
        logger.debug(f"feeder length: {len(feeder)}, path length: {len(path)}")
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
            if feeder.gps_times is None:
                logger.warning(f"Mapmatching: no gps times are found for feeder {feeder.id} (mode: {feeder.transport_mode})")
                return None
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
    
    def _get_emission_prob(self, gps: tuple[float, float], near_links: list[int], target_link_idx: list[int]| None = None) -> np.ndarray:
        """
        Get emission probability for the given GPS point and near links.

        Args:
            gps (tuple[float, float]): GPS point (x, y)
            near_links (list[int]]): list of near link idxs
            target_link_idx (list[int] | None): list of target link indices to calculate emission probabilities. If None, all near links are used.
        Returns:
            np.ndarray: emission probabilities for the near links
        """
        emission_prob = np.zeros(len(near_links), dtype=np.float32)
        if target_link_idx is None:
            target_link_idx = list(range(len(near_links)))
        for i in target_link_idx:
            link_idx = near_links[i]
            dist = self.get_point_line_dist(gps,
                self.network.node_xy[self.network.node_id2idx[self.network.link_start[link_idx]]],
                self.network.node_xy[self.network.node_id2idx[self.network.link_end[link_idx]]])
            emission_prob[i] = np.exp(-dist ** 2 / (2 * self.gps_error ** 2))
        emission_prob = emission_prob / np.sum(emission_prob)
        return emission_prob
    
    def _get_near_link_subsets(self, feeder: Feeder, tree: STRtree) -> list[tuple[Feeder, list[np.ndarray]]]:
        """
        Get subsets of near links for the given feeder. Split the feeder when single link is matched.

        Args:
            feeder (Feeder): Feeder data for mapmatching
            tree (STRtree): STRtree for the mode of the feeder
        Returns:
            list[tuple[Feeder, list[np.ndarray]]]: list of tuples of (feeder subset, list of near link indices for each gps point in the subset)
        """
        result = []  # list of (feeder, near_link_idxs)
        
        near_link_idxs_all = [tree.query(Point(feeder[i][1]).buffer(self.buffer_size)) for i in range(len(feeder))]

        feeder_remain = feeder
        near_link_idxs_tmp = []
        for i in range(len(near_link_idxs_all)):
            if len(near_link_idxs_all[i]) == 0:
                raise ValueError(f"Mapmatching: no near link is found for gps point {i} in feeder {feeder.id}")
            elif len(near_link_idxs_all[i]) == 1:
                near_link_idxs_tmp.append(near_link_idxs_all[i])
                feeder_pre, feeder_remain = feeder_remain.split(feeder[i][0])
                result.append((feeder_pre, near_link_idxs_tmp))

                near_link_idxs_tmp = [near_link_idxs_all[i]]
            else:
                near_link_idxs_tmp.append(near_link_idxs_all[i])
        if len(feeder_remain) > 0:
            result.append((feeder_remain, near_link_idxs_tmp))
        return result

    def _perform_viterbi(self, feeder: Feeder, near_link_idxs_all: list[np.ndarray]) -> list[int]:
        """
        Perform Viterbi algorithm for the given feeder and near links.

        Args:
            feeder (Feeder): Feeder data for mapmatching
            near_link_idxs_all (list[np.ndarray]): list of near link indices for each gps point in the feeder

        Returns:
            path (List[int]): list of link ids
        """
        near_link_idxs = np.unique(np.concatenate(near_link_idxs_all)).tolist()
        near_link_idxs_array = np.array(near_link_idxs)
        idx2near_link_idx = {i: j for j, i in enumerate(near_link_idxs)}

        # set transition_prob
        transition_prob = self.transition_prob[near_link_idxs_array[:, np.newaxis], near_link_idxs_array]
        prop_sum = np.sum(transition_prob, axis=1, keepdims=True)
        prop_sum[prop_sum == 0] = 1.0
        transition_prob = transition_prob / prop_sum

        # perform forward function by one by
        target_link_idx = [idx2near_link_idx[i] for i in near_link_idxs_all[0]]
        start_prob = self._get_emission_prob(feeder[0][1], near_link_idxs, target_link_idx=target_link_idx)

        viterbi = Viterbi(len(near_link_idxs))
        viterbi.initialize(start_prob)
        for i in range(1, len(feeder)):
            target_link_idx = [idx2near_link_idx[j] for j in near_link_idxs_all[i]]
            emission_prob = self._get_emission_prob(feeder[i][1], near_link_idxs, target_link_idx=target_link_idx)
            viterbi.forward(transition_prob, emission_prob)

        path = viterbi.get_path()[0]
        path = [near_link_idxs[i] for i in path]

        return path
    
    def _set_trees(self) -> None:
        """
        Build per-mode shapely STRtree of passable link geometries.
        """
        self.tree = STRtree([LineString([
            self.network.node_xy[self.network.node_id2idx[self.network.link_start[i]]], 
            self.network.node_xy[self.network.node_id2idx[self.network.link_end[i]]]]) 
            for i in range(self.network.n_link)])

    def _set_buffers(self) -> None:
        """
        Build a buffer polygon for filtering GPS points.
        """
        link_linestring = shapely.geometry.MultiLineString([[
            self.network.node_xy[self.network.node_id2idx[self.network.link_start[i]]], 
            self.network.node_xy[self.network.node_id2idx[self.network.link_end[i]]]] 
            for i in range(self.network.n_link)])
        self.buffer = link_linestring.buffer(self.buffer_size)

    def _filter_by_buffer(self, feeder: Feeder) -> Feeder:
        """
        Crop a feeder's GPS points by the precomputed per-mode buffer.

        Args:
            feeder (Feeder): Feeder to be cropped.

        Returns:
            Feeder: Cropped feeder. If the feeder's mode is unknown or the
            buffer is missing, returns an empty feeder with same id/mode.
        """
        if feeder.transport_mode not in self.modes:
            return Feeder(feeder.id, feeder.transport_mode, np.array([]), np.array([]))
        return feeder.crop(cast(shapely.geometry.Polygon, self.buffer))
        
    @staticmethod
    def get_prior_transition_prob(network: Network) ->  np.ndarray:
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
    def get_point_line_dist(p: tuple[float, float], start_coord: tuple[float, float], end_coord: tuple[float, float]) -> float:
        """
        Compute Euclidean distance from a point to a line segment.

        The perpendicular distance to the infinite line is computed first via
        Heron's formula and then clamped to the segment endpoints when the
        projection lies outside the segment.

        Args:
            p (tuple[float, float]): Point coordinates (x, y).
            start_coord (tuple[float, float]): Start point of the line segment (x, y).
            end_coord (tuple[float, float]): End point of the line segment (x, y).

        Returns:
            float: Distance in the same units as the network coordinates.
        """
        p0 = np.array([p[0], p[1]])
        p1 = np.array([start_coord[0], start_coord[1]])
        p2 = np.array([end_coord[0], end_coord[1]])
        length = np.linalg.norm(p2 - p1)
        dist = heron_vertex(tuple(p0), tuple(p1), tuple(p2)) / length * 2.0
        l0 = np.sqrt(dist ** 2 + length ** 2)
        l1 = float(np.linalg.norm(p1 - p0))
        l2 = float(np.linalg.norm(p2 - p0))
        if l1 > l0:  # p0 is out of the link (end point side)
            dist = l2
        elif l2 > l0:  # p0 is out of the link (start point side)
            dist = l1
        if l1 > l0 and l2 > l0:
            dist = min(l1, l2)
        return float(dist)

