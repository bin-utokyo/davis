import os
from logging import getLogger, StreamHandler, Formatter

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, csr_array
from pyproj import Proj

# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"
logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)

@dataclass
class Network:
    node_table: pd.DataFrame
    link_table: pd.DataFrame

    def __post_init__(self):
        self.n_node = len(self.node_table)
        self.n_link = len(self.link_table)
        self.node_list = self.node_table['NodeID'].values.tolist()
        self.link_list = self.link_table['LinkID'].values.tolist()
        self.link_start = self.link_table['ONodeID'].values.tolist()
        self.link_end = self.link_table['DNodeID'].values.tolist()

        self.f_name = [x for x in self.link_table.columns.tolist() if x not in ['LinkID', 'ONodeID', 'DNodeID']]  # リンク属性の名前
        self.attr: dict[str, list[float]] = {x: self.link_table[x].values.tolist() for x in self.f_name}  # リンク属性の値

        self.node_id2idx: dict[int, int] = {nid: i for i, nid in enumerate(self.node_list)}  # key: ノードID, value: ノードindex in self.node_table
        self.link_id2idx: dict[int, int] = {lid: i for i, lid in enumerate(self.link_list)}  # key: リンクID, value: リンクindex in self.link_table

        self.link_length = Network.get_link_length(self.link_table, self.node_table)  # リンクの長さ

        # down_link_idx  key: ノードID, value: 下流リンクindexのリスト
        self.down_link_idx: dict[int, list[int]] = {nid: [] for nid in self.node_list}
        for i in range(self.n_link):
            self.down_link_idx[self.link_start[i]].append(i)
        # up_link_idx  key: ノードID, value: 上流リンクindexのリスト
        self.up_link_idx: dict[int, list[int]] = {nid: [] for nid in self.node_list}
        for i in range(self.n_link):
            self.up_link_idx[self.link_end[i]].append(i)

        self.adj_matrix = self._adj_matrix()
        self.link_adj_matrix = self._link_adj_matrix()

        if not Network.check_attr(self.attr, min_thresh=-10, max_thresh=10):
            logger.warning("Some link attributes are out of the expected range (-10, 10).")

    def get_od_matrix(self, od_table: pd.DataFrame) -> tuple[csr_array, list[int]]:
        """
        Generate an origin-destination (OD) matrix from the given OD table.

        Args:
            od_table (pd.DataFrame): The OD table containing origin-destination pairs and their demand.

        Returns:
            tuple[csr_array, list[int]]: A tuple containing the OD matrix and a list of unique destination node indices.
        """
        od_matrix = lil_matrix((self.n_node, self.n_node), dtype=int)

        o_node_idxs = [self.node_id2idx.get(oz, -1) for oz in od_table['OriginNodeID']]
        d_node_idxs = [self.node_id2idx.get(dz, -1) for dz in od_table['DestinationNodeID']]
        demands = od_table['Demand'].values

        for i in range(len(od_table)):
            if o_node_idxs[i] != -1 and d_node_idxs[i] != -1:
                od_matrix[o_node_idxs[i], d_node_idxs[i]] = od_matrix[o_node_idxs[i], d_node_idxs[i]] + demands[i]

        d_node_idxs_unique = sorted(list(set(d_node_idxs) - {-1}))

        return od_matrix.tocsr(), d_node_idxs_unique

    def _adj_matrix(self) -> csr_array:
        """
        Generate the adjacency matrix for the network.

        Returns:
            csr_array: The adjacency matrix weighted by link lengths.
        """
        length_adj_matrix = lil_matrix((self.n_node, self.n_node), dtype=int)
        for i in range(self.n_link):
            length_adj_matrix[self.node_id2idx[self.link_start[i]], self.node_id2idx[self.link_end[i]]] = self.link_length[i]
        return length_adj_matrix.tocsr()

    def _link_adj_matrix(self) -> csr_array:
        """
        Generate the link adjacency matrix for the network.

        Returns:
            csr_array: The link adjacency matrix.
        """
        link_adj_matrix = lil_matrix((self.n_link, self.n_link), dtype=int)
        for i in range(self.n_link):
            link_adj_matrix[i, self.down_link_idx[self.link_end[i]]] = 1
        return link_adj_matrix.tocsr()

    @staticmethod
    def get_shortest_path(ori: int, des: int, prev_mat: np.ndarray) -> list[int]:
        # ポインタを逆にたどって最短経路を出力
        path = []
        p_app = path.append
        v = des - 1

        while v != ori - 1 and v >= 0:
            p_app(v + 1)
            v = prev_mat[ori - 1, v]

        if v < 0:
            sp = []
        else:
            p_app(v + 1)
            sp = list(reversed(path))

        return sp
    
    @staticmethod
    def get_link_length(link_table: pd.DataFrame, node_table: pd.DataFrame) -> np.ndarray:
        """
        Get the link lengths from the link table.

        Args:
            link_table (pd.DataFrame): The link table containing link attributes.
            node_table (pd.DataFrame): The node table containing node attributes.

        Returns:
            np.ndarray: An array of link lengths.
        """
        # Get UTM coordinates from node table
        lon = node_table["Longitude"].values
        lat = node_table["Latitude"].values
        # Define UTM projection
        zone = int(lon[0] // 6) + 31  # UTM zone calculation based on longitude
        utm_proj = Proj(proj="utm", zone=zone, ellps="WGS84")
        x, y = utm_proj(lon, lat)
        node_table_coords = pd.DataFrame({"NodeID": node_table["NodeID"], "X": x, "Y": y})
        node_table_coords.set_index("NodeID", inplace=True)

        # Calculate link lengths
        o_node = link_table["ONodeID"].values
        d_node = link_table["DNodeID"].values

        try:
            o_node_coords = node_table_coords.loc[o_node]
            d_node_coords = node_table_coords.loc[d_node]

            link_lengths = np.sqrt((d_node_coords["X"].values - o_node_coords["X"].values) ** 2 +
                                    (d_node_coords["Y"].values - o_node_coords["Y"].values) ** 2)
        except KeyError:
            raise ValueError("Node IDs in link table do not match those in node table.")

        return link_lengths
    
    @staticmethod
    def check_attr(attr_dict: dict[str, list[float]], min_thresh: float | None = None, max_thresh: float | None = None) -> bool:
        """
        Check if all attributes in the dictionary are within specified thresholds.

        Args:
            attr_dict (dict): Dictionary of attributes to check.
            min_thresh (float, optional): Minimum threshold for attribute values.
            max_thresh (float, optional): Maximum threshold for attribute values.

        Returns:
            bool: True if all attributes are within thresholds, False otherwise.
        """
        for key, value in attr_dict.items():
            if min_thresh is not None and min(value) < min_thresh:
                return False
            if max_thresh is not None and max(value) > max_thresh:
                return False
        return True
        
