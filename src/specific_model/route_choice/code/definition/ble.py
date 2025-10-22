import os
import sys
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import dask.dataframe as dd
import pandas as pd
from scipy.stats import norm
from scipy.sparse import csr_matrix

from network import Network


# logger
loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
log_format = "[%(asctime)s] %(levelname)s:%(filename)s %(lineno)d:%(message)s"
logger = getLogger(__name__)
formatter = Formatter(log_format)
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(loglevel)

__all__ = ["BLE", "BLEPoint", "BLERecord", "BLEIO", "BLENetwork"]


@dataclass
class BLE:
    points: dict[int, "BLEPoint"]
    record: "BLERecord"

    _network: Optional["BLENetwork"] = None
    link_distance_matrix: Optional[dict[int, csr_matrix]] = None  # key: BLEPoint.id, value: csr_matrix row: link_idx, col1: min_dist, col2: max_dist. If distance is inf, set to 0

    def set_network(self, network: "BLENetwork") -> None:
        """Set the BLE network for this BLE data.

        Args:
            network (BLENetwork): The BLE network object.
        """
        self._network = network

        # Precompute link distance matrix for each BLE point
        self.link_distance_matrix = {}
        for point_id, point in self.points.items():
            distances = point.get_link_distances(network)
            distance_array = np.array(list(distances.values()))  # shape: (link_num, 2)
            distance_array[np.isinf(distance_array)] = 0.0  # Set inf to 0 for sparse matrix
            self.link_distance_matrix[point_id] = csr_matrix(distance_array)

    def get_emission_probability(self, trip_data: dd.DataFrame) -> np.ndarray:
        """Calculate the emission probabilities for the given trip data.

        Args:
            trip_data (dd.DataFrame): The trip data containing BLE records for a specific MAC address.

        Returns:
            np.ndarray: The emission probabilities for the given trip data. Shape: (trip length, link_num).
        """
        emission_prob = np.zeros((len(trip_data), self.network.n_link), dtype=np.float32)
        for i, (bleid, rssi) in enumerate(zip(trip_data["ID"], trip_data["RSSI"])):
            if self.link_distance_matrix is None or bleid not in self.link_distance_matrix:
                continue
            link_distances = self.link_distance_matrix[bleid].toarray()  # csr_matrix row: link_idx, col1: min_dist, col2: max_dist
            non_zero_indices = link_distances[:, 1] > 0.0  # Get indices where min_dist is not inf (stored as >0.0)

            tmp_emission_prob = BLEPoint.RSSI_to_probability(rssi, n=self.points[bleid].n, min_dist=link_distances[non_zero_indices, 0], max_dist=link_distances[non_zero_indices, 1])

            emission_prob[i, non_zero_indices] = tmp_emission_prob

        # Normalize
        row_sums = emission_prob.sum(axis=1, keepdims=True)
        emission_prob = emission_prob / np.where(row_sums == 0, 1, row_sums)
        emission_prob[row_sums.flatten() == 0, :] = 1.0 / self.network.n_link
        return emission_prob


    @property
    def network(self) -> "BLENetwork":
        """Get the BLE network associated with this BLE data.

        Returns:
            BLENetwork: The BLE network object.

        Raises:
            ValueError: If the network has not been set.
        """
        if self._network is None:
            raise ValueError("BLE network has not been set.")
        return self._network


@dataclass
class BLEPoint:
    id: int
    x: float
    y: float
    z: float

    n: float = 2.0  # RSSI path-loss exponent

    def get_link_distances(self, network: "BLENetwork") -> dict[int, tuple[float, float]]:
        """Calculate the distances from this BLE point to all links in the network.

        Args:
            network (BLENetwork): The BLE network object.
        """
        distances: dict[int, tuple[float, float]] = dict()
        for link_id in network.link_list:
            start_coord, end_coord = network.get_link_coords(link_id)
            min_dist, max_dist = BLEPoint.get_point_line_distance((self.x, self.y, self.z), start_coord, end_coord)
            distances[link_id] = (min_dist, max_dist)
        return distances

    @staticmethod
    def RSSI_to_distance(rssi: float | np.ndarray, tx_power: float | np.ndarray = -40, n: float = 2.0) -> float | np.ndarray:
        """Convert RSSI to distance using the log-distance path loss model.

        Args:
            rssi (float | np.ndarray): Received Signal Strength Indicator (in dBm).
            tx_power (float | np.ndarray): Transmit power (in dBm). Default is -40 dBm.
            n (float): Path-loss exponent. Default is 2.0 (free space).

        Returns:
            float | np.ndarray: Estimated distance (in meters).
        """
        rssi = np.asarray(rssi)
        tx_power = np.asarray(tx_power)
        rssi = np.clip(rssi, a_min=None, a_max=tx_power)  # Ensure rssi does not exceed tx_power
        
        ratio = (tx_power - rssi) / (10 * n)
        distance = 10 ** ratio
        return distance
    
    @staticmethod
    def RSSI_to_probability(rssi: float | np.ndarray, tx_power: float | np.ndarray = -40, n: float = 2.0, min_dist: float | np.ndarray = 0.0, max_dist: float | np.ndarray = 100.0, sigma: float | np.ndarray = 4.0) -> float | np.ndarray:
        """Convert RSSI to probability using a Gaussian model.

        Args:
            rssi (float | np.ndarray): Received Signal Strength Indicator (in dBm).
            tx_power (float | np.ndarray): Transmit power (in dBm). Default is -40 dBm.
            n (float): Path-loss exponent. Default is 2.0 (free space).
            min_dist (float | np.ndarray): Minimum distance (in meters). Default is 0.0 m.
            max_dist (float | np.ndarray): Maximum distance (in meters). Default is 100.0 m.
            sigma (float | np.ndarray): Standard deviation (in meters). Default is 4.0 m.

        Returns:
            float | np.ndarray: Probability density function value.
        """
        # Convert to np.ndarray
        min_dist = np.asarray(min_dist)
        max_dist = np.asarray(max_dist)

        # If min_dist and max_dist are invalid
        MAX_DIST = 200.0  # meters
        
        # Check invalid min_dist and max_dist
        min_dist[min_dist < 0.0] = 0.0
        max_dist[max_dist <= min_dist] = min_dist[max_dist <= min_dist]
        max_dist[max_dist > MAX_DIST] = MAX_DIST
        
        distance = BLEPoint.RSSI_to_distance(rssi, tx_power, n)

        # Accumulate probability within [min_dist, max_dist]
        prob = norm.cdf(max_dist, loc=distance, scale=sigma) - norm.cdf(min_dist, loc=distance, scale=sigma)

        return prob
    
    @staticmethod
    def get_point_line_distance(p: tuple[float, float, float], start_coord: tuple[float, float, float], end_coord: tuple[float, float, float]) -> tuple[float, float]:
        """Calculate the shortest distance from a point to a line segment in 3D space.

        Args:
            p (tuple[float, float, float]): The point coordinates (x, y, z).
            start_coord (tuple[float, float, float]): The start coordinates of the line segment (x, y, z).
            end_coord (tuple[float, float, float]): The end coordinates of the line segment (x, y, z).

        Returns:
            tuple[float, float]: The shortest distance from the point to the line segment and the longest distance from the point to the line segment.
        """
        # If z is different, we regard it unconnected
        THRESH  = 1.0  # meters
        if abs(start_coord[2] - end_coord[2]) > THRESH or abs(start_coord[2] - p[2]) > THRESH or abs(end_coord[2] - p[2]) > THRESH:
            return float('inf'), float('inf')
        
        # Vector AB (start to end)
        AB = np.array([end_coord[0] - start_coord[0],
                       end_coord[1] - start_coord[1],
                       end_coord[2] - start_coord[2]])
        # Vector AP
        AP = np.array([p[0] - start_coord[0],
                       p[1] - start_coord[1],
                       p[2] - start_coord[2]])

        AB_squared = np.dot(AB, AB)
        if AB_squared == 0:
            # A and B are the same point
            return float(np.linalg.norm(AP)), float(np.linalg.norm(AP))

        t = np.dot(AP, AB) / AB_squared
        t = max(0, min(1, t))  # Clamp t to the range [0, 1]

        # Projection point D
        D = start_coord[0] + t * AB[0], start_coord[1] + t * AB[1], start_coord[2] + t * AB[2]
        PD = np.array([p[0] - D[0],
                       p[1] - D[1],
                       p[2] - D[2]])
        
        if t < 0.5:
            Q = end_coord
        else:
            Q = start_coord
        PQ = np.array([p[0] - Q[0],
                       p[1] - Q[1],
                       p[2] - Q[2]])
        return float(np.linalg.norm(PD)), float(np.linalg.norm(PQ))


@dataclass
class BLERecord:
    table: dd.DataFrame

    unique_macs: Optional[list[str]] = None

    def __post_init__(self) -> None:
        # sort table by time
        self.table.sort_values(by="time", inplace=True)

        self.unique_macs = self.table["MAC"].unique().compute().tolist()
        if not self.unique_macs:
            self.unique_macs = []

        # Clean unique_macs observed by only one BLE logger
        unique_bleid_counts = [self.table[self.table["MAC"] == mac]["ID"].nunique().compute() for mac in self.unique_macs]
        self.unique_macs = [mac for mac, count in zip(self.unique_macs, unique_bleid_counts) if count > 1]


    def __len__(self) -> int:
        return len(self.unique_macs) if self.unique_macs is not None else 0

    def __getitem__(self, index: int | slice) -> dd.DataFrame:
        """Get the BLE records for a specific MAC address.

        Args:
            index (int): Index of the MAC address in the unique_macs list.

        Returns:
            dd.DataFrame: BLE records for the specified MAC address.
        """
        if self.unique_macs is None:
            raise ValueError("No MAC addresses available in the BLE record.")
        if isinstance(index, slice):
            macs = self.unique_macs[index]
            return self.table[self.table["MAC"].isin(macs)]
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")
        mac = self.unique_macs[index]
        return self.table[self.table["MAC"] == mac]

    
class BLEIO:
    @staticmethod
    def load_from_csv(signal_file: str, point_file: str) -> BLE:
        """Load BLE data from a CSV file. Only support ISO datetime format.
        
        Args:
            signal_file (str): Path to the signal file.
            point_file (str): Path to the BLE logger point file.

        Returns:
            BLE: Loaded BLE data.
        """
        required_columns_signal = ["ID", "MAC", "RSSI", "time"]  # BLE logger ID, MAC address, RSSI, timestamp
        required_columns_point = ["bleid", "x", "y", "z"]  # BLE logger ID, x, y, z coordinates

        df_signal = dd.read_csv(signal_file)
        df_point = pd.read_csv(point_file)

        df_signal["time"] = dd.to_datetime(df_signal["time"], format=None, errors='coerce')

        # Validate columns
        for col in required_columns_signal:
            if col not in df_signal.columns:
                raise ValueError(f"Missing required column '{col}' in signal file.")
        for col in required_columns_point:
            if col not in df_point.columns:
                raise ValueError(f"Missing required column '{col}' in point file.")
            
        # Load points
        points = dict()
        for bleid, x, y, z in zip(df_point["bleid"].to_numpy(), df_point["x"].to_numpy(), df_point["y"].to_numpy(), df_point["z"].to_numpy()):
            ble_point = BLEPoint(
                id=int(bleid),
                x=float(x),
                y=float(y),
                z=float(z)
            )
            points[ble_point.id] = ble_point

        # Create BLERecord
        record = BLERecord(table=df_signal)
        return BLE(points=points, record=record)


@dataclass
class BLENetwork(Network):
    """
    Extended Network class to handle BLE-specific network attributes and methods.

    Args:
        node_table (pd.DataFrame): DataFrame containing node information. Columns: ["nodeid", "x", "y", "z"]
        link_table (pd.DataFrame): DataFrame containing link information. Columns: ["linkid", "o", "d", ...other attributes]
    """
    def __post_init__(self) -> None:
        self.n_node = len(self.node_table)
        self.n_link = len(self.link_table)
        self.node_list = self.node_table["nodeid"].to_numpy().tolist()
        self.link_list = self.link_table["linkid"].to_numpy().tolist()
        self.link_start = self.link_table["o"].to_numpy().tolist()
        self.link_end = self.link_table["d"].to_numpy().tolist()
        self.od_node_id2link_id: dict[tuple[int, int], int] = {(self.link_start[i], self.link_end[i]): self.link_list[i] for i in range(self.n_link)}

        self.f_name = [x for x in self.link_table.columns.tolist() if x not in ["linkid", "o", "d"]]  # リンク属性の名前
        self.attr: dict[str, list[float]] = {x: self.link_table[x].to_numpy().tolist() for x in self.f_name}  # リンク属性の値

        self.node_id2idx: dict[int, int] = {nid: i for i, nid in enumerate(self.node_list)}  # key: ノードID, value: ノードindex in self.node_table
        self.link_id2idx: dict[int, int] = {lid: i for i, lid in enumerate(self.link_list)}  # key: リンクID, value: リンクindex in self.link_table

        self.link_length = self.get_link_length(self.link_table, self.node_table)  # リンクの長さ
        self.link_center = self.get_link_center(self.link_table, self.node_table)  # リンクの中心座標 (utm)

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
        self.incidence_matrix = self._incidence_matrix()
        self.dist_matrix, self.predecessor = self._get_shortest_path()
        self.link_dist_matrix, self.link_predecessor = self._get_link_shortest_path()

        if not Network.check_attr(self.attr, min_thresh=-10, max_thresh=10):
            logger.warning("Some link attributes are out of the expected range (-10, 10).")

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
        node_table_coords = pd.DataFrame({"nodeid": node_table["nodeid"], "x": node_table["x"], "y": node_table["y"], "z": node_table["z"]})
        node_table_coords.set_index("nodeid", inplace=True)

        # Calculate link lengths
        o_node = link_table["o"].to_numpy()
        d_node = link_table["d"].to_numpy()

        try:
            o_node_coords = node_table_coords.loc[o_node]
            d_node_coords = node_table_coords.loc[d_node]

            link_lengths = np.sqrt((d_node_coords["x"].to_numpy() - o_node_coords["x"].to_numpy()) ** 2 +
                                    (d_node_coords["y"].to_numpy() - o_node_coords["y"].to_numpy()) ** 2 +
                                    (d_node_coords["z"].to_numpy() - o_node_coords["z"].to_numpy()) ** 2)
        except KeyError:
            raise ValueError("Node IDs in link table do not match those in node table.")

        return link_lengths
    
    @staticmethod
    def get_link_center(link_table: pd.DataFrame, node_table: pd.DataFrame) -> np.ndarray:
        """
        Get the link center coordinates from the link table.

        Args:
            link_table (pd.DataFrame): The link table containing link attributes.
            node_table (pd.DataFrame): The node table containing node attributes.

        Returns:
            np.ndarray: An array of link center coordinates (x, y, z).
        """
        node_table_coords = pd.DataFrame({"nodeid": node_table["nodeid"], "x": node_table["x"], "y": node_table["y"], "z": node_table["z"]})
        node_table_coords.set_index("nodeid", inplace=True)

        # Calculate link center coordinates
        o_node = link_table["o"].to_numpy()
        d_node = link_table["d"].to_numpy()

        try:
            o_node_coords = node_table_coords.loc[o_node]
            d_node_coords = node_table_coords.loc[d_node]

            link_centers_x = (o_node_coords["x"].to_numpy() + d_node_coords["x"].to_numpy()) / 2
            link_centers_y = (o_node_coords["y"].to_numpy() + d_node_coords["y"].to_numpy()) / 2
            link_centers_z = (o_node_coords["z"].to_numpy() + d_node_coords["z"].to_numpy()) / 2

            link_centers = np.vstack((link_centers_x, link_centers_y, link_centers_z)).T
        except KeyError:
            raise ValueError("Node IDs in link table do not match those in node table.")

        return link_centers
    
    def get_link_coords(self, link_id: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get the start and end coordinates of a link.

        Args:
            link_id (int): The ID of the link.

        Returns:
            tuple[tuple[float, float, float], tuple[float, float, float]]: The start and end coordinates of the link.
        """
        o_node_idx = self.node_id2idx[self.link_start[self.link_id2idx[link_id]]]
        d_node_idx = self.node_id2idx[self.link_end[self.link_id2idx[link_id]]]
        start_coord = self.node_table[["x", "y", "z"]].iloc[o_node_idx].to_numpy()
        end_coord = self.node_table[["x", "y", "z"]].iloc[d_node_idx].to_numpy()
        return (tuple(start_coord), tuple(end_coord))

    # Unused methods
    @staticmethod
    def get_node_xy(node_table: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("This method is not implemented for BLE networks.")
    
    @staticmethod
    def get_zone_num(node_table: pd.DataFrame) -> int:
        raise NotImplementedError("This method is not implemented for BLE networks.")




if __name__ == "__main__":
    pass