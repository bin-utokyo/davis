import os
from logging import getLogger, StreamHandler, Formatter
from typing import Optional

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, csr_array
from scipy.sparse.csgraph import floyd_warshall

import shapely
from pyproj import Proj

import osmnx as ox
import osm

__all__ = ["Network", "NetworkIO"]

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

    def __post_init__(self) -> None:
        self.n_node = len(self.node_table)
        self.n_link = len(self.link_table)
        self.node_list = self.node_table["NodeID"].to_numpy().tolist()
        self.link_list = self.link_table["LinkID"].to_numpy().tolist()
        self.link_start = self.link_table["ONodeID"].to_numpy().tolist()
        self.link_end = self.link_table["DNodeID"].to_numpy().tolist()
        self.od_node_id2link_id: dict[tuple[int, int], int] = {(self.link_start[i], self.link_end[i]): self.link_list[i] for i in range(self.n_link)}

        self.zone_num = Network.get_zone_num(self.node_table)

        self.f_name = [x for x in self.link_table.columns.tolist() if x not in ["LinkID", "ONodeID", "DNodeID"]]  # リンク属性の名前
        self.attr: dict[str, list[float]] = {x: self.link_table[x].to_numpy().tolist() for x in self.f_name}  # リンク属性の値

        self.node_id2idx: dict[int, int] = {nid: i for i, nid in enumerate(self.node_list)}  # key: ノードID, value: ノードindex in self.node_table
        self.link_id2idx: dict[int, int] = {lid: i for i, lid in enumerate(self.link_list)}  # key: リンクID, value: リンクindex in self.link_table

        self.node_xy = Network.get_node_xy(self.node_table)  # ノードの座標 (utm)
        self.link_length = Network.get_link_length(self.link_table, self.node_table)  # リンクの長さ
        self.link_center = Network.get_link_center(self.link_table, self.node_table)  # リンクの中心座標 (utm)

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

    def get_od_matrix(self, od_table: pd.DataFrame) -> tuple[csr_array, list[int]]:
        """
        Generate an origin-destination (OD) matrix from the given OD table.

        Args:
            od_table (pd.DataFrame): The OD table containing origin-destination pairs and their demand.

        Returns:
            tuple[csr_array, list[int]]: A tuple containing the OD matrix and a list of unique destination node indices.
        """
        od_matrix = lil_matrix((self.n_node, self.n_node), dtype=int)

        o_node_idxs = [self.node_id2idx.get(oz, -1) for oz in od_table["OriginNodeID"]]
        d_node_idxs = [self.node_id2idx.get(dz, -1) for dz in od_table["DestinationNodeID"]]
        demands = od_table["Demand"].to_numpy()

        for i in range(len(od_table)):
            if o_node_idxs[i] != -1 and d_node_idxs[i] != -1:
                od_matrix[o_node_idxs[i], d_node_idxs[i]] = od_matrix[o_node_idxs[i], d_node_idxs[i]] + demands[i]

        d_node_idxs_unique = sorted(list(set(d_node_idxs) - {-1}))

        return od_matrix.tocsr(), d_node_idxs_unique
    
    def get_shortest_path(self, o_node_id: int, d_node_id: int) -> tuple[float, list[int]]:
        """
        Get the shortest path length and the sequence of link ids from origin to destination.

        Args:
            o_node_id (int): Origin node ID.
            d_node_id (int): Destination node ID.

        Returns:
            tuple[float, list[int]]: A tuple containing the shortest path length and a list of link ids representing the path.
        """
        o_idx = self.node_id2idx[o_node_id]
        d_idx = self.node_id2idx[d_node_id]
        path_length = self.dist_matrix[o_idx, d_idx]

        path = []
        if np.isinf(path_length):
            return path_length, path

        # Reconstruct the shortest path using the predecessor matrix
        current_idx = d_idx
        while current_idx != o_idx:
            path.append(self.node_list[current_idx])
            current_idx = self.predecessor[o_idx, current_idx]
            if current_idx == -9999:
                # No path exists
                return float("inf"), []
        path.append(self.node_list[o_idx])
        path.reverse()  # Sequence of node IDs

        # Convert node path to link path
        link_path = []
        for i in range(len(path) - 1):
            link_id = self.od_node_id2link_id.get((path[i], path[i + 1]), None)
            if link_id is not None:
                link_path.append(link_id)
            else:
                # No link exists between these nodes
                return float("inf"), []

        return path_length, link_path

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
        link_adj_matrix = lil_matrix((self.n_link, self.n_link), dtype=float)
        for i in range(self.n_link):
            for j in self.down_link_idx[self.link_end[i]]:
                link_adj_matrix[i, j] = 0.5 * (self.link_length[i] + self.link_length[j])  # weight by average link length
        return link_adj_matrix.tocsr()
    
    def _incidence_matrix(self) -> csr_array:
        """
        Generate the incidence matrix for the network.

        Returns:
            csr_array: The incidence matrix.
        """
        incidence_matrix = lil_matrix((self.n_node, self.n_link), dtype=int)
        for i in range(self.n_link):
            incidence_matrix[self.node_id2idx[self.link_start[i]], i] = 1
            incidence_matrix[self.node_id2idx[self.link_end[i]], i] = -1
        return incidence_matrix.tocsr()

    def _get_shortest_path(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the shortest path lengths and predecessors between all pairs of nodes using the Floyd-Warshall algorithm.

        Returns:
            tuple(np.ndarray, np.ndarray): A tuple containing the distance matrix and predecessor matrix.
        """
        dist_matrix, predecessor = floyd_warshall(self.adj_matrix, return_predecessors=True)
        return dist_matrix, predecessor
    
    def _get_link_shortest_path(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the shortest path lengths and predecessors between all pairs of links using the Floyd-Warshall algorithm.

        Returns:
            tuple(np.ndarray, np.ndarray): A tuple containing the distance matrix and predecessor matrix for links.
        """
        dist_matrix, predecessor = floyd_warshall(self.link_adj_matrix, return_predecessors=True)
        return dist_matrix, predecessor
    
    @staticmethod
    def get_node_xy(node_table: pd.DataFrame) -> np.ndarray:
        """
        Get the node coordinates from the node table.

        Args:
            node_table (pd.DataFrame): The node table containing node attributes.

        Returns:
            np.ndarray: An array of node coordinates (x, y) in utm.
        """
        lon = node_table["Longitude"].to_numpy()
        lat = node_table["Latitude"].to_numpy()
        # Define UTM projection
        zone = Network.get_zone_num(node_table)
        utm_proj = Proj(proj="utm", zone=zone, ellps="WGS84")
        x, y = utm_proj(lon, lat)
        node_coords = np.vstack((x, y)).T
        return node_coords

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
        lon = node_table["Longitude"].to_numpy()
        lat = node_table["Latitude"].to_numpy()
        # Define UTM projection
        zone = Network.get_zone_num(node_table)
        utm_proj = Proj(proj="utm", zone=zone, ellps="WGS84")
        x, y = utm_proj(lon, lat)
        node_table_coords = pd.DataFrame({"NodeID": node_table["NodeID"], "X": x, "Y": y})
        node_table_coords.set_index("NodeID", inplace=True)

        # Calculate link lengths
        o_node = link_table["ONodeID"].to_numpy()
        d_node = link_table["DNodeID"].to_numpy()

        try:
            o_node_coords = node_table_coords.loc[o_node]
            d_node_coords = node_table_coords.loc[d_node]

            link_lengths = np.sqrt((d_node_coords["X"].to_numpy() - o_node_coords["X"].to_numpy()) ** 2 +
                                    (d_node_coords["Y"].to_numpy() - o_node_coords["Y"].to_numpy()) ** 2)
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
            np.ndarray: An array of link center coordinates (x, y) in utm.
        """
        # Get UTM coordinates from node table
        lon = node_table["Longitude"].to_numpy()
        lat = node_table["Latitude"].to_numpy()
        # Define UTM projection
        zone = Network.get_zone_num(node_table)
        utm_proj = Proj(proj="utm", zone=zone, ellps="WGS84")
        x, y = utm_proj(lon, lat)
        node_table_coords = pd.DataFrame({"NodeID": node_table["NodeID"], "X": x, "Y": y})
        node_table_coords.set_index("NodeID", inplace=True)

        # Calculate link center coordinates
        o_node = link_table["ONodeID"].to_numpy()
        d_node = link_table["DNodeID"].to_numpy()

        try:
            o_node_coords = node_table_coords.loc[o_node]
            d_node_coords = node_table_coords.loc[d_node]

            link_centers_x = (o_node_coords["X"].to_numpy() + d_node_coords["X"].to_numpy()) / 2
            link_centers_y = (o_node_coords["Y"].to_numpy() + d_node_coords["Y"].to_numpy()) / 2

            link_centers = np.vstack((link_centers_x, link_centers_y)).T
        except KeyError:
            raise ValueError("Node IDs in link table do not match those in node table.")

        return link_centers

    @staticmethod
    def check_attr(attr_dict: dict[str, list[float]], min_thresh: Optional[float] = None, max_thresh: Optional[float] = None) -> bool:
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
    
    @staticmethod
    def get_zone_num(node_table: pd.DataFrame) -> int:
        """
        Get the UTM zone number based on the longitude of the first node.

        Args:
            node_table (pd.DataFrame): The node table containing node attributes.

        Returns:
            int: The UTM zone number.
        """
        lon = node_table["Longitude"].to_numpy()[0]
        if lon < 0:
            lon += 360
        zone_num = int((lon + 180) / 6) + 1
        return zone_num


class NetworkIO:
    @staticmethod
    def get_from_osm(polygon_coord: list, node_path: str, link_path: str) -> None:
        """
        Get network data from OpenStreetMap within the specified polygon.

        Args:
            polygon_coord (list): [(lon, lat)] coordinates of the polygon to define the area
            node_path (str): Path to save the node data
            link_path (str): Path to save the link data

        Returns:
            None
        """
        bounding_box = shapely.geometry.Polygon(polygon_coord)
        filter = (
            '["highway"]["area"!~"yes"]'
            '["highway"!~"abandoned|bus_guideway|construction|cycleway|elevator|'
            'escalator|footway|no|planned|platform|proposed|raceway|razed|service|track"]'
            '["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
        )
        net = ox.graph.graph_from_polygon(bounding_box, simplify=True, retain_all=False, custom_filter=filter)
        gdfs = osm.NX2GDF(net, tolerance=5.0)
        _, edge_gdf, no_exist_gdf = gdfs.to_epsg(4326)

        # car availability
        ped_highway = {"bridleway", "corridor", "path", "pedestrian", "steps"}  # pedestrian only
        edge_gdf["car"] = True
        edge_gdf.loc[edge_gdf["highway"].isin(ped_highway), "car"] = False
        no_exist_gdf["car"] = False
        # ped availability
        edge_gdf["ped"] = edge_gdf["highway"] != "motorway"
        no_exist_gdf["ped"] = no_exist_gdf["highway"] != "motorway"
        no_exist_gdf = no_exist_gdf[no_exist_gdf["ped"]]
        edge_gdf["ped"] = True

        edge_gdf = pd.concat([edge_gdf, no_exist_gdf], axis=0)
        edge_gdf = edge_gdf[~edge_gdf["geometry"].duplicated()]
        edge_gdf["geometry"] = edge_gdf["geometry"].to_crs(epsg=4326)
        edge_gdf["id"] = np.arange(len(edge_gdf)) + 1

        node_set = set(sum([[*geom.coords] for geom in edge_gdf["geometry"]], []))
        nid2coord = {i+1: node for i, node in enumerate(node_set)}
        coord2nid = {v: k for k, v in nid2coord.items()}
        edge_gdf["start"] = edge_gdf["geometry"].apply(lambda x: coord2nid[x.coords[0]])
        edge_gdf["end"] = edge_gdf["geometry"].apply(lambda x: coord2nid[x.coords[-1]])

        node_df = pd.DataFrame({"NodeID": list(nid2coord.keys()),
                                "Longitude": [coord[0] for coord in nid2coord.values()],
                                "Latitude": [coord[1] for coord in nid2coord.values()]})
        node_df["NodeID"] = node_df["NodeID"].astype(int)

        # link properties
        cols = [col for col in edge_gdf.columns if col not in ["id", "start", "end", "car", "ped", "u", "v", "key", "osmid", "geometry"]]
        ## non-number columns
        cols_non_float = []
        for col in cols:
            try:
                converted = pd.to_numeric(edge_gdf[col], errors="coerce")
                if converted.isnull().any():
                    cols_non_float.append(col)
            except Exception:
                cols_non_float.append(col)

        org_cols = ["id", "start", "end", "car", "ped", *cols]
        cols = ["LinkID", "ONodeID", "DNodeID", "Car", "Ped", *cols]

        link_df = edge_gdf[org_cols].copy()
        link_df.columns = cols
        link_df.loc[:, ["LinkID", "ONodeID", "DNodeID"]] = link_df.loc[:, ["LinkID", "ONodeID", "DNodeID"]].astype(int)

        # coord
        link_df.loc[:, "OLon"] = link_df["ONodeID"].apply(lambda x: nid2coord[x][0])
        link_df.loc[:, "OLat"] = link_df["ONodeID"].apply(lambda x: nid2coord[x][1])
        link_df.loc[:, "DLon"] = link_df["DNodeID"].apply(lambda x: nid2coord[x][0])
        link_df.loc[:, "DLat"] = link_df["DNodeID"].apply(lambda x: nid2coord[x][1])

        node_df.to_csv(node_path, index=False)
        link_df.to_csv(link_path.replace(".csv", "_all.csv"), index=False)

        # remove non-number columns
        link_df = link_df.drop(columns=cols_non_float)
        link_df.to_csv(link_path, index=False)

