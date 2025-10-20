from typing import Optional, Union, cast
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import shapely.geometry

__all__ = ["NX2GDF"]


class NX2GDF:
    def __init__(self, G: nx.MultiDiGraph, tolerance: float = 5.0) -> None:
        """
        Convert a networkx MultiDiGraph to geopandas DataFrames for nodes and edges.

        Args:
            G (nx.MultiDiGraph): The input networkx MultiDiGraph.
            tolerance (float): Tolerance for simplifying geometries, default is 5.0 meters.

        Attributes:
            G (nx.MultiDiGraph): The input graph.
            tolerance (float): Tolerance for geometry simplification.
            epsg (int): EPSG code for the UTM zone.
            nodes (gpd.GeoDataFrame): GeoDataFrame containing node geometries. Geometries are points.
            edges (gpd.GeoDataFrame): GeoDataFrame containing edge geometries. Geometries are simplified and divided into LineStrings.
            no_exist_edges (gpd.GeoDataFrame): GeoDataFrame for edges that do not exist in the original graph.
        """
        self.G = G
        self.tolerance = tolerance
        self.nodes, self.edges = ox.graph_to_gdfs(G)
        self.edges = self.edges.reset_index()

        lon = cast(shapely.geometry.Point, self.nodes.geometry.iloc[0]).x
        if lon < 0:
            lon += 360
        zone_num = int((lon + 180) / 6) + 1
        self.epsg = zone_num + 32600  # UTM zone EPSG

        # simplify geometry
        self.edges.set_crs(epsg=4326, inplace=True)
        self.edges.to_crs(epsg=self.epsg, inplace=True)
        self.edges["geometry"] = self.edges["geometry"].simplify(tolerance=tolerance)
        self.edges.to_crs(epsg=4326, inplace=True)

        self._add_edge_info()
        self._divide_MultiLineString()
        self._divide_LineString()
        self._create_bidirection()

        self.nodes.set_crs(epsg=4326, inplace=True)
        self.edges.set_crs(epsg=4326, inplace=True)
        self.no_exist_edges.set_crs(epsg=4326, inplace=True)


    def save(self, node_file: str, edge_file: str, no_exist_edge_file: Optional[str] = None) -> None:
        """
        Save the GeoDataFrames to file.

        Args:
            node_file (str): Path to save the nodes GeoDataFrame.
            edge_file (str): Path to save the edges GeoDataFrame.
            no_exist_edge_file (Optional[str]): Path to save the no exist edges GeoDataFrame, if any.
        
        Returns:
            None
        """
        self.nodes.to_file(node_file)
        self.edges.to_file(edge_file)
        if no_exist_edge_file is not None:
            self.no_exist_edges.to_file(no_exist_edge_file)

    def to_epsg(self, epsg: int) -> tuple:
        """
        Convert the GeoDataFrames to a specified EPSG code.

        Args:
            epsg (int): The EPSG code to convert to.

        Returns:
            tuple: A tuple containing the nodes, edges, and no exist edges GeoDataFrames in the specified EPSG.
        """
        return self.nodes.to_crs(epsg=epsg, inplace=False), self.edges.to_crs(epsg=epsg, inplace=False), self.no_exist_edges.to_crs(epsg=epsg, inplace=False)

    def _add_edge_info(self) -> None:
        """
        Add additional information to the edges GeoDataFrame, such as 'lanes', and 'maxspeed'.
        """
        # lanes, highway
        if "lanes" not in self.edges.columns:
            self.edges["lanes"] = 1
        else:
            self.edges["lanes"] = self.edges["lanes"].apply(NX2GDF.get_lane)
        self.edges["maxspeed"] = self.edges["highway"].apply(NX2GDF.get_maxspeed)

    def _divide_MultiLineString(self) -> None:
        """
        Divide MultiLineString geometries into individual LineStrings in the edges GeoDataFrame.
        This method modifies the edges GeoDataFrame in place.
        """
        cols = self.edges.columns.tolist()
        geom_idx = cols.index("geometry")
        cols.remove("geometry")
        val = self.edges.to_numpy()
        geoms = self.edges["geometry"].to_numpy()
        append_val = None
        for i in range(len(self.edges)):
            tmp_geom = geoms[i]
            if type(tmp_geom) is shapely.geometry.MultiLineString:
                val[i, geom_idx] = tmp_geom.geoms[0]
                if len(tmp_geom.geoms) == 1:
                    continue
                tmp_val = np.repeat(val[[i], :], len(tmp_geom.geoms) - 1, axis=0)
                for j in range(1, len(tmp_geom.geoms)):
                    tmp_val[j-1, geom_idx] = tmp_geom.geoms[j]
                if append_val is None:
                    append_val = tmp_val
                else:
                    append_val = np.concatenate((append_val, tmp_val), axis=0)
        if append_val is not None:
            val = np.concatenate((val, append_val), axis=0)
        self.edges = gpd.GeoDataFrame(val, columns=self.edges.columns)

    def _divide_LineString(self) -> None:
        """
        Divide LineString geometries into segments in the edges GeoDataFrame.
        This method modifies the edges GeoDataFrame in place.
        """
        cols = self.edges.columns.tolist()
        geom_idx = cols.index("geometry")
        cols.remove("geometry")
        val = self.edges.to_numpy()
        geoms = self.edges["geometry"].to_numpy()
        append_val = None
        for i in range(len(self.edges)):
            tmp_geom = geoms[i]
            geom_list = [shapely.geometry.LineString(tmp_geom.coords[j:j+2]) for j in range(len(tmp_geom.coords)-1)]
            val[i, geom_idx] = geom_list[0]
            if len(geom_list) == 1:
                continue
            tmp_val = np.repeat(val[[i], :], len(geom_list)-1, axis=0)
            for j in range(1, len(geom_list)):
                tmp_val[j-1, geom_idx] = geom_list[j]
            if append_val is None:
                append_val = tmp_val
            else:
                append_val = np.concatenate((append_val, tmp_val), axis=0)
        if append_val is not None:
            val = np.concatenate((val, append_val), axis=0)
        self.edges =  gpd.GeoDataFrame(val, columns=self.edges.columns)

    def _create_bidirection(self) -> None:
        """
        Create bidirectional edges in the edges GeoDataFrame.
        This method modifies the edges and no_exist_edges GeoDataFrames in place.
        """
        cols = self.edges.columns.tolist()
        geom_idx = cols.index("geometry")
        cols.remove("geometry")
        val = self.edges.to_numpy()
        geoms = self.edges["geometry"].to_numpy()
        reversed_edge = self.edges["reversed"].to_numpy()
        motorway = np.array(self.edges["highway"].to_numpy() == "motorway")
        oneway = self.edges["oneway"].to_numpy()
        append_val = None
        no_exist_val = None  # opposite direction of oneway
        for i in range(len(self.edges)):
            tmp_geom = geoms[i]
            geoms[i] = shapely.geometry.LineString(tmp_geom.coords[::-1])  # reversed geometry
            if reversed_edge[i]:
                val[i, geom_idx] = geoms[i]
            if (not oneway[i]) and (not motorway[i]):
                tmp_val = val[[i], :]
                tmp_val[0, geom_idx] = geoms[i]
                if append_val is None:
                    append_val = tmp_val
                else:
                    append_val = np.concatenate((append_val, tmp_val), axis=0)
            else:
                tmp_val = val[[i], :]
                tmp_val[0, geom_idx] = geoms[i]
                if no_exist_val is None:
                    no_exist_val = tmp_val
                else:
                    no_exist_val = np.concatenate((no_exist_val, tmp_val), axis=0)
        if append_val is not None:
            val = np.concatenate((val, append_val), axis=0)
        self.edges = gpd.GeoDataFrame(val, columns=self.edges.columns)
        self.no_exist_edges = gpd.GeoDataFrame(no_exist_val, columns=self.edges.columns)
        # remove duplicated
        self.edges = self.edges[~self.edges["geometry"].duplicated()]
        all_edges = pd.concat([self.edges["geometry"], self.no_exist_edges["geometry"]], axis=0)
        idx = ~all_edges.duplicated().to_numpy()[len(self.edges):]
        self.no_exist_edges = self.no_exist_edges[idx]

    @staticmethod
    def get_lane(lane: int | str) -> int:
        """
        Convert lane information to an integer. If the input is a string, it attempts to convert it to an integer.
        If conversion fails, it defaults to 1.

        Args:
            lane (int | str): The lane information which can be an integer or a string.

        Returns:
            int: The number of lanes as an integer.
        """
        try:
            return int(lane)
        except (ValueError, TypeError):
            return 1

    @staticmethod
    def get_maxspeed(highway: str) -> Union[int, float]:
        """
        Get the maximum speed for a given highway type. If the highway type is not recognized, it defaults to 20.

        Args:
            highway (str): The highway type as a string.

        Returns:
            int: The maximum speed in km/h for the given highway type.
        """
        highway_dict = {
            "motorway": 80,
            "motorway_link": 40,
            "trunk": 50,
            "trunk_link": 40,
            "primary": 50,
            "primary_link": 40,
            "secondary": 50,
            "secondary_link": 40,
            "tertiary": 40,
            "tertiary_link": 40,
            "unclassified": 40,
            "road": 30,
            "residential": 30,
            "living_street": 20,
            "service": 10
        }
        if isinstance(highway, str):
            if highway in highway_dict:
                return highway_dict[highway]
            else:
                print(highway)
                return 20
        else:
            speed = [NX2GDF.get_maxspeed(h) for h in highway]
            return float(np.mean(speed))
