'''Hongoのインプットを作る関数を記述'''
from typing import Optional
import numpy as np
import pandas as pd

import geopandas as gpd

import shapely
from shapely import ops
import networkx as nx
import osmnx as ox
from osmnx import graph as ox_graph
import osmnx.features as ox_features
import momepy
import copy

from scipy.spatial import KDTree


# Utilityで使われている関数のみを個別インポート
from Utility import calcAngle, calcAngleMinus180to180



class HongoInput:
    def __init__(self, input_dir: str, epsg: int, bb_coord: list[list[float]]):
        """
        Args:
            input_dir (str): 入力ディレクトリのパス
            epsg (int): EPSGコード
            bb_coord (list[list[float]]): バウンディングボックスの座標
        """
        '''
        parameters
        ----------
        bb_coord: array_like
            バウンディングボックスの座標
        osm_data: None or str
            osmを使う場合はpyrosmデータの範囲を表す文字列
        nx: None or networkx.Multigraph
            道路ネットワークをosm以外から与える場合のネットワーク
        '''
        self.input_dir = input_dir
        self.bb_coord = bb_coord
        self.epsg = epsg
        
        self.nodes: Optional[dict[int, tuple[float, float]]] = None  #key: node_id, value: (lon, lat)
        self.signals: Optional[dict[int, tuple[float, float]]] = None
        
        self.od_df: Optional[pd.DataFrame] = None
        
        self.hwy_speeds = {"motorway": 80, "motorway_link": 40, "trunk": 50, "trunk_link": 40, "primary": 50, "primary_link": 40, "secondary": 50, "secondary_link": 40, "tertiary": 40, "tertiary_link": 40, "unclassified": 40, "road": 30, "residential": 30, "living_street": 20, "service": 10}
        
            
    def create_network_input(self) -> None:
        """
        インプットファイル書き出し：link.csv,node.csv,connectivity.csv,signal.csv

        link.csv: ID,ONodeID,DNodeID,Velocity,LeftCarNum,LeftPedNum,LeftBeltNum,RightCarNum,RightPedNum,RightBeltNum
        node.csv: ID,Lon,Lat
        connectivity.csv: NodeID,UpLinkID,UpLaneID,DnLinkID,DnLaneID
        signal.csv: ID,NodeID,RightTurn
        Returns:
            None
        """
        '''
        インプットファイル書き出し：link.csv,node.csv,connectivity.csv,signal.csv
        link.csv
            ID,ONodeID,DNodeID,Velocity,LeftCarNum,LeftPedNum,LeftBeltNum,RightCarNum,RightPedNum,RightBeltNum
        node.csv
            ID,Lon,Lat
        connectivity.csv
            NodeID,UpLinkID,UpLaneID,DnLinkID,DnLaneID
        signal.csv
            ID,NodeID,RightTurn
        '''
        if self.nodes is None:
            print("Plaease read osm data first.")
            return
        node_dict: dict[tuple[float, float], int] = {v: k for k,v in self.nodes.items()}#key:(lon,lat), value:id
        link_dict: dict[int, tuple[int, int, int]] = {i: (node_dict[val[0]], node_dict[(val[1])], val[2]) for i, val in enumerate(list(self.drive_net.edges))}#key:id, value:(oNodeId, dNodeId)

        node_pair_dict: dict[tuple[int, int], list[Optional[int]]] = dict()  # key:(nodeId1, nodeId2), value:[linkId]
        for k, v in link_dict.items():
            if (v[1], v[0]) in node_pair_dict:
                node_pair_dict[(v[1], v[0])][1] = k
            else:
                node_pair_dict[(v[0], v[1])] = [k, None]
        
        dlink_dict: dict[int, list[Optional[int]]] = dict()  #key:nodeId, value:[linkId]
        for k, v in node_pair_dict.items():
            if k[0] in dlink_dict:
                dlink_dict[k[0]].append(v[0])
            else:
                dlink_dict[k[0]] = [v[0]]
            if v[1] is None:
                continue
            if k[1] in dlink_dict:
                dlink_dict[k[1]].append(v[1])
            else:
                dlink_dict[k[1]] = [v[1]]
                
        self.lane_dict = dict()#key:linkId, value:lane_num
        for i, li in enumerate(list(self.drive_net.edges)):
            self.lane_dict[i] = self.drive_net.edges[li]["lanes"]
            
        self.node_dict = node_dict
        self.link_dict = link_dict
        self.node_pair_dict = node_pair_dict
        self.dlink_dict = dlink_dict
            
        df_node = self._create_node()
                
        df_link = self._create_link(df_node)
        
        self._create_connectivity(df_link)
        
        self._create_signal(df_node)
        
    def _create_node(self, node_file: Optional[str] = None) -> pd.DataFrame:
        """
        ノード情報をCSVに出力し、DataFrameを返す。

        Args:
            node_file (Optional[str]): 出力ファイルパス。Noneの場合はデフォルトパス。
        Returns:
            pd.DataFrame: ノード情報のDataFrame。self.nodesがNoneの場合は空のDataFrame。
        """
        if self.nodes is None:
            print("Please read osm data first.")
            return pd.DataFrame()
        #node.csv
        values = [[k, v[0], v[1]] for k,v in self.nodes.items()]
        df_node = pd.DataFrame(values, columns=["ID","Lon","Lat"])
        df_node["ID"] = df_node["ID"].astype(int)
        df_node.sort_values("ID",inplace=True)
        if node_file is None:
            df_node.to_csv(self.input_dir+"node.csv", index=False)
        else:
            df_node.to_csv(node_file, index=False)
        print("node.csv is written.")
        return df_node
        
    def _create_link(self, df_node: pd.DataFrame, link_file: Optional[str] = None) -> pd.DataFrame:
        """
        リンク情報をCSVに出力し、DataFrameを返す。

        Args:
            df_node (pd.DataFrame): ノード情報のDataFrame
            link_file (Optional[str]): 出力ファイルパス。Noneの場合はデフォルトパス。
        Returns:
            pd.DataFrame: リンク情報のDataFrame。必要な属性がNoneの場合は空のDataFrame。
        """
        if self.nodes is None or self.node_pair_dict is None or self.link_dict is None:
            print("Please read osm data first.")
            return pd.DataFrame()
        
        #link.csv
        self.link_id_dict = dict()#key:link_id, value:csvのlink_id
        values = [[0 for i in range(10)] for j in range(len(self.node_pair_dict))]
        idx = 0
        for k, v in self.node_pair_dict.items():
            values[idx][0] = idx+1
            values[idx][1] = k[0]
            values[idx][2] = k[1]
            v_0 = v[0]
            if v_0 is None:
                print(f"Link {k} has no link ID.")
                continue
            values[idx][3] = int(self.drive_net.edges[(self.nodes[self.link_dict[v_0][0]], self.nodes[self.link_dict[v_0][1]], self.link_dict[v_0][2])]["speed"])#leftCarLinkの属性にアクセス
            values[idx][4] = self.lane_dict[v[0]] if v[0] is not None else 0
            values[idx][5] = 1
            if v[1] is not None:
                values[idx][7] = self.lane_dict[v[1]]
                values[idx][8] = 1
                
                self.link_id_dict[v[1]] = idx+1
            
            
            self.link_id_dict[v_0] = idx+1
            
            idx += 1
            
        df_link = pd.DataFrame(values, columns=["ID","ONodeID","DNodeID","Velocity","LeftCarNum","LeftPedNum","LeftBeltNum","RightCarNum","RightPedNum","RightBeltNum"])
        df_link[["ID","ONodeID","DNodeID","LeftCarNum","LeftPedNum","LeftBeltNum","RightCarNum","RightPedNum","RightBeltNum"]] = df_link[["ID","ONodeID","DNodeID","LeftCarNum","LeftPedNum","LeftBeltNum","RightCarNum","RightPedNum","RightBeltNum"]].astype(int)
        
        df_node.columns = ["ONodeID","OLon","OLat"]
        df_link = pd.merge(df_link, df_node, on="ONodeID")
        df_node.columns = ["DNodeID","DLon","DLat"]
        df_link = pd.merge(df_link, df_node, on="DNodeID")
        df_node.columns = ["ID","Lon","Lat"]
        
        df_link.sort_values("ID",inplace=True)
        if link_file is None:
            df_link.to_csv(self.input_dir+"link.csv", index=False)
        else:
            df_link.to_csv(link_file, index=False)
        print("link.csv is written.")
        return df_link
    
    def _create_connectivity(self, df_link: pd.DataFrame, connectivity_file: Optional[str] = None) -> pd.DataFrame:
        """
        コネクティビティ情報をCSVに出力し、DataFrameを返す。

        Args:
            df_link (pd.DataFrame): リンク情報のDataFrame
            connectivity_file (Optional[str]): 出力ファイルパス。Noneの場合はデフォルトパス。
        Returns:
            pd.DataFrame: コネクティビティ情報のDataFrame。
        """
        #connectivity.csv
        df_link["angle"] = np.vectorize(calcAngle)(df_link["OLon"], df_link["OLat"], df_link["DLon"], df_link["DLat"])
        df_link.set_index("ID", drop=False, inplace=True)
        
        values = []
        for od_nodes,link_ids in self.node_pair_dict.items():
            for j in range(2):
                olink_id = link_ids[j]
                if j == 0:
                    od = od_nodes#v:(oNodeId, dNodeId)
                else:
                    od = (od_nodes[1],od_nodes[0])
                if olink_id is None:
                    continue
                up_lane = self.lane_dict[olink_id]
                node_id = od[1]#下流ノード
                cross = 0#逆行以外の行き先が2つ以上あるか否か
                if od in self.node_pair_dict:
                    up_direction = 1#順方向
                else:
                    up_direction = -1#逆方向
                if node_id not in self.dlink_dict:
                    continue
                for dlink_id in self.dlink_dict[node_id]:
                    if dlink_id is None:
                        continue
                    dlink_od = (self.link_dict[dlink_id][0], self.link_dict[dlink_id][1])
                    if dlink_od == (od[1], od[0]):#逆行リンク
                        continue
                    cross += 1
                cross = (cross>1)*1#逆行以外の行き先が2つ以上
                for dlink_id in self.dlink_dict[node_id]:
                    if dlink_id is None:
                        continue
                    dlink_od = (self.link_dict[dlink_id][0], self.link_dict[dlink_id][1])
                    if dlink_od == (od[1], od[0]):#逆行リンク
                        continue
                    if dlink_od in self.node_pair_dict:
                        dn_direction = 1#順方向
                    else:
                        dn_direction = -1#逆方向
                    dn_lane = self.lane_dict[dlink_id]
                    tmp_angle = calcAngleMinus180to180(float(df_link.loc[self.link_id_dict[olink_id], "angle"]) + 180.0 * (up_direction == 1), float(df_link.loc[self.link_id_dict[dlink_id], "angle"]) + 180.0 * (dn_direction == 1))

                    if abs(tmp_angle) < 40:#曲がるかどうか
                        turn = 0
                    else:
                        turn = 1
                        
                    if (turn == 0) or (cross == 0 and tmp_angle != 180):#直下リンクか行き先が一つ、Uターンは除く
                        first_num = 1
                        if (turn == 0) and (up_lane >= 3):#直下リンクかつ、車線数>=3
                            for dlink_id2 in self.dlink_dict[node_id]:
                                if dlink_id2 is None:
                                    continue
                                dlink_od2 = (self.link_dict[dlink_id2][0], self.link_dict[dlink_id2][1])
                                if dlink_od2 in self.node_pair_dict:
                                    dn_direction2 = 1#順方向
                                else:
                                    dn_direction2 = -1#逆方向
                                tmp_angle2 = calcAngleMinus180to180(float(df_link.loc[self.link_id_dict[olink_id], "angle"]) + 180.0 * (up_direction == 1), float(df_link.loc[self.link_id_dict[dlink_id2], "angle"]) + 180.0 * (dn_direction2 == 1))
                                if tmp_angle2 <= -40:#右折がある
                                    first_num = 2#右折専用レーンを飛ばす
                                    break
                        if dn_lane <= up_lane:
                            for i in range(first_num, dn_lane+1):
                                values.append([node_id, self.link_id_dict[olink_id], i*up_direction, self.link_id_dict[dlink_id], i*dn_direction])
                            for i in range(dn_lane+1,up_lane+1):
                                values.append([node_id, self.link_id_dict[olink_id], i*up_direction, self.link_id_dict[dlink_id], dn_lane*dn_direction])
                        else:
                            for i in range(first_num,up_lane+1):
                                values.append([node_id, self.link_id_dict[olink_id], i*up_direction, self.link_id_dict[dlink_id], i*dn_direction])
                            for i in range(up_lane+1,dn_lane+1):
                                values.append([node_id, self.link_id_dict[olink_id], up_lane*up_direction, self.link_id_dict[dlink_id], i*dn_direction])
                    elif tmp_angle >= 40 and tmp_angle < 180:#左折
                        for i in range(1, dn_lane+1):
                            values.append([node_id, self.link_id_dict[olink_id], up_lane*up_direction, self.link_id_dict[dlink_id], i*dn_direction])
                    elif tmp_angle <= -40:#右折
                        for i in range(1,dn_lane+1):
                            values.append([node_id, self.link_id_dict[olink_id], 1*up_direction, self.link_id_dict[dlink_id], i*dn_direction])
                    #歩道
                    values.append([node_id, self.link_id_dict[olink_id], (up_lane+1)*up_direction, self.link_id_dict[dlink_id], (dn_lane+1)*dn_direction])
                            
        df_connectivity = pd.DataFrame(values, columns=["NodeID","UpLinkID","UpLaneID","DnLinkID","DnLaneID"])
        df_connectivity = df_connectivity.astype(int)
        df_connectivity.sort_values(["NodeID","UpLinkID","UpLaneID"],inplace=True)
        if connectivity_file is None:
            df_connectivity.to_csv(self.input_dir+"connectivity.csv", index=False)
        else:
            df_connectivity.to_csv(connectivity_file, index=False)
        print("connectivity.csv is written.")
        return df_connectivity
    
    def _create_signal(self, df_node: pd.DataFrame, signal_file: Optional[str] = None) -> pd.DataFrame:
        """
        信号情報をCSVに出力し、DataFrameを返す。

        Args:
            df_node (pd.DataFrame): ノード情報のDataFrame
            signal_file (Optional[str]): 出力ファイルパス。Noneの場合はデフォルトパス。
        Returns:
            pd.DataFrame: 信号情報のDataFrame。必要な属性がNoneの場合は空のDataFrame。
        """
        #signal.csv
        signalized_nodes = dict()#key:nodeId, value:signalId
        #g = Spatial3.Spatial(pd.DataFrame([[v[0],v[1]] for v in self.nodes.values()],columns=["lon","lat"]), distFun=self._hubeny)
        #g.calcVoronoi()
        #signal_id = 1
        #for k,v in self.signals.items():
        #    p = g.lineMgr.addPoint(v[0], v[1])
        #    node = g.searchNearestNode(p)
        #    node_coord = g.lineMgr.pointMgr.pointList[node]
        #    signalized_nodes[self.node_dict[(node_coord[0],node_coord[1])]] = signal_id
        #    signal_id += 1
        
        # 信号の座標の最近傍探索
        if self.signals is None:
            print("Please read osm data first.")
            return pd.DataFrame()
        if self.nodes is None:
            print("Please read osm data first.")
            return pd.DataFrame()
        node_data = df_node[["Lon","Lat"]].values
        
        tree = KDTree(node_data)
        
        signal_id = 1
        node_id_vals = df_node["ID"].values
        for k,v in self.signals.items():
            distances, ids = tree.query([v], k=1)
            if not hasattr(ids, "__len__") or len(ids) == 0:
                continue
            target_node_id = node_id_vals[ids[0]]
            
            signalized_nodes[target_node_id] = signal_id
            signal_id += 1
            
        
        for k1 in signalized_nodes.keys():
            for k2 in signalized_nodes.keys():
                if k1 == k2:
                    continue
                if ((k1, k2) in self.node_pair_dict) or ((k2, k1) in self.node_pair_dict):#信号付きノードがグラフ上で隣接していた場合
                    if HongoInput._hubeny(self.nodes[k1], self.nodes[k2]) < 25:#同一交差点と判定
                        sid = min(signalized_nodes[k1], signalized_nodes[k2])
                        signalized_nodes[k1] = sid
                        signalized_nodes[k2] = sid
        signal_ids = list(set([v for v in signalized_nodes.values()]))
        for i in range(len(signal_ids)):#信号IDを連番にする
            for k in signalized_nodes.keys():
                if signalized_nodes[k] == signal_ids[i]:
                    signalized_nodes[k] = i+1
        df_signal = pd.DataFrame([[v,k,0] for k,v in signalized_nodes.items()], columns=["ID", "NodeID", "RightTurn"])
        df_node.columns = ["NodeID", "Lon", "Lat"]
        df_signal = pd.merge(df_signal, df_node, on="NodeID")
        df_signal[["ID","NodeID"]] = df_signal[["ID","NodeID"]].astype(int)
        
        if signal_file is None:
            df_signal.to_csv(self.input_dir+"signal.csv", index=False)
        else:
            df_signal.to_csv(signal_file, index=False)
        print("")
        print("signal.csv is written.")
        return df_signal
            
    def get_data_from_osm(self) -> None:
        """
        OSMデータからネットワーク・信号情報を取得し、インスタンス変数に格納する。

        Returns:
            None
        """
        # pyrosm.OSMの用意
    
        # osmnx版
        bounding_box = shapely.geometry.Polygon(self.bb_coord)
        #drive_net = ox.graph.graph_from_polygon(bounding_box, simplify=True, retain_all=False, custom_filter='["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|road|residential|living_street|service"]')
        drive_net = ox_graph.graph_from_polygon(bounding_box, simplify=True, retain_all=False, network_type='drive')
        #walk_net = ox.graph.graph_from_polygon(bounding_box, simplify=True, retain_all=False, network_type='walk')
        node_gdf, edge_gdf = momepy.nx_to_gdf(drive_net, points=True, lines=True, spatial_weights=False)
        #node_gdf_walk, edge_gdf_walk = momepy.nx_to_gdf(walk_net, points=True, lines=True, spatial_weights=False)
        # geometryを追加
        if not isinstance(node_gdf, gpd.GeoDataFrame) or not isinstance(edge_gdf, gpd.GeoDataFrame):
            drive_net = HongoInput._add_geometry(node_gdf, edge_gdf)
        else:
            raise ValueError("node_gdf and edge_gdf must be GeoDataFrames.")
        #walk_net = HongoInput._add_geometry(node_gdf_walk, edge_gdf_walk)

        drive_net_light = copy.deepcopy(drive_net)
        drive_net_light.set_crs(epsg=4326, inplace=True)
        drive_net_light = self._process_drive_gdf_osmnx(drive_net_light)

        #walk_net_light = copy.deepcopy(walk_net)
        #walk_net_light.set_crs(epsg=4326, inplace=True)
        #walk_net_light = self._process_walk_gdf_osmnx(walk_net_light)

        self.drive_net = momepy.gdf_to_nx(drive_net_light,approach='primal',directed=True)
        #self.walk_net = momepy.gdf_to_nx(walk_net_light,approach='primal',directed=True)
        print(f'Drive network has {len(self.drive_net.nodes)} nodes and {len(self.drive_net.edges)} edges.')

        cnt = 0
        node_dict = {(n[0],n[1]):idx+cnt+1 for idx,n in enumerate(list(self.drive_net.nodes))}
        #cnt = len(node_dict)+1
        #walk_node_list = list(self.walk_net.nodes)
        ##歩道と車道で重複するノードは同一のノードとして扱う
        #for idx in range(len(walk_node_list)):
        #    tmp = (walk_node_list[idx][0],walk_node_list[idx][1])
        #    if not tmp in node_dict:
        #        node_dict[tmp] = cnt
        #        cnt += 1

        self.nodes = {v:k for k,v in node_dict.items()}

        # 信号位置の取得
        signal = ox.features.features_from_polygon(bounding_box, tags={"highway":["traffic_signals"]})
        self.signals = {n:(float(val.x), float(val.y)) for n,val in signal.geometry.items()}

        print("OSM data is read.")
        assert ((self.drive_net is not None) and (self.nodes is not None) and (self.signals is not None))
    
    def _process_drive_gdf_osmnx(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        OSMnxのdriveネットワークGeoDataFrameを加工する。

        Args:
            gdf (gpd.GeoDataFrame): OSMnxから取得したGeoDataFrame
        Returns:
            gpd.GeoDataFrame: 加工後のGeoDataFrame
        """
        gdf["lanes"]=gdf["lanes"].apply(HongoInput._get_lane)
        gdf["speed"]=gdf["highway"].apply(HongoInput._get_maxspeed)
        gdf=HongoInput._divide_multiLineString(gdf)
        gdf["geometry"]=gdf["geometry"].to_crs(epsg=self.epsg).simplify(5).to_crs(gdf.crs)
        gdf=HongoInput._divide_LineString(gdf)
        gdf=HongoInput._create_bidirection(gdf)
        
        return gdf

    def process_walk_gdf_osmnx(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        OSMnxのwalkネットワークGeoDataFrameを加工する。

        Args:
            gdf (gpd.GeoDataFrame): OSMnxから取得したGeoDataFrame
        Returns:
            gpd.GeoDataFrame: 加工後のGeoDataFrame
        """
        gdf["lanes"] = 1
        gdf["speed"] = 4.5
        gdf = HongoInput._divide_multiLineString(gdf)
        gdf["geometry"] = gdf["geometry"].to_crs(epsg=self.epsg).simplify(5).to_crs(gdf.crs)
        gdf = HongoInput._divide_LineString(gdf)
        return gdf
        
    @staticmethod
    def get_coords(gdf_link: gpd.GeoDataFrame) -> list:
        """
        GeoDataFrameから全座標を抽出する。

        Args:
            gdf_link (gpd.GeoDataFrame): リンクのGeoDataFrame
        Returns:
            list: 座標リスト
        """
        coords = []
        vals = gdf_link["geometry"]
        for geom in vals:
            if isinstance(geom, shapely.geometry.MultiLineString):
                for geo in geom.geoms:
                    coords.extend(list(geo.coords))
            else:
                coords.extend(list(geom.coords))
        return coords
        
    @staticmethod
    def get_coord(geom: shapely.geometry.base.BaseGeometry) -> list:
        """
        ジオメトリから座標リストを抽出する。

        Args:
            geom (shapely.geometry.base.BaseGeometry): ジオメトリ
        Returns:
            list: 座標リスト
        """
        coord=[]
        if isinstance(geom, shapely.geometry.MultiLineString):
            for geo in geom.geoms:
                coord.extend(geo.coords)
        else:
            coord.extend(geom.coords)
        return coord
        
    @staticmethod
    def _divide_multiLineString(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        MultiLineStringを分割して単一LineStringに変換する。

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame
        Returns:
            gpd.GeoDataFrame: 分割後のGeoDataFrame
        """
        v=gdf.values.tolist()
        columns=gdf.columns
        geom_index=columns.tolist().index('geometry')
        for i in range(len(gdf.geometry)):
            tmp=gdf.geometry.iloc[i]
            if isinstance(tmp, shapely.geometry.MultiLineString):
                v[i][geom_index]=tmp.geoms[0]
                appendval=[]
                for j in range(1,len(tmp.geoms)):
                    rowval=copy.deepcopy(v[i])
                    rowval[geom_index]=tmp.geoms[j]
                    appendval.append(rowval)
                v.extend(appendval)
        gdf_new=gpd.GeoDataFrame(v,columns=columns)
        if gdf.crs is not None:
            gdf_new.set_crs(gdf.crs, inplace=True)
        return gdf_new
    
    @staticmethod
    def _divide_LineString(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        LineStringを2点ごとに分割する。

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame
        Returns:
            gpd.GeoDataFrame: 分割後のGeoDataFrame
        """
        v=gdf.values.tolist()
        columns=gdf.columns
        geom_index=columns.tolist().index('geometry')
        for i in range(len(gdf.geometry)):
            tmp: shapely.geometry.LineString = gdf.geometry.iloc[i]
            if len(tmp.coords)>2:
                v[i][geom_index]=shapely.geometry.LineString(tmp.coords[0:2])
                appendval=[]
                for j in range(1,len(tmp.coords)-1):
                    rowval=copy.deepcopy(v[i])
                    rowval[geom_index]=shapely.geometry.LineString(tmp.coords[j:j+2])
                    appendval.append(rowval)
                v.extend(appendval)
        gdf_new=gpd.GeoDataFrame(v,columns=columns)
        if gdf.crs is not None:
            gdf_new.set_crs(gdf.crs, inplace=True)
        return gdf_new
        
    @staticmethod
    def _create_bidirection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        双方向リンクを生成する。

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame
        Returns:
            gpd.GeoDataFrame: 双方向化後のGeoDataFrame
        """
        #geometryは全て単一LineString
        val=gdf.values.tolist()
        columns=gdf.columns
        geom_index=columns.tolist().index('geometry')
        append_vals = []
        oneway_vals = gdf[['highway','oneway']].values
        for i in range(len(gdf.oneway)):
            if (oneway_vals[i, 0] != 'motorway') and (not oneway_vals[i, 1]):
                inv_val = copy.deepcopy(val[i])
                inv_val[geom_index] = shapely.geometry.LineString([c for c in reversed(list(inv_val[geom_index].coords))])
                append_vals.append(inv_val)
        val.append(inv_val)
        gdf_new=gpd.GeoDataFrame(val,columns=columns)
        if gdf.crs is not None:
            gdf_new.set_crs(gdf.crs, inplace=True)
        return gdf_new
        
    @staticmethod
    def _add_attribute_to_G(g: nx.Graph, gdf: gpd.GeoDataFrame) -> nx.Graph:
        """
        GeoDataFrameの属性をnetworkxグラフに追加する。

        Args:
            g (nx.Graph): networkxグラフ
            gdf (gpd.GeoDataFrame): 属性を持つGeoDataFrame
        Returns:
            nx.Graph: 属性追加後のグラフ
        """
        geoms = gdf.geometry.values
        idx=[(geoms[n].coords[0],geoms[n].coords[1],n) for n in range(len(geoms))]
        gdf["idx"]=idx
        gdf.set_index('idx',inplace=True)
        for col in gdf.columns:
            if col=="geometry":
                continue
            nx.set_edge_attributes(g,dict(gdf.loc[:,col]),name=col)
        return g
        
    @staticmethod
    def _get_shared_node(gdf: gpd.GeoDataFrame) -> set:
        """
        複数リンクで共有されるノード座標を抽出する。

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame
        Returns:
            set: 共有ノード座標の集合
        """
        node_dict: dict[list[float], int] = dict()
        tmpvals=gdf.geometry.values
        res = set()
        for idx in range(len(gdf)):
            coords=list(set(HongoInput.get_coord(tmpvals[idx])))
            for coord in coords:
                if coord in node_dict:
                    node_dict[coord]+=1
                else:
                    node_dict[coord]=1
            res=set([k for k,v in node_dict.items() if v>1])
        return res
        
    @staticmethod
    def _divide_by_shared_node(linestring: shapely.geometry.base.BaseGeometry, shared_nodes: set) -> shapely.geometry.base.BaseGeometry:
        """
        共有ノードでLineStringを分割する。

        Args:
            linestring (shapely.geometry.base.BaseGeometry): ジオメトリ
            shared_nodes (set): 共有ノード座標
        Returns:
            shapely.geometry.base.BaseGeometry: 分割後のジオメトリ
        """
        nodes=set(HongoInput.get_coord(linestring))
        target_nodes=nodes & shared_nodes
        for node in list(target_nodes):
            if isinstance(linestring, shapely.geometry.MultiLineString):
                geoms=list(linestring.geoms)
                for i in range(len(geoms)):
                    g=geoms[i]
                    if node in set(HongoInput.get_coord(g)):
                        geoms[i:i+1]=list(ops.split(g,shapely.geometry.Point(node)).geoms)
                        break
                linestring=shapely.geometry.MultiLineString(geoms)
            else:
                linestring=ops.split(linestring,shapely.geometry.Point(node))
                linestring=shapely.geometry.MultiLineString(linestring.geoms)
        return linestring
    
    @staticmethod
    def _add_geometry(node_gdf: gpd.GeoDataFrame, edge_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        ノード・エッジ情報からジオメトリを追加する。

        Args:
            node_gdf (gpd.GeoDataFrame): ノードGeoDataFrame
            edge_gdf (gpd.GeoDataFrame): エッジGeoDataFrame
        Returns:
            gpd.GeoDataFrame: ジオメトリ追加後のエッジGeoDataFrame
        """
        node_gdf.set_index('nodeID', drop=False, inplace=True)
        geoms = []
        od_nodes = edge_gdf.loc[:,["node_start","node_end"]].values
        geom_original = edge_gdf["geometry"].values
        for i in range(len(od_nodes)):
            if geom_original[i] is None:
                geoms.append(shapely.geometry.LineString([node_gdf.loc[od_nodes[i,0], "geometry"],node_gdf.loc[od_nodes[i,1], "geometry"]]))
            else:
                geoms.append(geom_original[i])
        edge_gdf["geometry"] = geoms
        return edge_gdf
        
    @staticmethod
    def _get_maxspeed(highway: str) -> int:
        """
        道路種別から最大速度を返す。

        Args:
            highway (str): 道路種別
        Returns:
            int: 最大速度
        """
        if highway=="motorway":
            return 80
        elif highway=="motorway_link":
            return 40
        elif highway=="trunk":
            return 50
        elif highway=="trunk_link":
            return 40
        elif highway=="primary":
            return 50
        elif highway=="primary_link":
            return 40
        elif highway=="secondary":
            return 50
        elif highway=="secondary_link":
            return 40
        elif highway=="tertiary":
            return 40
        elif highway=="tertiary_link":
            return 40
        elif highway=="unclassified":
            return 40
        elif highway=="road":
            return 30
        elif highway=="residential":
            return 30
        elif highway=="living_street":
            return 20
        elif highway=="service":
            return 10
        else:
            print(highway)
            return 20
        
    @staticmethod
    def _get_lane(lane: Optional[int | str]) -> int:
        """
        レーン数を整数で返す。

        Args:
            lane (int | str): レーン数または文字列
        Returns:
            int: レーン数
        """
        try:
            x = int(str(lane))
        except ValueError:
            x = 1
        return x
    
    
    @staticmethod
    def _hubeny(o_coord: tuple[float, float], d_coord: tuple[float, float]) -> float:
        """
        ヒュベニの公式で2点間距離を計算する。

        Args:
            o_coord (tuple[float, float]): 始点座標
            d_coord (tuple[float, float]): 終点座標
        Returns:
            float: 距離（メートル）
        """
        #ヒュベニの公式
        olon, olat = o_coord
        dlon, dlat = d_coord
        
        rx=6378137.0
        ry=6356752.314245
        
        dy = (dlat - olat) / 180 * np.pi
        dx = (dlon - olon) / 180 * np.pi
        p=((olat + dlat) / 2) / 180 * np.pi
        
        e=((rx**2 - ry**2) / (rx**2))**0.5
        
        w=(1 - e**2 * np.sin(p)**2)**0.5
        
        n=rx/w
        m=rx * (1 - e**2) / w**3
        
        return ((dy * m)**2 + (dx * n * np.cos(p))**2)**0.5
    
    @staticmethod
    def show_network(input_dir: str) -> None:
        """
        入力ディレクトリのネットワークを可視化する。

        Args:
            input_dir (str): 入力ディレクトリのパス
        Returns:
            None
        """
        link = pd.read_csv(input_dir+"link.csv")
        node = pd.read_csv(input_dir+"node.csv")
        # connectivity = pd.read_csv(input_dir+"connectivity.csv")
        # signal = pd.read_csv(input_dir+"signal.csv")
        if not set(["OLon","OLat","DLon","DLat"])<=set(link.columns.tolist()):
            node.columns = ["ONodeID","OLon","OLat"]
            link = pd.merge(link, node, on="ONodeID")
            node.columns = ["DNodeID","DLon","DLat"]
            link = pd.merge(link, node, on="DNodeID")
            node.columns = ["ID","Lon","Lat"]
        link["geometry"] = [shapely.geometry.LineString([link.loc[idx,["OLon","OLat"]].values.tolist(),link.loc[idx,["DLon","DLat"]].values.tolist()]) for idx in link.index]
        
        crs_pre = 4326
        crs = 2446
        df_link = gpd.GeoDataFrame(link)
        df_link.crs = f'epsg:{crs_pre}'
        df_link = df_link.to_crs(epsg=crs)
        df_link.plot()

    @staticmethod
    def create_connectivity(df_link: pd.DataFrame, connectivity_file: str) -> pd.DataFrame:
        """
        link.csvからconnectivity.csvを生成する。

        Args:
            df_link (pd.DataFrame): リンク情報のDataFrame
            connectivity_file (str): 出力ファイルパス
        Returns:
            pd.DataFrame: コネクティビティ情報のDataFrame
        """
        link_vals = df_link[["ID","ONodeID","DNodeID","LeftCarNum","RightCarNum"]].values

        dlink_dict: dict[int, list[int]] = dict()#key:nodeId, value:[linkId]
        for i in range(len(link_vals)):
            if link_vals[i,1] in dlink_dict:
                dlink_dict[link_vals[i,1]].append(link_vals[i,0])
            else:
                dlink_dict[link_vals[i,1]] = [link_vals[i,0]]
            if link_vals[i,2] == 0:
                continue
            if link_vals[i,2] in dlink_dict:
                dlink_dict[link_vals[i,2]].append(link_vals[i,0])
            else:
                dlink_dict[link_vals[i,2]] = [link_vals[i,0]]

        
        #connectivity.csv
        df_link["angle"] = np.vectorize(calcAngle)(df_link["OLon"], df_link["OLat"], df_link["DLon"], df_link["DLat"])
        df_link.set_index("ID", drop=False, inplace=True)
        link_vals = df_link[["ID","ONodeID","DNodeID","LeftCarNum","RightCarNum","angle"]].values

        values = []
        for idx in range(len(link_vals)):
            olink_id = link_vals[idx,0]
            for j in range(2):
                if j == 0:
                    # od = (link_vals[idx,1],link_vals[idx,2])#v:(oNodeId, dNodeId)
                    up_lane = link_vals[idx,3]
                    node_id = link_vals[idx,2]
                    up_direction = 1#順方向
                else:
                    # od = (link_vals[idx,2],link_vals[idx,1])
                    up_lane = link_vals[idx,4]
                    node_id = link_vals[idx,1]
                    up_direction = -1#逆方向
                    if link_vals[idx,4] == 0:
                        continue
                
                
                cross = 0#逆行以外の行き先が2つ以上あるか否か
                if node_id not in dlink_dict:
                    continue
                for dlink_id in dlink_dict[node_id]:
                    if dlink_id == olink_id:#逆行リンク
                        continue
                    cross += 1
                cross = (cross>1)*1#逆行以外の行き先が2つ以上
                for dlink_id in dlink_dict[node_id]:
                    if dlink_id == olink_id:#逆行リンク
                        continue
                    if node_id == df_link.loc[dlink_id,"ONodeID"]:
                        dn_direction = 1#順方向
                        dn_lane = int(df_link.loc[dlink_id,"LeftCarNum"])
                    else:
                        dn_direction = -1#逆方向
                        dn_lane = int(df_link.loc[dlink_id,"RightCarNum"])
                    tmp_angle = calcAngleMinus180to180(float(df_link.loc[olink_id, "angle"] + 180.0 * (up_direction == 1)), float(df_link.loc[dlink_id,"angle"] + 180.0 * (dn_direction == 1)))

                    if abs(tmp_angle) < 40:#曲がるかどうか
                        turn = 0
                    else:
                        turn = 1

                    if (turn == 0) or (cross == 0 and tmp_angle != 180):#直下リンクか行き先が一つ、Uターンは除く
                        first_num = 1
                        if (turn == 0) and (up_lane >= 3):#直下リンクかつ、車線数>=3
                            for dlink_id2 in  dlink_dict[node_id]:
                                if node_id == df_link.loc[dlink_id2,"ONodeID"]:
                                    dn_direction2 = 1#順方向
                                else:
                                    dn_direction2 = -1#逆方向
                                tmp_angle2 = calcAngleMinus180to180(float(df_link.loc[olink_id, "angle"] + 180.0 * (up_direction == 1)), float(df_link.loc[dlink_id2,"angle"] + 180.0 * (dn_direction2 == 1)))
                                if tmp_angle2 <= -40:#右折がある
                                    first_num = 2#右折専用レーンを飛ばす
                                    break
                        if dn_lane <= up_lane:
                            for i in range(first_num, dn_lane+1):
                                values.append([node_id, olink_id, i*up_direction, dlink_id, i*dn_direction])
                            for i in range(dn_lane+1,up_lane+1):
                                values.append([node_id, olink_id, i*up_direction, dlink_id, dn_lane*dn_direction])
                        else:
                            for i in range(first_num,up_lane+1):
                                values.append([node_id, olink_id, i*up_direction, dlink_id, i*dn_direction])
                            for i in range(up_lane+1,dn_lane+1):
                                values.append([node_id, olink_id, up_lane*up_direction, dlink_id, i*dn_direction])
                    elif tmp_angle >= 40 and tmp_angle < 180:#左折
                        for i in range(1, dn_lane+1):
                            values.append([node_id, olink_id, up_lane*up_direction, dlink_id, i*dn_direction])
                    elif tmp_angle <= -40:#右折
                        for i in range(1,dn_lane+1):
                            values.append([node_id, olink_id, 1*up_direction, dlink_id, i*dn_direction])
                    #歩道
                    values.append([node_id, olink_id, (up_lane+1)*up_direction, dlink_id, (dn_lane+1)*dn_direction])

        df_connectivity = pd.DataFrame(values, columns=["NodeID","UpLinkID","UpLaneID","DnLinkID","DnLaneID"])
        df_connectivity = df_connectivity.astype(int)
        df_connectivity.sort_values(["NodeID","UpLinkID","UpLaneID"],inplace=True)
        df_connectivity.to_csv(connectivity_file, index=False)
        return df_connectivity
        
if __name__ == "__main__":
    hongo_input = HongoInput(input_dir="../", epsg=4326, bb_coord=[[132.684188,33.864796],[132.817895,33.863369],[132.828858,33.745195],[132.700745,33.755126]])
    hongo_input.create_network_input()
    #HongoInput.show_network("../")

