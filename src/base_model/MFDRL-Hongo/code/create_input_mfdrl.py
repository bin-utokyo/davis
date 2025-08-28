import os

import numpy as np
import pandas as pd

import geopandas as gpd
import pyproj
import shapely

class MFDRLInput:
    '''
    node_file = self.input_dir + "node_car1.csv"#KEY_CODE,nodeID,lon,lat,DisToSea,Shinsui,Population,Household,Elevation,edge
    link_file = self.input_dir + "link.csv"#linkID,O,D,Olon,Olat,Dlon,Dlat,limit2010
    RLparameter_file = self.input_dir + "EstimationResult.csv"#Kamaishi,Ofunato,Rikuzentakata,Kesennuma,Ishinomaki,Higashimatsushima,Iwanuma,Iwaki
    MFDparameter_file = self.input_dir + "mfd_params3.csv"#a,b,d_link,elevation,population,DisToSea,a_init,b_init
    '''
    def __init__(self, input_dir: str, epsg: int, bb_coord: list[tuple[float, float]]):
        """
        Args:
            input_dir (str): 入力ディレクトリのパス
            epsg (int): 平面直角座標のEPSGコード
            bb_coord (list[tuple[float, float]]): バウンディングボックス座標
        """
        self.input_dir = input_dir
        self.epsg = epsg#平面直角座標のepsg
        self.bb_coord = bb_coord
        
    def read_data(
        self,
        mesh_file: str,
        population_file: str | None,
        hongo_node_file: str,
        hongo_link_file: str
    ) -> None:
        """
        データファイルを読み込み、属性として保持する。

        Args:
            mesh_file (str): メッシュデータのファイルパス
            population_file (str | None): 人口データのファイルパス（Noneの場合は未使用）
            hongo_node_file (str): Hongoノードデータのファイルパス
            hongo_link_file (str): Hongoリンクデータのファイルパス
        """
        # read_mesh_data
        self.mesh_gdf = gpd.read_file(mesh_file)

        # read population_file
        if population_file is None:
            self.population = None
        else:
            ext = os.path.splitext(population_file)  # 拡張子
            if ext == '.csv':
                self.population = pd.read_csv(population_file)
            else:
                self.population = gpd.read_file(population_file, encoding="shift-jis")
        
        #Hongoのネットワークデータ　リンク容量を計算するため．前地域のネットワークデータ
        self.hongo_node = pd.read_csv(hongo_node_file)
        self.hongo_link = pd.read_csv(hongo_link_file)

        self.hongo_link.set_index('ID', drop=False, inplace=True)
        self.hongo_node.set_index('ID', drop=False, inplace=True)
        wgs84 = pyproj.CRS('EPSG:4326')
        to = pyproj.CRS(f'EPSG:{self.epsg}')
        project = pyproj.Transformer.from_crs(wgs84, to, always_xy=True).transform
        lengths = []
        for i in self.hongo_link.index:
            o_coord = self.hongo_node.loc[self.hongo_link.loc[i, 'ONodeID'], ['Lon', 'Lat']].values
            d_coord = self.hongo_node.loc[self.hongo_link.loc[i, 'DNodeID'], ['Lon', 'Lat']].values

            o_geom = shapely.geometry.Point(o_coord)
            d_geom = shapely.geometry.Point(d_coord)
            o_geom = shapely.ops.transform(project, o_geom)
            d_geom = shapely.ops.transform(project, d_geom)

            le = shapely.geometry.LineString([o_geom, d_geom]).length
            lengths.append(le)
        self.hongo_link['length'] = lengths
        
    def create_network(
        self,
        tmp_input_dir: str,
        node_mesh_dict: dict[int, int],
        start_time: int,
        end_time: int
    ) -> None:
        """
        ネットワークデータ（node.csv, link.csv）を作成する。

        Args:
            tmp_input_dir (str): 一時入力ディレクトリ
            node_mesh_dict (dict[int, int]): ノードIDとメッシュIDの対応辞書
            start_time (int): 開始時刻
            end_time (int): 終了時刻
        """
        # input_dirにlink.csvとnode.csvを作る
        #node.csv KEY_CODE,nodeID,lon,lat,Population,edge
        #link.csv linkID,O,D,Olon,Olat,Dlon,Dlat,limit  limitはHongoの道路ネットワークから算出
        
        #node.csv

        demand_dict = dict()  # key: mesh_key_code, value: demand(population)
        if self.population is not None:
            if "area" in self.population.columns and "population" in self.population.columns:  # モバ空
                vals = self.population[["area", "population"]].values
            elif "MESH_ID" in self.population.columns and "POP2020" in self.population.columns:  # 国土数値情報将来人口推計
                vals = self.population[["MESH_ID", "POP2020"]].values
            else:
                vals = []
            for i in range(len(vals)):
                demand_dict[int(vals[i, 0])] = int(vals[i, 1])
        
        vals = []#KEY_CODE,nodeID,lon,lat,Population,edge
        for i in self.mesh_gdf.index:
            mesh_code = self.mesh_gdf.loc[i,"KEY_CODE"]
            geom = self.mesh_gdf.loc[i,"geometry"]
            lon,lat = self.mesh_gdf.loc[i,"geometry"].centroid.coords[0]
            demand = 0
            if mesh_code in demand_dict:
                demand = demand_dict[mesh_code]
            vals.append([mesh_code, self.mesh_gdf.loc[i, "nodeID"], lon, lat, demand, 0, geom])
                

        self.node_gdf = gpd.GeoDataFrame(vals, columns=["KEY_CODE","nodeID","lon","lat","Population","edge","geometry"])
        
        node = pd.DataFrame([v[0:-1] for v in vals], columns=["KEY_CODE","nodeID","lon","lat","Population","edge"])
        node.to_csv(tmp_input_dir+"node.csv", index=False)
        
        #link.csv linkID,O,D,Olon,Olat,Dlon,Dlat,limit  limitは1時間の容量
        
        self.node_gdf.set_crs(epsg=4326, inplace=True)
        self.node_gdf.to_crs(epsg=self.epsg, inplace=True)
        
        vals = []
        for i in self.hongo_node.index:
            vals.append([self.hongo_node.loc[i,"ID"], shapely.geometry.Point((self.hongo_node.loc[i,"Lon"], self.hongo_node.loc[i,"Lat"]))])
        hongo_node_gdf = gpd.GeoDataFrame(vals, columns=["ID", "geometry"])
        hongo_node_gdf.set_crs(epsg=4326, inplace=True)
        hongo_node_gdf.to_crs(epsg=self.epsg, inplace=True)
        
        
        # nodeのあるmesh
        self.node_mesh_dict = node_mesh_dict
            
            
        link_dict: dict[tuple[int, int], int] = dict()#key: (o_mesh,d_mesh),value:lim
        link_length: dict[int, float] = dict()#key: mesh_id, value: total link length in mesh
        link_mesh: dict[int, list[int]] = dict()#key: link_id, value: [mesh_id]
        mesh_link: dict[int, list[int]] = dict()#key: mesh_id, value: [link_id]
        for i in self.hongo_link.index:
            try:
                o_mesh,d_mesh = self.node_mesh_dict[self.hongo_link.loc[i, "ONodeID"]],self.node_mesh_dict[self.hongo_link.loc[i, "DNodeID"]]
            except KeyError:
                continue
            if (o_mesh,d_mesh) in link_dict:
                link_dict[(o_mesh,d_mesh)] += self.hongo_link.loc[i,"LeftCarNum"]*360
            else:
                link_dict[(o_mesh,d_mesh)] = self.hongo_link.loc[i,"LeftCarNum"]*360
            if (d_mesh,o_mesh) in link_dict:
                link_dict[(d_mesh,o_mesh)] += self.hongo_link.loc[i,"RightCarNum"]*360
            else:
                link_dict[(d_mesh,o_mesh)] = self.hongo_link.loc[i,"RightCarNum"]*360
            link_id = self.hongo_link.loc[i, 'ID']
            if o_mesh == d_mesh:
                if o_mesh in link_length:
                    link_length[o_mesh] += self.hongo_link.loc[i, 'length'] * sum(self.hongo_link.loc[i, ['LeftCarNum', 'RightCarNum']].values)
                else:
                    link_length[o_mesh] = self.hongo_link.loc[i, 'length'] * sum(self.hongo_link.loc[i, ['LeftCarNum', 'RightCarNum']].values)
                link_mesh[link_id] = [o_mesh]
                if o_mesh in mesh_link:
                    mesh_link[o_mesh].append(link_id)
                else:
                    mesh_link[o_mesh] = [link_id]
            else:
                link_mesh[link_id] = [o_mesh, d_mesh]
                if o_mesh in mesh_link:
                    mesh_link[o_mesh].append(link_id)
                else:
                    mesh_link[o_mesh] = [link_id]
                if d_mesh in mesh_link:
                    mesh_link[d_mesh].append(link_id)
                else:
                    mesh_link[d_mesh] = [link_id]
        self.link_length = link_length
        self.link_mesh = link_mesh
        self.mesh_link = mesh_link
        
        vals = []
        lid = 1
        for mesh_pair, lim in link_dict.items():
            mesh_i = self.node_gdf.index[(self.node_gdf["nodeID"]==mesh_pair[0])][0]
            mesh_i2 = self.node_gdf.index[(self.node_gdf["nodeID"]==mesh_pair[1])][0]
            geom = self.node_gdf.loc[mesh_i, "geometry"].centroid
            geom2 = self.node_gdf.loc[mesh_i2, "geometry"].centroid
            if geom.distance(geom2)<600:
                mesh_link_geom = shapely.geometry.LineString([(geom.x,geom.y), (geom2.x,geom2.y)])
                vals.append([lid, self.node_gdf.loc[mesh_i,"nodeID"],self.node_gdf.loc[mesh_i2,"nodeID"],lim,mesh_link_geom])
                lid += 1
        
        self.link_gdf = gpd.GeoDataFrame(vals, columns=["linkID","O","D","limit","geometry"])
        
        self.link_gdf.set_crs(epsg=self.epsg, inplace=True)
        self.link_gdf.to_crs(epsg=4326, inplace=True)
        self.node_gdf.to_crs(epsg=4326, inplace=True)
        
        link_coords = []# Olon, Olat, Dlon, Dlat
        
        for i in self.link_gdf.index:
            o_coord,d_coord = self.link_gdf.loc[i,"geometry"].coords[0],self.link_gdf.loc[i,"geometry"].coords[1]
            link_coords.append([o_coord[0], o_coord[1], d_coord[0], d_coord[1]])
        self.link_gdf = gpd.GeoDataFrame(np.concatenate([self.link_gdf.values, np.array(link_coords)], axis=1), columns=["linkID","O","D","limit","geometry","Olon","Olat","Dlon","Dlat"])
        self.link_gdf = self.link_gdf.loc[:,["linkID","O","D","Olon","Olat","Dlon","Dlat","limit","geometry"]]
        
        link = pd.DataFrame(self.link_gdf.values[:,0:-1], columns=["linkID","O","D","Olon","Olat","Dlon","Dlat","limit"])
        link.to_csv(tmp_input_dir+"link.csv",index=False)
        
        
    def create_mesh(self, mesh_file: str) -> str:
        """
        メッシュデータを読み込み、GeoDataFrameを作成しGeoJSONとして保存する。

        Args:
            mesh_file (str): メッシュデータのCSVファイルパス

        Returns:
            str: 作成したGeoJSONファイルのパス
        """
        # read_mesh_data
        mesh_df = pd.read_csv(mesh_file)
        bb = shapely.geometry.Polygon(self.bb_coord)
        
        mesh_code = mesh_df["Code"].values
        mesh_coord = mesh_df[["Long","Lat"]].values
        
        k_codes = []
        coords: list[tuple[float, float]] = []
        polygons = []
        for i in range(len(mesh_df)):
            if (not np.isnan(mesh_code[i])):
                k_codes.append(int(mesh_code[i]))
                if len(coords)>0:#次のポリゴンの行に移ったので，前までのポリゴンを追加
                    new_polygon = shapely.geometry.Polygon(coords)
                    if bb.intersects(new_polygon):
                        polygons.append(new_polygon)
                    else:
                        k_codes.pop()
                    coords = []
            coords.append((mesh_coord[i, 0], mesh_coord[i,1]))
            
        new_polygon = shapely.geometry.Polygon(coords)
        if bb.intersects(new_polygon):
            polygons.append(new_polygon)
        else:
            k_codes.pop()
        
        self.mesh_gdf = gpd.GeoDataFrame({"KEY_CODE": k_codes,"nodeID": list(range(1,len(polygons)+1)), "geometry": polygons})
        self.mesh_gdf.set_crs(epsg=4326, inplace=True)
        if (len(self.mesh_gdf)==len(self.mesh_gdf.KEY_CODE.unique())-1):
            print("Not Unique KEY_CODE")
        file_name = os.extsep.join(mesh_file.split(os.extsep)[0:-1])
        geofile = f"{file_name}.geojson"
        self.mesh_gdf.to_file(geofile, driver="GeoJSON", index=False)
        return geofile
        
        
        
        
        
                
        
        
        
    
