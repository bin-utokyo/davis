import sys
import os
# 環境変数にHongoのビルドディレクトリがある場合はそちらも追加
if "HONGO_BUILD_DIR" in os.environ:
    sys.path.append(os.environ["HONGO_BUILD_DIR"])
else: 
    sys.path.append(os.sep.join(["..", "Hongo","build"]))#hongoのビルドディレクトリのデフォルトパス

import hongo
import create_input_hongo

import shapely
from shapely import ops
import geopandas as gpd
import pyproj
import pandas as pd
import numpy as np

from scipy.spatial import KDTree

import pickle

def get_input_obj(input_dir: str, epsg: int, bb_coord: list[list[float]]) -> 'create_input_hongo.HongoInput':
    """
    入力オブジェクトを生成する。

    Args:
        input_dir (str): 入力ディレクトリのパス。
        epsg (int): EPSGコード。
        bb_coord (list[list[float]]): バウンディングボックス座標。

    Returns:
        create_input_hongo.HongoInput: 入力オブジェクト。
    """
    input_obj = create_input_hongo.HongoInput(input_dir, epsg, bb_coord)
    return input_obj
    
def create_input_initial(input_obj: 'create_input_hongo.HongoInput', geo_mesh_file: str, node_mesh_file: str) -> 'create_input_hongo.HongoInput':
    """
    初期入力データを作成し、ノードメッシュ情報を保存する。

    Args:
        input_obj (create_input_hongo.HongoInput): 入力オブジェクト。
        geo_mesh_file (str): メッシュGeoJSONファイルのパス。
        node_mesh_file (str): ノードメッシュ情報保存先パス。

    Returns:
        create_input_hongo.HongoInput: 入力オブジェクト。
    """
    print("Starting to load OSM data.")
    input_obj.get_data_from_osm()
    print("Finish loading OSM data.")
    input_obj.create_network_input()
    #input_obj.create_od_from_pt(pt_czone_file, pt_master_file)
    
    #node_mesh
    node_mesh = dict()#key: nodeID_hongo, value: nodeID_mfdrl
    node = pd.read_csv(input_obj.input_dir+"node.csv")
    mesh_gdf = gpd.read_file(geo_mesh_file)
    for i in node.index:
        p = shapely.geometry.Point(node.loc[i, ["Lon", "Lat"]])
        idx = mesh_gdf.geometry.contains(p)
        if sum(idx) > 0:#nodeを含むメッシュがある場合
            node_mesh[node.loc[i, "ID"]] = mesh_gdf.loc[idx, "nodeID"].values[0]
    pickle_dump(node_mesh, node_mesh_file)
    return input_obj
    
    
def create_input(
    input_obj: 'create_input_hongo.HongoInput',
    tmp_input_dir: str,
    node_satellite_file: str | None,
    link_satellite_file: str | None
) -> 'create_input_hongo.HongoInput':
    """
    衛星データを用いてリンク情報を更新し、各種入力ファイルを生成する。

    Args:
        input_obj (create_input_hongo.HongoInput): 入力オブジェクト。
        tmp_input_dir (str): 一時入力ディレクトリ。
        node_satellite_file (str | None): 衛星ノードデータファイル。
        link_satellite_file (str | None): 衛星リンクデータファイル。

    Returns:
        create_input_hongo.HongoInput: 入力オブジェクト。
    """
    #衛星データよりリンクの車線数を更新する．
    # link_satelliteのdamage列が0以上のものに対応するリンクを探索し，切断する．
    #　satellite_linkのo_node,d_nodeの近傍kのノードをそれぞれ抽出し，その間にエッジが張られているような組を探索する．
    if (node_satellite_file is None) or (link_satellite_file is None):
        node_satellite = None
        link_satellite = None
    else:
        node_satellite = pd.read_csv(node_satellite_file)#node_id,lat,lon
        node_satellite.set_index("node_id", drop=False, inplace=True)
        link_satellite = pd.read_csv(link_satellite_file)#from,to,LeftCarNum,RightCarNum,damage
    
    #node.csv
    node = pd.read_csv(input_obj.input_dir+"node.csv")
    
    node["ID"] = node["ID"].astype(int)
    node.to_csv(tmp_input_dir+"node.csv",index=False)
    
    #link.csv
    link = pd.read_csv(input_obj.input_dir+"link.csv")
    link[["ID","ONodeID","DNodeID","LeftCarNum","LeftPedNum","LeftBeltNum","RightCarNum","RightPedNum","RightBeltNum"]] = link[["ID","ONodeID","DNodeID","LeftCarNum","LeftPedNum","LeftBeltNum","RightCarNum","RightPedNum","RightBeltNum"]].astype(int)
    link.set_index("ID", drop=False, inplace=True)
    link_od_dict = dict()#key: (o_node, d_node), values: link_id
    link_vals = link[["ID","ONodeID","DNodeID"]].values

    change_link_dict: dict[int, list[int]] = dict()  #key: linkID, value:[leftcardiff,rightcardiff]
    for i in range(len(link)):
        link_od_dict[(link_vals[i,1],link_vals[i,2])] = link_vals[i,0]
        link_od_dict[(link_vals[i,2],link_vals[i,1])] = link_vals[i,0]
        
    if (node_satellite is not None) and (link_satellite is not None):
        # 最近傍探索の前処理
        node_data = node[["Lon","Lat"]].values
        node_geom = shapely.geometry.MultiPoint(node_data)
        wgs84 = pyproj.CRS("EPSG:4326")
        utm = pyproj.CRS(f'EPSG:{input_obj.epsg}')
        proj = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        node_geom_utm = ops.transform(proj, node_geom)
        node_data_utm = [(float(node_geom_utm.geoms[i].coords[0][0]), float(node_geom_utm.geoms[i].coords[0][1])) for i in range(len(node_geom_utm.geoms))]

        tree = KDTree(node_data_utm)
     
        link_satellite_onode_vals = link_satellite["from"].values
        link_satellite_dnode_vals = link_satellite["to"].values
        link_satellite_lane_vals = link_satellite["Lanes"].values
        #link_satellite_left_vals = link_satellite["LeftCarNum"].values
        #link_satellite_right_vals = link_satellite["RightCarNum"].values
        node_id_vals = node["ID"].values
        for i in range(len(link_satellite)):
            onode_geom = shapely.geometry.Point(node_satellite.loc[link_satellite_onode_vals[i],["lon","lat"]].values)
            dnode_geom = shapely.geometry.Point(node_satellite.loc[link_satellite_dnode_vals[i],["lon","lat"]].values)
            onode_geom_utm = ops.transform(proj, onode_geom)
            dnode_geom_utm = ops.transform(proj, dnode_geom)

            onode_utm = (float(onode_geom_utm.coords[0][0]), float(onode_geom_utm.coords[0][1]))
            dnode_utm = (float(dnode_geom_utm.coords[0][0]), float(dnode_geom_utm.coords[0][1]))

            o_distances, o_ids = tree.query([onode_utm], k=5)
            d_distances, d_ids = tree.query([dnode_utm], k=5)

            # check attr of o_ids and d_ids
            if isinstance(o_ids, (int, np.integer)):
                o_ids = [o_ids]
                o_distances = [o_distances]
            if isinstance(d_ids, (int, np.integer)):
                d_ids = [d_ids]
                d_distances = [d_distances]

            damaged_id = -1
            od_pair = [-1,-1]
            min_dist = 1000
            # o_nodeとd_nodeが最も近いリンクを探索
            for j in range(len(o_ids)):
                for k in range(len(d_ids)):
                    o = node_id_vals[o_ids[j]]
                    d = node_id_vals[d_ids[k]]
                    if (o,d) in link_od_dict:
                        dist = o_distances[j] + d_distances[k]
                        if dist < min_dist:
                            damaged_id = link_od_dict[(o,d)]
                            od_pair = [o, d]
                            min_dist = dist
            if damaged_id > 0:
                if od_pair[0] == link.loc[damaged_id,"ONodeID"]:#順方向に進んでいる場合
                    if link_satellite_lane_vals[i] < link.loc[damaged_id,"LeftCarNum"]:
                        link_satellite_lane_vals[i] = 0
                    if damaged_id in change_link_dict:
                        change_link_dict[damaged_id][0] = link_satellite_lane_vals[i]-link.loc[damaged_id, "LeftCarNum"]
                    else:
                        change_link_dict[damaged_id] = [link_satellite_lane_vals[i] - link.loc[damaged_id, "LeftCarNum"], 0]
                    link.loc[damaged_id,"LeftCarNum"] = link_satellite_lane_vals[i]
                    #link.loc[damaged_id,"RightCarNum"] = link_satellite_right_vals[i]
                else:
                    if link_satellite_lane_vals[i] < link.loc[damaged_id,"RightCarNum"]:
                        link_satellite_lane_vals[i] = 0
                    if damaged_id in change_link_dict:
                        change_link_dict[damaged_id][1] = link_satellite_lane_vals[i]-link.loc[damaged_id, "RightCarNum"]
                    else:
                        change_link_dict[damaged_id] = [0, link_satellite_lane_vals[i] - link.loc[damaged_id, "RightCarNum"]]
                    #link.loc[damaged_id,"LeftCarNum"] = link_satellite_right_vals[i]
                    link.loc[damaged_id,"RightCarNum"] = link_satellite_lane_vals[i]

    change_link_vals = []
    for k,v in change_link_dict.items():
        tmp = link.loc[link["ID"] == k, ["ID", "ONodeID", "DNodeID", "LeftCarNum", "RightCarNum", "OLon", "OLat", "DLon", "DLat"]].values.tolist() + v
        change_link_vals.append(tmp)
    change_link = pd.DataFrame(change_link_vals,
        columns=["ID", "ONodeID", "DNodeID", "LeftCarNum", "RightCarNum", "OLon", "OLat",
                 "DLon", "DLat","LeftCarDiff", "RightCarDiff"])
    link.to_csv(tmp_input_dir+"link.csv", index=False)
    change_link.to_csv(tmp_input_dir+"change_link.csv",index=False)
    
    #connectivity.csv
    create_input_hongo.HongoInput.create_connectivity(link, tmp_input_dir+"connectivity.csv",)
    
    #signal.csv
    signal = pd.read_csv(input_obj.input_dir+"signal.csv")
    signal[["ID","NodeID"]] = signal[["ID","NodeID"]].astype(int)
    signal.to_csv(tmp_input_dir+"signal.csv", index=False)
    
    return input_obj

def create_od_hongo(
    tmp_agent_file: str,
    node_mesh: dict[int, int],
    hongo_meshes: list[int],
    matrix: pd.DataFrame,
    start_time: int,
    end_time: int,
    initial_agent_id: int
) -> int:
    """
    OD行列からエージェント情報を生成し、CSVに保存する。

    Args:
        tmp_agent_file (str): エージェント情報保存先ファイル。
        node_mesh (dict[int, int]): ノードID対応辞書。
        hongo_meshes (list[int]): Hongo計算対象メッシュIDリスト。
        matrix (pd.DataFrame): OD行列。
        start_time (int): シミュレーション開始時刻。
        end_time (int): シミュレーション終了時刻。
        initial_agent_id (int): 初期エージェントID。

    Returns:
        int: 最終エージェントID。
    """
    #node_mesh  key: nodeID_hongo, value: nodeID_mfdrl
    mesh_node: dict[int, list[int]] = dict()
    for k,v in node_mesh.items():
        if v in mesh_node:
            mesh_node[v].append(k)
        else:
            mesh_node[v] = [k]
            
    vals = []
    aid = initial_agent_id
    target_mesh_set = set(hongo_meshes)
    car_ratio = 0.7#歩行者に対する自動車の割合
    for i in matrix.index:
        o_mesh = int(matrix.loc[i,"k"])
        d_mesh = int(matrix.loc[i,"a"])
        if (o_mesh not in target_mesh_set) and (d_mesh not in target_mesh_set):#OまたはDがHongoメッシュに入っているものを抽出する．Hongoメッシュに入っていない方はHongoメッシュの隣接メッシュになる．
            continue
        for j in range(int(matrix.loc[i, "vod"])):
            o_node_id = mesh_node[o_mesh][np.random.randint(0, len(mesh_node[o_mesh]))]
            d_node_id = mesh_node[d_mesh][np.random.randint(0, len(mesh_node[d_mesh]))]
            sim_time = np.random.randint(start_time, end_time)
            mode = 0
            if np.random.rand() < car_ratio:
                mode = 3#徒歩
            vals.append([aid, sim_time, mode, o_node_id, d_node_id])
            
            aid += 1
            
    agent = pd.DataFrame(vals, columns=["ID","StartTime","Mode","ONodeID","DNodeID"])
    agent.to_csv(tmp_agent_file, index=False)
    return aid
    
    
def get_simulation(
    input_dir: str,
    output_dir: str,
    start_time: int,
    end_time: int,
    rect_plane_num: int
) -> 'hongo.Simulation':
    """
    シミュレーションオブジェクトを生成する。

    Args:
        input_dir (str): 入力ディレクトリ。
        output_dir (str): 出力ディレクトリ。
        start_time (int): シミュレーション開始時刻。
        end_time (int): シミュレーション終了時刻。
        rect_plane_num (int): 平面直角座標系番号。

    Returns:
        hongo.Simulation: シミュレーションオブジェクト。
    """
    #シミュレーションオブジェクトを作って返す
    timestep = 1.0
    
    noOutput = False#デバッグ用（出力なし）
    
    sim = hongo.Simulation(output_dir, start_time, end_time, timestep, noOutput, rect_plane_num)
    
    sim.readData(input_dir)
    sim.initialize()
    
    return sim
    
def main_hongo(
    sim: 'hongo.Simulation',
    step_num: int,
    end_step_num: int,
    mesh_gdf: gpd.GeoDataFrame,
    mfdrl_link: pd.DataFrame,
    node_mesh: dict[int, int],
    finish_agents_set: set[int]
) -> tuple[dict[tuple[int, int], int], set[int]]:
    """
    シミュレーションを実行し、終了エージェント数を集計する。

    Args:
        sim (hongo.Simulation): シミュレーションオブジェクト。
        step_num (int): 現在のステップ番号。
        end_step_num (int): 終了ステップ番号。
        mesh_gdf (gpd.GeoDataFrame): メッシュGeoDataFrame。
        mfdrl_link (pd.DataFrame): MFD-RLリンク情報。
        node_mesh (dict[int, int]): ノードID対応辞書。
        finish_agents_set (set[int]): 終了済みエージェントID集合。

    Returns:
        tuple[dict[tuple[int, int], int], set[int]]: 終了ODペアごとのエージェント数と終了済みエージェントID集合。
    """
    #メッシュ内終了トリップ→トリップ終了
    #メッシュ外終了トリップ→境界ノードに到達したことをMFD-RLに教えて終了
    outputInterval = 1
    update_interval = 90  #時空間ネットワークおよび旅行時間更新タイミングの時間間隔（ステップ）
    
    target_meshs = set([v for v in node_mesh.values()])
    finish_count: dict[tuple[int, int], int] = {(int(mfdrl_link.loc[i,"O"]), int(mfdrl_link.loc[i,"D"])):0 for i in mfdrl_link.index if (mfdrl_link.loc[i, "O"] in target_meshs) and (mfdrl_link.loc[i, "D"] in target_meshs)}#key: (oNodeId,dNodeId), value: finish_count

    while ((not sim.isFinish()) and (step_num < end_step_num)):
        finish_agents = sim.calculation()
        if not len(finish_agents)==0:
            for i in range(len(finish_agents)):
                agent = finish_agents[i]
                if agent[0] in finish_agents_set:
                    continue
                o_node,d_node = agent[1]
                finish_agents_set.add(agent[0])
                finish_count[(node_mesh[o_node],node_mesh[d_node])] += 1#周辺メッシュへ移動したエージェントの数を記録
        
        if (step_num % outputInterval == 0):
            sim.writeAgentOutput(True, True)#destination, location
            
        if (step_num % update_interval == 0):
            sim.writeLinkOutput()
            sim.updateNetwork()
            
        step_num += 1
    if sim.isFinish():
        sim.close()
        
    # まだトリップが終わっていないagentについても境界を越えているかをチェック
    tmp_agents = sim.getTmpAgents()
    for i in range(len(tmp_agents)):
        agent = tmp_agents[i]
        if agent[0][0] in finish_agents_set:
            continue
        o_node,d_node = agent[0][1]
        p = shapely.geometry.Point(agent[1])
        o_mesh = int(node_mesh[o_node])
        d_mesh = int(node_mesh[d_node])
        if mesh_gdf.loc[(mesh_gdf["nodeID"] == d_mesh),"geometry"].values[0].contains(p):
            finish_count[(o_mesh, d_mesh)] += 1
            finish_agents_set.add(agent.getId())
        
    return finish_count, finish_agents_set

def modify_matrix(finish_count: dict[tuple[int, int], int], matrix: pd.DataFrame) -> pd.DataFrame:
    """
    ODペアごとの流量でOD行列を更新する。

    Args:
        finish_count (dict[tuple[int, int], int]): ODペアごとの流量。
        matrix (pd.DataFrame): OD行列。

    Returns:
        pd.DataFrame: 更新後のOD行列。
    """
    for od, flow in finish_count.items():
        i = matrix.index[((matrix["k"]==od[0])&(matrix["a"]==od[1]))][0]
        matrix.loc[i,"flow"] = flow
    return matrix


def pickle_dump(obj: object, path: str) -> None:
    """
    オブジェクトをpickle形式で保存する。

    Args:
        obj (Any): 保存するオブジェクト。
        path (str): 保存先パス。
    """
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)
        
def pickle_load(path: str) -> object:
    """
    pickleファイルからオブジェクトを読み込む。

    Args:
        path (str): 読み込み元パス。

    Returns:
        Any: 読み込まれたオブジェクト。
    """
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
    return obj
    


if __name__ == "__main__":
    import os
    import sys
    
    print(os.path.dirname(__file__))
    sys.exit()
    
    
    start_time = 3600*10
    end_time = 3600*11
    PLANE_EPSG = {i: i+2442 for i in range(1, 20)}
    
    plane_num = 4#平面直角座標
    epsg = PLANE_EPSG[plane_num]
    
    hongo_meshes = [359]#Hongo計算対象のnodeID
    mfdrl_timestep = 90
    sim = get_simulation("/Users/dogawa/Desktop/BIN/新道路システム開発/simulation/Hongo/input/359/", "/Users/dogawa/Desktop/BIN/新道路システム開発/simulation/Hongo/output/359/", start_time, end_time, plane_num)
    sim.addAgent("/Users/dogawa/Desktop/BIN/新道路システム開発/simulation/Hongo/input/359/agent36000.csv")
    ## タイムステップごとのシミュレーションの実行
    finish_agents_set = set()
    initial_agent_id = 1
    outputInterval = 1
    update_interval = 90  #時空間ネットワークおよび旅行時間更新タイミングの時間間隔（ステップ）
    
    noOutput = False#デバッグ用（出力なし）
        
    step_num = 0
    
    while ((not sim.isFinish())):
        finish_agents = sim.calculation()
        print(type(finish_agents))
        if not len(finish_agents)==0:
            print(len(finish_agents))
            for i in range(len(finish_agents)):
                agent = finish_agents[i]
                if agent[0] in finish_agents_set:
                    print(agent[0])
                    continue
                o_node,d_node = agent[1]
                finish_agents_set.add(agent[0])
        
        if (step_num % outputInterval == 0):
            sim.writeAgentOutput(True, True)#destination, location
            
        if (step_num % update_interval == 0):
            sim.writeLinkOutput()
            sim.updateNetwork()
            
        step_num += 1
    if sim.isFinish():
        sim.close()

