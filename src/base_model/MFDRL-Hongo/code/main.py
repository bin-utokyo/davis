'''シミュレーションの実行 MFD-RL Hongo'''
import os
import sys
import json
import pickle
from typing import Optional

import hongopy
import mfdrlpy

def main(config: str, initialize: bool = True) -> None:
    """
    メイン処理を実行する関数。

    コマンドライン引数で指定された設定ファイルを読み込み、
    シミュレーションの初期化・実行・結果出力までを行う。

    Args:
        config (str): 設定ファイルのパス。
        initialize (bool): Trueの場合、シミュレーションの初期化を行う。
                           Falseの場合、既存の設定に基づいてシミュレーションを実行する。

    Raises:
        SystemExit: 初期化時や処理終了時にシステムを終了する場合。
    """
    print("Start to load config.")
    conf: dict = json_load(config)
    print("conf:", conf)
    
    PLANE_EPSG: dict[int, int] = {i: i+2442 for i in range(1, 20)}
    
    input_dir_hongo: str = conf["input_dir_hongo"]
    output_dir_hongo: str = conf["output_dir_hongo"]
    
    input_dir_mfdrl: str = conf["input_dir_mfdrl"]
    output_dir_mfdrl: str = conf["output_dir_mfdrl"]

    output_dir: str = conf["output_dir"]
    
    mesh_file_mfdrl: str = conf["mesh_file_mfdrl"]
    if "geo_mesh_file_mfdrl" in conf:
        geo_mesh_file_mfdrl = conf["geo_mesh_file_mfdrl"]
    else:
        geo_mesh_file_mfdrl = os.extsep.join(mesh_file_mfdrl.split(os.extsep)[0:-1]+["geojson"])
        
    node_mesh_file: str = conf["node_mesh_file"]  # key: nodeID_hongo, value: nodeID_mfdrl  (pickle file)

    bb_coord: list = conf["bb_coord"]
    
    plane_num: int = conf["plane_num"]  # 平面直角座標
    epsg: int = PLANE_EPSG[plane_num]

    start_time: int = conf["start_time"]
    end_time: int = conf["end_time"]
    
    hongo_meshes: list = conf["hongo_meshes"]  # Hongo計算対象のnodeID
    population_file: str = conf["population_file"]
    
    satellite_dir: Optional[str] = conf["satellite_dir"]
    if satellite_dir is None:
        node_satellite_file: Optional[str] = None
        link_satellite_file: Optional[str] = None
    else:
        node_satellite_file = satellite_dir + "node_for_sim.csv"
        link_satellite_file = satellite_dir + "edge_for_sim.csv"
    print("Finish loading config file.")
    
    
    ## インプットの生成
    print("Create input.")
    input_obj_mfdrl = mfdrlpy.get_input_obj(input_dir_mfdrl, epsg, bb_coord)
    input_obj_hongo = hongopy.get_input_obj(input_dir_hongo, epsg, bb_coord)
    if initialize:
        print("Initialize input data.")
        input_obj_mfdrl,geo_mesh_file_mfdrl  = mfdrlpy.create_input_initial(input_obj_mfdrl, mesh_file_mfdrl)
        print("MFD-RL input initialized.")
        input_obj_hongo = hongopy.create_input_initial(input_obj_hongo, geo_mesh_file_mfdrl, node_mesh_file)
        print("Hongo input initialized.")
        print('Finish!')
        sys.exit()

    mesh_string = "-".join([str(x) for x in hongo_meshes])
    time_string = "-".join([str(start_time), str(end_time)])

    tmp_input_dir_mfdrl = input_dir_mfdrl+time_string+"/"+mesh_string+"/"
    tmp_input_dir_hongo = input_dir_hongo+time_string+"/"+mesh_string+"/"
    tmp_output_dir_mfdrl = output_dir_mfdrl+time_string+"/"+mesh_string+"/"
    tmp_output_dir_hongo = output_dir_hongo+time_string+"/"+mesh_string+"/"
    
    try:
        os.makedirs(tmp_input_dir_mfdrl)
    except FileExistsError:
        print("Directories already exist.",tmp_input_dir_mfdrl)
    try:
        os.makedirs(tmp_input_dir_hongo)
    except FileExistsError:
        print("Directories already exist.",tmp_input_dir_hongo)
    try:
        os.makedirs(tmp_output_dir_mfdrl)
    except FileExistsError:
        print("Directories already exist.",tmp_output_dir_mfdrl)
    try:
        os.makedirs(tmp_output_dir_hongo)
    except FileExistsError:
        print("Directories already exist.",tmp_output_dir_hongo)
        
    print("Create Hongo input.")
    # 衛星画像と整合が取れるように，Hongoのネットワークデータを加工する．
    input_obj_hongo = hongopy.create_input(input_obj_hongo, tmp_input_dir_hongo, node_satellite_file, link_satellite_file)
    print("Hongo input creation done.")
    
    
    print("Create MFD-RL input for Hongo mesh",",".join([str(x) for x in hongo_meshes]))
    # 現ステップで使うHongoのインプットのネットワーク（衛星データにより修正済み）からメッシュベースネットワークを作成
    input_obj_mfdrl.read_data(geo_mesh_file_mfdrl, population_file, tmp_input_dir_hongo+"node.csv", tmp_input_dir_hongo+"link.csv")
    #mfdrlpy.copy_param_files(input_dir_mfdrl, tmp_input_dir_mfdrl)
    input_obj_mfdrl = mfdrlpy.create_input(input_obj_mfdrl, tmp_input_dir_mfdrl, node_mesh_file, start_time, end_time)
    mfdrl_code_id_dict = {input_obj_mfdrl.mesh_gdf.loc[i, "KEY_CODE"]: input_obj_mfdrl.mesh_gdf.loc[i,"nodeID"] for i in input_obj_mfdrl.mesh_gdf.index}
    hongo_meshes_nodeid = [mfdrl_code_id_dict[nid] for nid in hongo_meshes]
    print("MFD-RL input creation done.")
    
    
    ## シミュレーションオブジェクトの取得
    #  MFD-RL1ステップにつきHongoを90ステップ(1.5分)
    mfdrl_timestep = 90
    hongo_sim = hongopy.get_simulation(tmp_input_dir_hongo, tmp_output_dir_hongo, start_time, end_time, plane_num)
    mfdrl_sim = mfdrlpy.get_simulation(tmp_input_dir_mfdrl, tmp_output_dir_mfdrl, start_time, end_time, mfdrl_timestep)
    
    node_mesh = pickle_load(node_mesh_file)
    if not isinstance(node_mesh, dict):
        raise TypeError(f"node_mesh_file {node_mesh_file} is not a dict. Please check the file format.")
    
    ## タイムステップごとのシミュレーションの実行
    print("Calculate step by step.")
    finish_agents_set: set[int] = set()
    initial_agent_id = 1
    for step_num in range(0, end_time-start_time, mfdrl_timestep):
        print("    Step num:",step_num)
        #MFD-RLを1ステップ計算
        n_i, matrix = mfdrlpy.main_mfdrl(mfdrl_sim)
        if n_i is None or matrix is None:
            continue
        #MFD-RLのVODからHongoのODを作る．
        initial_agent_id = hongopy.create_od_hongo(tmp_input_dir_hongo+f"agent{start_time+step_num}.csv", node_mesh, hongo_meshes_nodeid, matrix, start_time+step_num, start_time+step_num+mfdrl_timestep, initial_agent_id)
        #Hongoをstep_numからhongo_step回計算
        hongo_sim.addAgent(tmp_input_dir_hongo+f"agent{start_time+step_num}.csv")#Hongoのエージェントを追加
        finish_count, finish_agents_set = hongopy.main_hongo(hongo_sim, step_num, step_num+mfdrl_timestep, input_obj_mfdrl.node_gdf, input_obj_mfdrl.link_gdf, node_mesh, finish_agents_set)
        
        #MFD-RLとHongoの結果を交換（matrixをHongoメッシュとその隣接メッシュについて書き換える）
        new_matrix = hongopy.modify_matrix(finish_count, matrix)
        mfdrl_sim.set_modified_matrix(new_matrix)
    if not hongo_sim.isFinish():
        hongo_sim.close()
    mfdrl_sim.write_result(geo_mesh_file_mfdrl, input_obj_mfdrl)

    #print("Writing geojson results start.")
    #converter_obj = geoconverter.Converter(conf, tmp_output_dir_mfdrl, tmp_output_dir_hongo, input_dir_hongo+"link.csv", epsg)
    #converter_obj.process_data()
    #converter_obj.to_geojson(output_dir)
    #print("Writing geojson results done.")
    print("Finish All Processes!")
    
    
    
def json_dump(obj: dict, path: str) -> None:
    """
    オブジェクトをJSONファイルに保存する。

    Args:
        obj (dict): 保存するオブジェクト。
        path (str): 保存先のファイルパス。
    """
    with open(path, mode="w") as f:
        json.dump(obj, f, ensure_ascii=False)

def json_load(path: str) -> dict:
    """
    JSONファイルからオブジェクトを読み込む。

    Args:
        path (str): 読み込むファイルパス。

    Returns:
        dict: 読み込まれたオブジェクト。
    """
    with open(path, mode="r") as f:
        obj = json.load(f)
    return obj

def pickle_dump(obj: object, path: str) -> None:
    """
    オブジェクトをpickle形式でファイルに保存する。

    Args:
        obj (object): 保存するオブジェクト。
        path (str): 保存先のファイルパス。
    """
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)

def pickle_load(path: str) -> object:
    """
    pickleファイルからオブジェクトを読み込む。

    Args:
        path (str): 読み込むファイルパス。

    Returns:
        object: 読み込まれたオブジェクト。
    """
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
    return obj



if __name__=='__main__':
    args = sys.argv
    if len(args) < 2:
        print("Please set config file.")
        sys.exit(1)
    elif len(args) != 3:
        initialize = True
    else:
        initialize = args[2] == "true"

    main(config=args[1], initialize=initialize)
    sys.exit()

