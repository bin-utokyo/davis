import pandas as pd
import numpy as np

def to_hongo_params(result_paths):
    """
    Convert parameters to Hongo format.
    
    Parameters:
    result_paths (dict): Dictionary with keys 0, 3 or 6 (0: car, 3: ped, 6: bus), each containing the path to the result file.
    Returns:
    pd.DataFrame: DataFrame with parameters and feature names.
    """
    dfs = []
    for k, v in result_paths.items():
        if k not in [0, 3, 6]:
            raise ValueError(f"Invalid key {k} in result_paths. Expected 0, 3, or 6.")
        result = pd.read_csv(v, header=None)
        # 修正尤度の行を削除
        result = result[result[0] != '修正尤度']
        # Columns: [PropName, Coeff]
        tmp = pd.DataFrame({
            "Mode": k,
            "PropName": result[0],
            "Coeff": result[1]
        })
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    return df


def to_hongo_link(link_veh, node, link_ped=None, velocity=None, lanes=None):
    """
    Convert link data to Hongo format.
    
    Parameters:
    link_veh (pd.DataFrame): DataFrame containing vehicle link data with columns LinkID, ONodeID, DNodeID, prop1, prop2, ...
    node (pd.DataFrame): DataFrame containing node data with columns NodeID, Longitude, Latitude.
    link_ped (pd.DataFrame, optional): DataFrame containing pedestrian link data with columns LinkID, ONodeID, DNodeID, prop1, prop2, ...
    velocity (np.ndarray, optional): Default velocity for link_veh. If None, defaults to 30.
    lanes (np.ndarray, optional): Default number of lanes for link_veh. If None, defaults to 1.

    Returns:
    pd.DataFrame: DataFrame with Hongo format link data. Columns: ID,ONodeID,DNodeID,Velocity,LeftCarNum,LeftPedNum,LeftBeltNum,RightCarNum,RightPedNum,RightBeltNum,upNodeLat,upNodeLon,dnNodeLat,dnNodeLon
    pd.DataFrame: DataFrame with Hongo format link prop data. Columns: LinkID,prop1,prop2,...
    """
    node = node.set_index('NodeID', drop=False)

    # 逆方向リンクの対応付
    nids2lid = dict()
    for i in link_veh.index:
        key = (link_veh.loc[i, "ONodeID"], link_veh.loc[i, "DNodeID"])
        if key[0] > key[1]:
            key = (key[1], key[0])
        if key not in nids2lid:
            nids2lid[key] = link_veh.loc[i, "LinkID"]
    if link_ped is not None:
        for i in link_ped.index:
            key = (link_ped.loc[i, "ONodeID"], link_ped.loc[i, "DNodeID"])
            if key[0] > key[1]:
                key = (key[1], key[0])
            if key not in nids2lid:
                nids2lid[key] = link_ped.loc[i, "LinkID"]
    
    # Fill with default values
    link_hongo = pd.DataFrame({"ID": nids2lid.values(), "ONodeID": [k[0] for k in nids2lid.keys()],
                               "DNodeID": [k[1] for k in nids2lid.keys()]})
    link_hongo["Velocity"] = 30
    link_hongo[["LeftCarNum", "LeftPedNum", "LeftBeltNum",
                "RightCarNum", "RightPedNum", "RightBeltNum"]] = 0
    link_hongo.set_index("ID", inplace=True, drop=False)
    
    # Fill with link data
    for i, idx in enumerate(link_veh.index):
        key = (link_veh.loc[idx, "ONodeID"], link_veh.loc[idx, "DNodeID"])
        left = True
        if key[0] > key[1]:
            key = (key[1], key[0])
            left = False
        lid = nids2lid[key]

        link_hongo.loc[lid, "Velocity"] = velocity[i] if velocity is not None else 30
        if left:
            link_hongo.loc[lid, "LeftCarNum"] = lanes[i] if lanes is not None else 2
        else:
            link_hongo.loc[lid, "RightCarNum"] = lanes[i] if lanes is not None else 2
    # Fill with pedestrian link data
    if link_ped is not None:
        for i, idx in enumerate(link_ped.index):
            key = (link_ped.loc[idx, "ONodeID"], link_ped.loc[idx, "DNodeID"])
            left = True
            if key[0] > key[1]:
                key = (key[1], key[0])
                left = False
            lid = nids2lid[key]

            if left:
                link_hongo.loc[lid, "LeftPedNum"] = 1
            else:
                link_hongo.loc[lid, "RightPedNum"] = 1
    else:
        link_hongo["LeftPedNum"] = 1
        link_hongo["RightPedNum"] = 1
    # Set coordinates
    link_hongo["OLat"] = link_hongo["ONodeID"].map(node["Latitude"])
    link_hongo["OLon"] = link_hongo["ONodeID"].map(node["Longitude"])
    link_hongo["DLat"] = link_hongo["DNodeID"].map(node["Latitude"])
    link_hongo["DLon"] = link_hongo["DNodeID"].map(node["Longitude"])

    # Fill link properties
    columns = link_veh.columns.to_list()
    columns.remove("ONodeID")
    columns.remove("DNodeID")

    lids = link_hongo["ID"].unique()
    link_prop = link_veh[columns].copy()
    link_prop = link_prop[link_veh["LinkID"].isin(lids)]

    return link_hongo, link_prop

def to_hongo_node(nodes):
    """
    Convert node data to Hongo format.
    
    Parameters:
    nodes (pd.DataFrame | list[pd.DataFrame]): DataFrame containing node data with columns NodeID, Longitude, Latitude.
    
    Returns:
    pd.DataFrame: DataFrame with Hongo format node data. Columns: ID,Lat,Lon
    """
    if isinstance(nodes, list):
        nodes = pd.concat(nodes, ignore_index=True)
    duplicated = nodes.duplicated(subset='NodeID')
    nodes = nodes[~duplicated]

    hongo_nodes = pd.DataFrame({
        "ID": nodes["NodeID"],
        "Lat": nodes["Latitude"],
        "Lon": nodes["Longitude"]
    })
    return hongo_nodes

