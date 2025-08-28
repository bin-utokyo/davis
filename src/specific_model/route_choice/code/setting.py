# -*- coding: utf-8 -*-

#import time
import numpy as np
import heapq
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, csr_array
from pyproj import Proj, transform


class Network:
    def __init__(self, node_table: pd.DataFrame, link_table: pd.DataFrame):
        self.mode = 'walk'
        self.n_node = len(node_table)
        self.n_link = len(link_table)
        self.node_list = node_table['NodeID'].values.tolist()
        self.link_list = link_table['LinkID'].values.tolist()
        self.link_start = link_table['ONodeID'].values.tolist()
        self.link_end = link_table['DNodeID'].values.tolist()

        self.f_name = [x for x in link_table.columns.tolist() if x not in ['LinkID', 'ONodeID', 'DNodeID']]  # リンク属性の名前
        self.attr = {x: link_table[x].values.tolist() for x in self.f_name}  # リンク属性の値

        self.link_length = Network.get_link_length(link_table, node_table)  # リンクの長さ
        self.ev_list = self.succeed_link()  # 各ノードを始点とするリンクの集合のリスト
        self.sp_mat = np.zeros((self.n_node, self.n_node))
        self.sp_step = None
        self.adj_link = None

        if not Network.check_attr(self.attr, min_thresh=-10, max_thresh=10):
            print("Warning: Some link attributes are out of the expected range (-10, 10).")

    def succeed_link(self) -> list[list[int]]:
        ## 各ノードを始点とするリンクの集合
        lst = []
        lst_app = lst.append
        #lst = [[l + 1 for l, start in enumerate(self.link_start) if start == v] for v in self.node_list]

        for v in self.node_list:
            links = [l + 1 for l, start in enumerate(self.link_start)
                        if start == v  
                        ]
            lst_app(links)

        return lst

    def odmat(self, odlst: pd.DataFrame, d_list: list[int]) -> csr_array:
        row = []
        col = []
        val = []
        r_ext = row.extend
        c_ext = col.extend
        v_ext = val.append

        for l in range(len(odlst)):
            o = odlst.iat[l,0]
            nexts = self.func2(self.node_list,o)
            r_ext(nexts)
            d = odlst.iat[l, 1]
            nexts = self.func1(d_list,d)
            c_ext(nexts)
            nexts = odlst.iat[l, 2]
            v_ext(nexts)

        return coo_matrix((val, (row, col)), shape=(self.n_link + self.n_node+1, len(d_list))).tocsr()


    def previous_link(self) -> list[list[int]]:
        ## 各ノードを終点とするリンクの集合
        lst = []
        lst_app = lst.append
        # lst = [[l + 1 for l, start in enumerate(self.link_start) if start == v] for v in self.node_list]

        for v in self.node_list:
            links = [l + 1 for l, end in enumerate(self.link_end)
                        if end == v  # 徒歩の場合
                        ]
            lst_app(links)

        return lst

    def adj_matrix(self) -> csr_array:
        ## ノード間隣接行列
        # Dijkstra法の計算に使用
        length = self.link_length.tolist()
        #if self.mode == 'car':
        #    avl = self.link_avail
        #    length = [l if avl[ai] == 0 else 0 for ai, l in enumerate(length)]
        row = [v - 1 for v in self.link_start]
        col = [v - 1 for v in self.link_end]
        return coo_matrix((length, (row, col)), shape=(self.n_node, self.n_node)).tocsr()

    def adj_mat(self, method: str) -> csr_array:
        ## リンク間隣接行列，ダミーリンクは含まない
        # RecursiveLogit等で使用
        row = []
        col = []
        r_ext = row.extend
        c_ext = col.extend

        for l in range(self.n_link):
            v = self.link_end[l]
            nexts = self.func1(self.link_start,v)
            r_ext([l] * len(nexts))
            c_ext(nexts)
            if method != 'ndrl' and method != 'nldrl'and method != 'setdrl' and method != 'rl':
                nexts = list(map(lambda y: y + self.n_link,self.func1(self.node_list, v)))
                r_ext([l] * len(nexts))
                c_ext(nexts)

        if method!='ndrl' and method !='nldrl'and method != 'setdrl' and method != 'rl':
            for l in range(self.n_node):
                v = self.node_list[l]
                nexts = self.func1(self.link_start,v)
                r_ext([l+self.n_link] * len(nexts))
                c_ext(nexts)

        row = [l for l in row]
        col = [l for l in col]
        val = [1] * len(row)

        if method == 'ndrl' or method == 'nldrl'or method == 'setdrl' or method == 'rl':
            return coo_matrix((val, (row, col)), shape=(self.n_link , self.n_link)).tocsr()
        else:
            return coo_matrix((val, (row, col)), shape=(self.n_link+self.n_node, self.n_link+self.n_node)).tocsr()

    def func1(self, list: list, value) -> list[int]:
        return [i for i, x in enumerate(list) if x == value]
    def func2(self, list: list, value) -> list[int]:
        return [i+self.n_link for i, x in enumerate(list) if x == value]


    def adj_mat_link(self) -> csr_array:
        ## リンク間隣接行列，ダミーリンクは含まない
        # RecursiveLogit等で使用
        row = []
        col = []
        r_ext = row.extend
        c_ext = col.extend

        for l in range(self.n_link):
            v = self.link_end[l]
            nexts = self.ev_list[v - 1]
            r_ext([l + 1] * len(nexts))
            c_ext(nexts)

        row = [l - 1 for l in row]
        col = [l - 1 for l in col]
        val = [1] * len(row)

        return coo_matrix((val, (row, col)), shape=(self.n_link, self.n_link)).tocsr()

    def dijkstra(self, adj: csr_matrix, ori: int) -> tuple[list[int | None], list[float]]:
        ### scipyに任せたのでおやくごめん
        ## ダイクストラ法
        # 始点oriから各ノードまでの最短距離を出力

        # 初期化
        N = self.n_node
        min_cost = [float('inf')] * N
        checked = [False] * N  # 確定したか否か
        forward: list[int | None] = [None] * N  # 最短経路をたどるための直前ポインタ
        q = []  # 優先度付きキュー，各要素はタプル(コスト，ノードID)，未確定ノードのうちコストが有限のもの

        min_cost[ori - 1] = 0
        forward[ori - 1] = ori
        heapq.heappush(q, (0, ori))

        while not all(checked):
            # 全てのノードを探索し確定するまで繰り返し
            _, v = heapq.heappop(q)  # 費用が最小のノードをvとする
            checked[v - 1] = True  # ノードvを確定

            for i in range(N):
                dist = adj[v - 1, i]
                if (not checked[i]) and dist != 0:  # 未確定隣接ノードについて
                    tmp_cost = min_cost[v - 1] + dist
                    if tmp_cost < min_cost[i]:
                        # コストとポインタを更新
                        min_cost[i] = tmp_cost
                        forward[i] = v
                        heapq.heappush(q, (min_cost[i], i + 1))

        return forward, min_cost

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

    def sampling_weight(self, ori: int, des: int) -> np.ndarray:
        ## ランダムウォークで経路生成する際の，各リンクに対する重み
        # O-Dペアに固有，全リンクに関する配列を返す
        sp_mat = self.sp_mat
        sp0 = sp_mat[ori - 1, des - 1]  # SP(ori, des)

        def xl(link):
            v = self.link_start[link - 1]
            w = self.link_end[link - 1]
            sp1 = sp_mat[ori - 1, v - 1]
            sp2 = sp_mat[w - 1, des - 1]
            length = self.link_length[link - 1]
            return sp0 / (sp1 + length + sp2)

        xl_array = np.array([xl(l + 1) for l in range(self.n_link)])

        # 形状パラメータ
        b1 = 10  # b1:大 → 最短経路に近い経路をより重く
        b2 = 1  # 固定
        omega = 1 - (1 - xl_array ** b1) ** b2  # Kumaraswamy分布

        return omega
    
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
        

'''
class TimeSpaceNW:
    ### 時間構造化ネットワーク
    ## 結局使わなかったと思われる

    def __init__(self, network, ori, des, t_max):
        self.network = network
        self.ori = ori  # 起点
        self.des = des  # 終点
        self.t_max = t_max  # 時間制約（要検討）
        AT = network.n_link * t_max + 2
        self.I_t = lil_matrix((AT, 1))  # 各状態の存在可能性ダミー行列
        self.Delta = self.prism_constraint(
            network.sp_step, self.link_connection(network.adj_link))  # 時空間prism制約行列

    @classmethod
    def construct_network(cls, network, data_set):
        ## 時空間ネットワークの構築
        ori = data_set.ori
        des = data_set.des
        steps = [p.n_link for p in data_set.path_list]
        t_max = max(steps) + 10  # 時間制約Tを指定（要検討）
        return cls(network, ori, des, t_max)

    def link_connection(self, adj_link):
        ## リンク間の空間的接続関係delta
        A = self.network.n_link
        delta = lil_matrix((A + 2, A + 2))  # (A+2, A+2)疎行列
        delta[1:-1, 1:-1] = adj_link  # adj_link: (A, A)疎行列

        # 始点，終点ダミーリンクとの接続
        lst_o = self.network.ev_list[self.ori - 1]  # 起点ノードoriの次のリンク
        lst_d = self.network.bv_list[self.des - 1]  # 終点ノードdesの前のリンク
        val_o = [0] + [1 if ia + 1 in lst_o else 0 for ia in range(A)] + [0]
        val_d = [0] + [1 if ia + 1 in lst_d else 0 for ia in range(A)] + [1]
        val_d = [[v] for v in val_d]
        delta[0, :] = val_o
        delta[:, -1] = val_d

        return delta.tocsr()

    def prism_constraint(self, sp_step, delta):
        t0 = time.time()
        ## 時空間上のprism制約Delta_t(a'|a)
        T = self.t_max
        A = self.network.n_link
        l_list = self.network.link_list
        start = self.network.link_start
        end = self.network.link_end

        # 最短経路step数D^o(a), D^d(a)
        minstep_ori = np.array([sp_step[self.ori - 1, start[a - 1] - 1] for a in l_list])
        minstep_des = np.array([sp_step[end[a - 1] - 1, self.des - 1] - 1 for a in l_list])

        # 状態s_t=(t,a)の存在可能性I_t(a)
        md_ori = csr_matrix([[(step <= t) for step in minstep_ori] for t in range(1, T + 1)])
        md_des = csr_matrix([[(step <= T - t) for step in minstep_des] for t in range(1, T + 1)])
        exist_mat = md_ori.multiply(md_des).tocsr()  # exist_mat: (T, A)

        self.I_t[1:-1, :] = exist_mat.reshape(A * T, 1)
        self.I_t[0, :] = 1
        self.I_t[-1, :] = 1

        # アークe_t=(s_t,s_(t+1))の存在可能性Delta_t
        # sparse matrixで保持
        AT = 2 + A * T  # o,dを含む全状態数
        Delta = lil_matrix((AT, AT))
        Delta[0, 1: A + 1] = delta[0, 1:-1]  # originからの接続
        for t in range(T):
            st = t * A + 1
            en = st + A
            It = exist_mat[t, :].T.tocsr()
            if t < T - 1:
                It1 = exist_mat[t + 1, :]
                Delta[st:en, en: en + A] = (It.multiply(delta[1:-1, 1:-1])).multiply(It1)
            Delta[st:en, -1] = delta[1:-1, -1].multiply(It)  # destinationへの接続

        t1 = time.time()
        print('({0}, {1})    T = {2}    '.format(self.ori, self.des, T), end='')
        print('{} sec'.format(t1 - t0))

        return Delta.tocsr()

    def choice_result(self, path_list):
        ## リンク選択結果を行列に格納
        T = self.t_max
        A = self.network.n_link
        AT = 2 + A * T
        mat = lil_matrix((AT, AT))

        for p in path_list:
            lst = p.link_list
            state_list = [i * A + k for i, k in enumerate(lst)]
            k_list = [0] + state_list
            a_list = state_list + [AT - 1]
            val = [1] * len(k_list)
            mat = mat + coo_matrix((val, (k_list, a_list)), (AT, AT))

        return mat.tocsr()

'''
