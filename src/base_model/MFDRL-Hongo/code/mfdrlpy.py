from typing import Any, Optional
import shutil
import numpy as np
import pandas as pd
import random
import math

import create_input_mfdrl

import geopandas as gpd
import pickle


class MFDRL:
    def __init__(self, input_dir: str, output_dir: str, start_time: int, end_time: int, timestep: int) -> None:
        """
        Args:
            input_dir (str): 入力ディレクトリのパス。
            output_dir (str): 出力ディレクトリのパス。
            start_time (int): シミュレーション開始時刻（秒）。
            end_time (int): シミュレーション終了時刻（秒）。
            timestep (int): タイムステップ（秒）。
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.start_time = start_time
        self.end_time = end_time
        self.timestep = timestep

        self.t = 0
        self.TS1 = 23 - 7 + 1 + 1  # target time of activity simulation is 7:00-23:00. Making choices TS1 times.
        self.TS2 = int((end_time - start_time) / timestep) + 1  # time step of route choice. Making choices TS2 times.

        self.n_i: np.ndarray
        self.n_i_total: np.ndarray
        self.n_i_out: np.ndarray
        self.wait_timeline: np.ndarray
        self.link_flow: np.ndarray
        self.link_vod: np.ndarray
        self.read_data()

    def read_data(self) -> None:
        """
        必要なデータファイルを読み込み、モデルの初期化を行う。
        """
        print("reading data")
        # File name
        node_file = self.input_dir + "node.csv"  # KEY_CODE,nodeID,lon,lat, Population,edge
        link_file = self.input_dir + "link.csv"  # linkID,O,D,Olon,Olat,Dlon,Dlat,limit
        RLparameter_file1 = self.input_dir + "Activity_params.csv"
        RLparameter_file2 = self.input_dir + "Route_params.csv"
        MFDparameter_file = self.input_dir + "mfd_params3.csv"  # a,b,d_link,elevation,population,DisToSea,a_init,b_init

        self._read_param_file(RLparameter_file1, RLparameter_file2, MFDparameter_file)
        self._read_node_file(node_file)
        self._read_link_file(link_file)
        self._initialize_variables()

    def set_prob(self) -> None:
        """
        遷移確率行列を計算し、self.pに格納する。
        """
        print("setting probability")
        p_n = np.zeros((self.N, self.TS2, self.N, self.N), dtype="float64")
        p = np.zeros((self.TS2, self.N, self.N), dtype="float64")

        n_rep = 10
        n_list = np.random.choice(range(0, self.N - 1), n_rep, replace=False)
        # print(n_list)
        for n_idx, n in enumerate(n_list):  # for all mesh #self.N-1
            print(str(n_idx + 1) + "/" + str(n_rep))
            p1 = np.zeros((self.N, self.TS1, self.N, self.N), dtype="float64")
            ## function for calculating transition probability
            theta1 = self.x1[len(self.x1) - 1]  # theta is discount factor

            # -- utility function matrix M--#
            M = np.zeros((self.TS1, self.N, self.N), dtype="float64")
            self.b1_home_yoru[n, 13:17, :self.N - 1, n] = np.exp(1)  # 20:00-23:00
            self.b1_home_asa[n, 0:3, n, n] = np.exp(1)  # 7:00-8:00

            for ts in range(0, self.TS1):
                if ts == self.TS1 - 1:
                    Its = self.ITS1  # adjacency matrix for terminal step TS
                else:
                    Its = self.I1[n, :, :]  # spatial adjacency matrix
                Mts = Its * (self.b1_length ** self.x1[0]) * (self.b1_home_yoru[n, ts, :, :] ** self.x1[1]) * (
                            self.b1_home_asa[n, ts, :, :] ** self.x1[2]) ** (self.b1_stay[ts, :, :] ** self.x1[
                    3])  # element product (not matrix product) #*(self.b1_population**0)
                M[ts, :, :] = Mts

            # -- value function z --#
            # t=t-1, calculate z(t)=Mz(t+1)+b until t=0 (backward calculation)
            z = np.ones((self.TS1 + 1, self.N), dtype="float64")  # exp(V_d)
            for t in reversed(range(0, self.TS1)):
                zi = np.dot(M[t, :, :],
                            (z[t + 1] ** theta1))  # matrix product. Value function for all spatial nodes at ts.
                z[t, :] = (zi == 0) * 1 + (zi != 0) * zi

            # -- calculating probability --#
            for ts in range(0, self.TS1):
                zt = np.array(z[ts + 1, :] ** theta1).reshape(self.N, 1)
                ZD = np.tile(zt, (1, self.N)).T
                Mz = np.multiply(np.dot(M[ts, :, :], zt) != 0, np.dot(M[ts, :, :], zt)) + (np.dot(M[ts, :, :], zt) == 0)
                MZ = np.tile(Mz, (1, self.N))
                pt = (M[ts, :, :] * ZD) / MZ
                p1[n, ts, :, :] = pt

            # must go to absorbing state at TS
            pt = np.zeros((self.N, self.N), dtype=int)
            pt[:, self.N - 1] = 1
            p1[n, self.TS1 - 1, :, :] = pt

            # print(p1[n,3,:,:])

            # 全てのODについてやると確率の計算が終わらないのでODをサンプリングする
            n_sample = 100
            mat = p1[n, int(self.start_time / 3600 - 7), :, :]
            od_list = []
            for smpl in range(0, n_sample):
                idx = np.argpartition(mat.flatten(), -smpl - 1)[-smpl - 1]
                od_list.append(np.unravel_index(idx, mat.shape))
            # print("od_list")
            # print(od_list)

            for od2 in range(len(od_list)):
                o2 = od_list[od2][0]
                d2 = od_list[od2][1]
                # print(str(o2)+", "+str(d2))
                p2 = np.zeros((self.TS2, self.N, self.N), dtype="float64")

                ## function for calculating transition probability
                theta2 = self.x2[len(self.x2) - 1]  # theta is discount factor
                # print(theta2)
                # adjacency matrix (0 if link capacity ==0)
                for i in range(0, self.Lnum):  # for all link except links to an absorbing node
                    k = self.matrix.k[i]
                    a = self.matrix.a[i]
                    if self.limit[i] == 0:
                        self.I2[k - 1, a - 1] = 0  # 0 if capacity of link ka == 0

                # -- utility function matrix M--#
                M = np.zeros((self.TS2, self.N, self.N), dtype="float64")
                if o2 == d2:
                    self.b2_stay_o[n, :, o2, d2] = np.exp(1)

                for ts in range(0, self.TS2):
                    if ts == self.TS2 - 1:
                        ITS2 = np.zeros((self.N, self.N), dtype=int)
                        ITS2[d2, self.N - 1] = 1
                        Its = ITS2  # adjacency matrix for terminal step TS
                    else:
                        Its = self.I2  # spatial adjacency matrix
                    Mts = Its * (self.b2_length ** self.x2[0]) * (self.b2_stay_o[n, ts, :, :] ** self.x2[1]) * (
                                self.b2_stay[:, :] ** self.x2[
                            2])  # element product (not matrix product) #*(self.b2_population**0)
                    M[ts, :, :] = Mts
                # print(M[0,:,:])

                # -- value function z --#
                # t=t-1, calculate z(t)=Mz(t+1)+b until t=0 (backward calculation)
                z = np.ones((self.TS2 + 1, self.N), dtype="float64")  # exp(V_d)
                for t in reversed(range(0, self.TS2)):
                    zi = np.dot(M[t, :, :],
                                (z[t + 1] ** theta2))  # matrix product. Value function for all spatial nodes at ts.
                    z[t, :] = (zi == 0) * 1 + (zi != 0) * zi
                # print(z[1,:])

                # -- calculating probability --#
                for ts in range(0, self.TS2):
                    zt = np.array(z[ts + 1, :] ** theta2).reshape(self.N, 1)
                    ZD = np.tile(zt, (1, self.N)).T
                    Mz = np.multiply(np.dot(M[ts, :, :], zt) != 0, np.dot(M[ts, :, :], zt)) + (
                                np.dot(M[ts, :, :], zt) == 0)
                    MZ = np.tile(Mz, (1, self.N))
                    pt = (M[ts, :, :] * ZD) / MZ
                    p2[ts, :, :] = pt

                # must go to absorbing state at TS
                pt = np.zeros((self.N, self.N), dtype=int)
                pt[:, self.N - 1] = 1
                p2[self.TS2 - 1, :, :] = pt

                # print(p1[n,int(self.start_time/3600 -7),o2,d2])
                # print(p2[0,:,:])
                p_n[n, :, :, :] += p1[n, int(self.start_time / 3600 - 7), o2, d2] * p2[:, :, :]

            # print(p_n[n,0,:,:])
            p[:, :, :] += p_n[n, :, :, :] / n_rep

        for tm in range(0, self.TS2):
            for ax in range(0, self.N - 1):
                p[tm, ax, :] = p[tm, ax, :] / np.sum(p[tm, ax, :])  # 遷移確率の和が1になるように等倍

        p[self.TS2 - 1, :, self.N - 1] = 1
        # for t in range(0,self.TS2):
        # print(np.sum(p[t,:,:], axis=0))

        self.p = p

    def assign(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        需要・供給・フロー・人口分布の更新を行う。

        Returns:
            tuple[np.ndarray, pd.DataFrame]: 更新後の人口分布とリンク情報のデータフレーム。
        """
        # demand
        # calculation of VOD
        t = self.t
        ratio_car = 0.08  # 自動車分担率0.08（東京都区部想定）
        # print("ratio_car: "+str(ratio_car))

        '''
        sum_vod 全人口に分担率をかけた値（自動車交通量）
        n_i メッシュ内全人口
        vod_in メッシュ内に止まる全人口（車、非車は関係なし）
        n_i_out 車でメッシュ外へ流出する人口
        n_out メッシュiからの流出人口
        '''

        # print(self.n_i[:20])

        vod_in = [0] * (self.N - 1)  # num. of vehicles that want to stay in a grid
        self.matrix.vod = [0] * self.L
        sum_vod = [0] * (self.N - 1)  # num. of vehicles that want to go outside of a grid
        for i in range(1, self.N):  # nodeid[N-1]は吸収ノードなので計算しなくていい
            a_list = self.matrix[self.matrix.k == self.nodeid[i - 1]].a
            a_list = sorted(a_list.values)
            a_list = a_list[0:len(a_list) - 1]  # to eliminate cases of k=i, a=absorbing state
            if len(a_list) > 0:
                for a in a_list:
                    ka = self.link[(self.link.O == self.nodeid[i - 1]) & (self.link.D == a)].linkID.values[0]
                    vod_ka = int(
                        self.p[t, i - 1, a - 1] * self.n_i[i - 1] * ratio_car + random.random())  # ransuu marume
                    sum_vod[i - 1] = sum_vod[i - 1] + vod_ka
                    if self.n_i[i - 1] - sum_vod[i - 1] >= 0:
                        self.matrix.iloc[ka - 1, 2] = vod_ka
                    else:
                        break
            vod_in[i - 1] = self.n_i[i - 1] - sum(self.matrix[self.matrix.k == self.nodeid[i - 1]].vod)
        self.n_i_out[:, t] = self.n_i - vod_in
        # print("vod_in")
        # print(vod_in)

        # delta = ratio of the num of veh. that want to go outside of a grid against the total veh. of the grid
        # now, we do not use this variable for calculation
        delta = [0] * (self.N - 1)
        for i in range(1, self.N):
            delta[i - 1] = (sum(self.matrix[self.matrix.k == self.nodeid[i - 1]].vod)) / (
                        (self.n_i[i - 1] == 0) + (self.n_i[i - 1] != 0) * self.n_i[i - 1])

        # q_d (outflow considering the effect of demand pattern differs from Kim et al.)
        # 　q_d: メッシュ内でのトリップ終了車両数
        q_d = [0.0] * (self.N - 1)
        for i in range(1, self.N):  # nodeid[N-1]は吸収ノードなので計算しなくていい
            # approximation of Greenshields' MFD
            #para_a = self.mfd_params.a[i - 1]
            #para_b = self.mfd_params.b[i - 1]
            para_a = self.mfd_params.a[0]
            para_b = self.mfd_params.b[0]
            n_out = sum(self.matrix[self.matrix.k == self.nodeid[
                i - 1]].vod)  # only the vehicles that go outside their mesh influence accumulation
            q_d[i - 1] = ((n_out <= (para_b / -para_a)) * (para_a * (n_out ** 2) + para_b * n_out) + (
                        n_out > (para_b / -para_a)) * 0)  # whether accumulation exceeds jam density
            q_d[i - 1] = q_d[i - 1] * 0.025  # 1.5/60. unit of q is scl(veh)/1.5min considering timestep
        # print("q_d")
        # print(q_d)

        # q_max (outflow considering the effect of boundary capacity
        # q_max: リンク容量とデマンドの内小さい方
        q_max = [0] * (self.N - 1)
        for i in range(1, self.N):
            a_list = self.matrix[self.matrix.k == self.nodeid[i - 1]].a
            a_list = sorted(a_list.values)
            a_list = a_list[0:len(a_list) - 1]  # to eliminate cases of k=i, a=absorbing state
            if len(a_list) > 0:
                for a in a_list:
                    ka: int = self.link[(self.link.O == self.nodeid[i - 1]) & (self.link.D == a)].linkID.values[0]
                    q_max[i - 1] = q_max[i - 1] + min(self.matrix.vod[ka - 1], self.limit[ka - 1])
            else:
                q_max[i - 1] = 0
        print(f"q_max: {q_max}, vod: {self.matrix.vod}, limit: {self.limit}")

        # q_demand(after considering the effect of demand pattern and boundary capacity)
        # q_demand: メッシュ内のトリップ終了車両数とリンク通過可能数の内小さい方
        q_demand = [0.0] * (self.N - 1)
        for i in range(1, self.N):
            if self.node.edge[i - 1] == 0:
                q_demand[i - 1] = min(q_max[i - 1], q_d[i - 1])
            else:
                q_demand[i - 1] = q_max[i - 1]  # ignore congestion inside the mesh in case of exogenous mesh

        # supply
        # supply: 受け入れメッシュの受け入れ可能車両台数
        supply = np.zeros(self.N - 1, dtype=int)
        for i in range(1, self.N):
            #para_a = self.mfd_params.a[i - 1]
            #para_b = self.mfd_params.b[i - 1]
            para_a = self.mfd_params.a[0]
            para_b = self.mfd_params.b[0]
            n_out = sum(self.matrix[self.matrix.k == self.nodeid[
                i - 1]].vod)  # only the vehicles that go outside their mesh influence accumulation
            supply[i - 1] = (n_out <= (para_b / (-2 * para_a))) * ((para_b ** 2) / (-4 * para_a)) + (
                        (para_b / (-2 * para_a)) < n_out <= (para_b / -para_a)) * (
                                        para_a * (n_out ** 2) + para_b * n_out) + (n_out > (para_b / -para_a)) * 0
            supply[i - 1] = supply[i - 1] * 0.025  # 1.5/60. unit of q is scl(veh)/1.5min considering timestep

        supply[np.where(self.node.edge == 1)] = 100000  # capacity of grids that is edge of NW is infinite

        # print("supply")
        # print(supply)

        # flow between cells
        self.matrix.flow = [0] * self.L
        sum_flow = [0] * (self.N - 1)
        for i in range(0, self.Lnum):
            i_d: int = self.matrix.k[i]
            j_s: int = self.matrix.a[i]
            ka: int = self.link[(self.link.O == i_d) & (self.link.D == j_s)].linkID.values[0]
            deno_d = sum(self.matrix[self.matrix.k == i_d].vod)
            delta_d = (self.matrix.vod[ka - 1]) / (
                        (deno_d == 0) + (deno_d != 0) * deno_d)  # メッシュi_dを出発する需要のうちリンクkaを通る者の割合
            deno_s = sum(self.matrix[self.matrix.a == j_s].vod)  # + vod_in[j_s - 1]
            delta_s = (self.matrix.vod[ka - 1]) / (
                        (deno_s == 0) + (deno_s != 0) * deno_s)  # メッシュj_sに到着する需要のうちリンクkaを通る者の割合
            # flow from cell i to cell j
            flow_ka = int(min(q_demand[i_d - 1] * delta_d,
                              supply[j_s - 1] * delta_s) + random.random())  # ransuu marume, リンクkaの需要と供給のうち小さい方
            sum_flow[i_d - 1] = sum_flow[i_d - 1] + flow_ka
            if self.n_i[i_d - 1] - sum_flow[i_d - 1] >= 0:
                self.matrix.iat[ka - 1, 3] = flow_ka
            else:
                self.matrix.loc[ka - 1, "flow"] = 0
            self.wait_timeline[t, ka - 1] = self.matrix.vod[ka - 1] - self.matrix.flow[ka - 1]
            self.link_flow[t, ka - 1] = self.matrix.flow[ka - 1]
            self.link_vod[t, ka - 1] = self.matrix.vod[ka - 1]

        # print(self.matrix.flow[:20])

        # updated population distribution
        for i in range(1, self.N):
            nid = self.nodeid[i - 1]
            self.n_i[i - 1] = self.n_i[i - 1] - sum(self.matrix[self.matrix.k == nid].flow) + sum(
                self.matrix[self.matrix.a == nid].flow)

        # print(self.n_i[:20])
        self.n_i_total[:, t + 1] = self.n_i

        self.t += 1
        return self.n_i, self.matrix

    def set_modified_matrix(self, new_matrix: pd.DataFrame) -> None:
        """
        外部から与えられた新しいマトリックスで人口分布・フローを更新する。

        Args:
            new_matrix (pd.DataFrame): 更新後のリンク情報データフレーム。
        """
        # flow between cells
        self.n_i = self.n_i_total[:, self.t - 1]
        for i in range(0, self.Lnum):
            # flow from cell i to cell j
            i_d = new_matrix.k[i]
            j_s = new_matrix.a[i]
            ka = self.link[(self.link.O == i_d) & (self.link.D == j_s)].linkID.values[0]
            self.wait_timeline[self.t - 1, ka - 1] = new_matrix.vod[ka - 1] - new_matrix.flow[ka - 1]
            self.link_flow[self.t - 1, ka - 1] = new_matrix.flow[ka - 1]
            self.link_vod[self.t - 1, ka - 1] = new_matrix.vod[ka - 1]

        # updated population distribution
        for i in range(1, self.N):
            nid = self.nodeid[i]
            self.n_i[i - 1] = self.n_i[i - 1] - sum(new_matrix[new_matrix.k == nid].flow) + sum(
                new_matrix[new_matrix.a == nid].flow)
        self.n_i_total[:, self.t] = self.n_i

    def is_finish(self) -> bool:
        """
        シミュレーションが終了したか判定する。

        Returns:
            bool: 終了していればTrue。
        """
        return self.t >= self.TS2 - 1

    def write_result(self, mesh_file: str, input_obj: Any) -> None:
        """
        シミュレーション結果をファイルに書き出す。

        Args:
            mesh_file (str): メッシュファイルのパス。
            input_obj (Any): 入力データオブジェクト。
        """
        mesh_gdf = gpd.read_file(mesh_file)
        mesh_gdf.set_index("nodeID", drop=False, inplace=True)

        n_i_total = pd.DataFrame(self.n_i_total, index=self.nodeid[0:-1], columns=list(range(self.TS2)))
        n_i_total.to_csv(self.output_dir + "NodePopulation.csv", index=False)

        geometry = np.expand_dims(mesh_gdf.loc[self.nodeid[0:-1], "geometry"].values, axis=1)
        n_i_total_gdf = gpd.GeoDataFrame(np.concatenate([self.n_i_total, geometry], axis=1),
                                         columns=[str(i) for i in range(self.TS2)] + ["geometry"])
        n_i_total_gdf.to_file(self.output_dir + "NodePopulation.geojson", driver="GeoJSON", index=False)

        timeline_vals = []
        for i in range(self.TS2):
            sim_time = self.start_time + self.timestep * i
            time = sim_time * 1000 + 1602082800000 + 9 * 3600 * 1000
            tmpval = np.concatenate(
                [np.full((len(self.nodeid) - 1, 1), time), np.full((len(self.nodeid) - 1, 1), sim_time),
                 np.expand_dims(self.nodeid[0:-1], axis=1), n_i_total_gdf.loc[:, [str(i), "geometry"]].values],
                axis=1).tolist()
            timeline_vals.extend(tmpval)
        timeline_gdf = gpd.GeoDataFrame(timeline_vals,
                                        columns=["Time", "SimulaitonTime", "NodeID", "Population", "geometry"])
        timeline_gdf.to_file(self.output_dir + "PopulationTimeline.geojson", driver="GeoJSON", index=False)

        # Hongo Linkに速度を割り付け
        sum_velocity = dict()  # key: node_id, value: sum of max velocity in each link
        for i in range(self.N - 1):
            s = 0
            if self.nodeid[i] not in input_obj.mesh_link:
                continue
            for li in input_obj.mesh_link[self.nodeid[i]]:
                s += input_obj.hongo_link.loc[li, 'Velocity']
            sum_velocity[self.nodeid[i]] = s

        vals = []
        for t in range(self.TS2 - 1):
            n_i_out = self.n_i_out[:, t]
            sim_time = self.start_time + t * 90
            uni_time = sim_time * 1000 + 1602082800000 + 9 * 3600 * 1000
            link_velocity: dict[int, float] = dict()  # key: link_id, value: velocity(km/h)
            node_density_dict: dict[int, float] = dict()  # key: node_id, value: density(veh/m)
            link_density_dict: dict[int, float] = dict()  # key: link_id, value: density
            for i in range(self.N - 1):
                para_a = self.mfd_params.a[0]
                para_b = self.mfd_params.b[0]
                n_out = n_i_out[i]
                supply = (n_out <= (para_b / -2 / para_a)) * ((para_b ** 2) / (-4 * para_a)) + (
                        (para_b / -2 / para_a) < n_out <= (para_b / -para_a)) * (
                                 para_a * (n_out ** 2) + para_b * n_out) + (n_out > (para_b / -para_a)) * 0
                # unit: veh/hour

                # 制限速度に対する実際の速度の割合
                node_density_dict[self.nodeid[i]] = 0.001
                if (self.nodeid[i] in input_obj.link_length) and (self.nodeid[i] in sum_velocity):
                    density = max(0.001, n_out / input_obj.link_length[self.nodeid[i]])  # veh/m
                    node_density_dict[self.nodeid[i]] = density
                    move_length = supply / density / 1000  # total travel length(km)
                    if (move_length > 1) or (n_out > - para_b / para_a):
                        rate = max(min(1, move_length / sum_velocity[self.nodeid[i]]), 0.01)
                    else:
                        rate = 1
                else:
                    rate = 1

                if self.nodeid[i] in input_obj.mesh_link:
                    for li in input_obj.mesh_link[self.nodeid[i]]:
                        if li in link_velocity:
                            link_velocity[li] += (input_obj.hongo_link.loc[li, "LeftCarNum"] > 0) * \
                                                 input_obj.hongo_link.loc[li, 'Velocity'] * rate / len(
                                input_obj.link_mesh[li])
                            link_density_dict[li] += node_density_dict[self.nodeid[i]] / len(input_obj.link_mesh[li])
                        else:
                            link_velocity[li] = (input_obj.hongo_link.loc[li, "RightCarNum"] > 0) * \
                                                input_obj.hongo_link.loc[li, 'Velocity'] * rate / len(
                                input_obj.link_mesh[li])
                            link_density_dict[li] = node_density_dict[self.nodeid[i]] / len(input_obj.link_mesh[li])
            # Time,SimulationTime,LinkID,CarTrafficVolume(OD),CarVelocity(km/h)(OD),CarVelocity(km/h)(DO),CarTrafficVolume(DO),CarVelocity(km/h),TrafficVolume,Velocity(km/h),ONodeID,DNodeID,OLon,OLat,DLon,DLat
            for li, v in link_velocity.items():
                # values in 1.5s
                density = 0.001
                if li in link_density_dict:
                    density = link_density_dict[li]
                traffic_volume = max(0, (v / 3.6 * 1.5 - input_obj.hongo_link.loc[li, 'length'])) * density
                o_node = input_obj.hongo_link.loc[li, 'ONodeID']
                d_node = input_obj.hongo_link.loc[li, 'DNodeID']
                o_coord = input_obj.hongo_node.loc[o_node, ['Lon', 'Lat']].values
                d_coord = input_obj.hongo_node.loc[d_node, ['Lon', 'Lat']].values
                vals.append([uni_time, sim_time, li, traffic_volume * input_obj.hongo_link.loc[li, 'LeftCarNum'],
                             traffic_volume * input_obj.hongo_link.loc[li, 'RightCarNum'], v, v, v,
                             traffic_volume * sum(input_obj.hongo_link.loc[li, ['LeftCarNum', 'RightCarNum']]), 0,
                             o_node, d_node, o_coord[0], o_coord[1], d_coord[0], d_coord[1]])
        link_output_df = pd.DataFrame(vals, columns=['Time', 'SimulationTime', 'LinkID', 'CarTrafficVolume(OD)',
                                                     'CarTrafficVolume(DO)', 'CarVelocity(km/h)(OD)',
                                                     'CarVelocity(km/h)(DO)', 'CarVelocity(km/h)', 'TrafficVolume',
                                                     'Velocity(km/h)', 'ONodeID', 'DNodeID', 'OLon', 'OLat', 'DLon',
                                                     'DLat'])
        link_output_df.to_csv(self.output_dir + "LinkOutput.csv", index=False)

    def _read_param_file(self, RLparameter_file1: str, RLparameter_file2: str, MFDparameter_file: str) -> None:
        """
        パラメータファイルを読み込み、モデルパラメータを初期化する。

        Args:
            RLparameter_file1 (str): 活動パラメータファイル。
            RLparameter_file2 (str): 経路パラメータファイル。
            MFDparameter_file (str): MFDパラメータファイル。
        """
        Activity_params = pd.read_csv(RLparameter_file1)
        Route_params = pd.read_csv(RLparameter_file2)
        MFDparams = pd.read_csv(MFDparameter_file)
        b1 = Activity_params.Toyosu
        b2 = Route_params.Toyosu

        """re-writing MFD parameters according to the result of regression.  
        This is for clear output of assignment, and essentially we have to omit this part with more accurate estimation of MFD parameters.

        p5 = -1.282074e-04
        p7 = 1.00241e-04
        p8 = 4.979904e-05
        MFDparams.b = 1/(p7*(MFDparams.d_link) + p8*(MFDparams.population))
        MFDparams.a = (p5)*(MFDparams.b)**2
        """

        self.x1 = b1
        self.x2 = b2
        self.mfd_params = MFDparams

    def _read_node_file(self, node_file: str) -> None:
        """
        ノード情報ファイルを読み込み、ノード関連変数を初期化する。

        Args:
            node_file (str): ノード情報ファイルのパス。
        """
        node = pd.read_csv(node_file)
        node = pd.concat([node[["nodeID", "KEY_CODE"]], node.iloc[:, 2:len(node.columns)]],
                         axis=1)  # change columns' order
        node.loc[len(node)] = [max(node.nodeID) + 1] + [0] * (len(node.columns) - 1)  # absorbing node
        node.sort_values(by='nodeID', ascending=True)

        self.node = node
        self.nodeid = sorted(node.nodeID.unique().tolist())
        self.N = len(self.nodeid)

        # demand data
        self.n_i = node.Population[:-1]  # shape: (N-1, )

    def _read_link_file(self, link_file: str) -> None:
        """
        リンク情報ファイルを読み込み、リンク関連変数を初期化する。

        Args:
            link_file (str): リンク情報ファイルのパス。
        """
        link = pd.read_csv(link_file)
        absLink = pd.DataFrame(np.zeros((self.N - 1) * len(link.columns)).reshape((self.N - 1), len(link.columns)),
                               columns=link.columns)  # absorbing link
        absLink.O = [i for i in range(1, self.N)]
        absLink.Olat = self.node.lat[:-1]
        absLink.Olon = self.node.lon[:-1]
        absLink.D = self.N
        absLink.Dlat = self.node.lat[self.N - 1]
        absLink.Dlon = self.node.lon[self.N - 1]
        link = pd.concat([link, absLink], axis=0).reset_index(drop=True)
        link.linkID = pd.Series(range(1, len(link) + 1), index=link.index)  # linkID starts from 1

        self.link = link
        self.limit = (link.limit) * 0.025  # link capacity in 2010, 0.025 = 1.5min/60min
        self.linkid = sorted(link.linkID.unique())
        self.L = len(self.linkid)  # the num. of links including absorbing links
        self.Lnum = self.L - len(absLink)  # the actual num. of links

        # matrix data (OD node and vehicle on demand(VOD) of each link)
        matrix = pd.DataFrame(np.zeros(self.L * 4).reshape(self.L, 4), columns=["k", "a", "vod", "flow"])
        matrix.k = link.O
        matrix.a = link.D

        self.matrix = matrix

        # adjacency matrix
        I2 = np.zeros((self.N, self.N), dtype=int)
        for i in range(0, self.L):
            I2[self.nodeid.index(matrix.k[i]), self.nodeid.index(matrix.a[i])] = 1  # 1 if node k and a are connected
        for i in range(0, self.N):
            I2[i, i] = 1  # for staying behavior
        I2[:, self.N - 1] = 0  # cannot go to absorbing node before TS

        self.I2 = I2

        # Since too many alternatives leads to small probablity for each alternative ,we sample destinaions w.r.t activity model
        # staying behavior and going back home is always included in the choice set.
        I1 = np.zeros((self.N, self.N, self.N), dtype=bool)
        for i in range(0, self.N - 1):
            for j in range(self.N - 1):
                idx = np.random.choice(self.N - 1, 2, replace=False)
                cset = [i] + [k if k<i else k+1 for k in idx]
                I1[i,j,cset] = True

        I1[:, self.N - 1, :] = False
        I1[:, :, self.N - 1] = False  # cannot go to absorbing node before TS
        self.I1 = I1

        # must go to absorbing node in TS
        ITS1 = np.zeros((self.N, self.N), dtype=int)
        ITS1[:, self.N - 1] = 1

        self.ITS1 = ITS1

    def _initialize_variables(self) -> None:
        """
        モデルで使用する各種変数を初期化する。
        """
        #####################################
        ####variables for activity choice of 1day#####
        #####################################
        # ---variables (if traveler chooses to move)---#
        # travel time matrix (N*N matrix).
        b1_Length = np.full((self.N, self.N), np.exp(0))
        for i in range(0, self.L):
            latlon_k = (self.node.lat[self.nodeid.index(self.matrix.k[i])],
                        self.node.lon[self.nodeid.index(self.matrix.k[i])])  # lat, lon of k
            latlon_a = (self.node.lat[self.nodeid.index(self.matrix.a[i])],
                        self.node.lon[self.nodeid.index(self.matrix.a[i])])  # lat, lon of a
            if self.matrix.k[i] == max(self.node.nodeID) or self.matrix.a[i] == max(self.node.nodeID):
                b1_Length[self.nodeid.index(self.matrix.k[i]), self.nodeid.index(self.matrix.a[i])] = np.exp(0)
            else:
                b1_Length[self.nodeid.index(self.matrix.k[i]), self.nodeid.index(self.matrix.a[i])] = np.exp(
                    cal_distance(latlon_k, latlon_a) / 10)
        b1_Length[:, self.N - 1] = np.exp(0)  # transition to abs. node
        # return-to-home dummy in the night matrix
        b1_Home_yoru = np.full((self.N, self.TS1, self.N, self.N), np.exp(0))
        # ---variables (if traveler chooses to stay)---#
        # population matrix
        b1_Population = np.full((self.N, self.N), np.exp(0))
        for i in range(0, self.N - 1):
            if self.node.Population[i] > 0:
                b1_Population[i, i] = np.exp(np.log(self.node.Population[i]) / 10)
            else:
                b1_Population[i, i] = np.exp(0)
        # stay-home dummy in the morning matrix
        b1_Home_asa = np.full((self.N, self.TS1, self.N, self.N), np.exp(0))
        b1_Stay = np.full((self.TS1, self.N, self.N), np.exp(0))
        for i in range(0, self.N - 1):
            b1_Stay[:, i, i] = np.exp(1)

        self.b1_length = b1_Length
        self.b1_population = b1_Population
        self.b1_home_yoru = b1_Home_yoru
        self.b1_home_asa = b1_Home_asa
        self.b1_stay = b1_Stay

        #############################
        ####variables for route choice#####
        #############################
        # ---variables (if traveler chooses to move)---#
        # travel time matrix (N*N matrix).
        b2_Length = np.full((self.N, self.N), np.exp(0))
        for i in range(0, self.L):
            b2_Length[self.nodeid.index(self.matrix.k[i]), self.nodeid.index(self.matrix.a[i])] = np.exp(0.5)
        b2_Length[:, self.N - 1] = np.exp(0)  # transition to abs. node
        # ---variables (if traveler chooses to stay)---#
        # population matrix
        b2_Population = np.full((self.N, self.N), np.exp(0))
        for i in range(0, self.N):
            if self.node.Population[i] > 0:
                b2_Population[i, i] = np.exp(np.log(self.node.Population[i]) / 10)
            else:
                b2_Population[i, i] = np.exp(0)
        # stay-home dummy in the morning matrix
        b2_Stay_o = np.full((self.N, self.TS2, self.N, self.N), np.exp(0))
        b2_Stay = np.full((self.N, self.N), np.exp(0))
        for i in range(0, self.N - 1):
            b2_Stay[i, i] = np.exp(1)

        self.b2_length = b2_Length
        self.b2_population = b2_Population
        self.b2_stay_o = b2_Stay_o
        self.b2_stay = b2_Stay

        n_i_total = np.zeros((self.N - 1, self.TS2), dtype="int")
        n_i_total[:, 0] = self.n_i  # initial population distribution
        n_i_out = np.zeros((self.N - 1, self.TS2 - 1), dtype="int")

        R = sum(self.n_i)  # total num. of demand(vehicles)
        wait_timeline = np.zeros((self.TS2 - 1, self.Lnum), dtype="int")  # num. of waiting for each link
        link_flow = np.zeros((self.TS2 - 1, self.Lnum), dtype="int")  # link flow at each timestep
        link_vod = np.zeros((self.TS2 - 1, self.Lnum), dtype="int")  # vehicle on demand

        self.p = np.zeros((self.TS2, self.N, self.N), dtype="float64")
        self.n_i_total = n_i_total
        self.n_i_out = n_i_out
        self.R = R
        self.wait_timeline = wait_timeline
        self.link_flow = link_flow
        self.link_vod = link_vod


def get_simulation(input_dir: str, output_dir: str, start_time: int, end_time: int, timestep: int) -> MFDRL:
    """
    シミュレーション用のMFDRLインスタンスを生成し、初期化する。

    Args:
        input_dir (str): 入力ディレクトリ。
        output_dir (str): 出力ディレクトリ。
        start_time (int): 開始時刻（秒）。
        end_time (int): 終了時刻（秒）。
        timestep (int): タイムステップ（秒）。

    Returns:
        MFDRL: 初期化済みのMFDRLインスタンス。
    """
    sim = MFDRL(input_dir, output_dir, start_time, end_time, timestep)
    sim.read_data()
    sim.set_prob()
    return sim


def main_mfdrl(sim: MFDRL) -> tuple[np.ndarray, pd.DataFrame] | tuple[None, None]:
    """
    1ステップ分の人口分布・リンクフローを返す。

    Args:
        sim (MFDRL): MFDRLインスタンス。

    Returns:
        tuple[np.ndarray, pd.DataFrame] | tuple[None, None]: 人口分布とリンク情報、または終了時は(None, None)。
    """
    if not sim.is_finish():
        n_i, matrix = sim.assign()
        return n_i, matrix
    else:
        return None, None


def get_input_obj(input_dir: str, epsg: int, bb_coord: list[tuple[float, float]]) -> Any:
    """
    入力オブジェクトを生成する。

    Args:
        input_dir (str): 入力ディレクトリ。
        epsg (int): EPSGコード。
        bb_coord (list[tuple[float, float]]): バウンディングボックス座標。

    Returns:
        Any: MFDRLInputインスタンス。
    """
    mfdrl_input = create_input_mfdrl.MFDRLInput(input_dir, epsg, bb_coord)
    return mfdrl_input


def create_input_initial(input_obj: Any, mesh_file: str) -> tuple[Any, str]:
    """
    メッシュファイルを作成し、入力オブジェクトとファイルパスを返す。

    Args:
        input_obj (Any): 入力オブジェクト。
        mesh_file (str): メッシュファイルパス。

    Returns:
        tuple[Any, str]: 入力オブジェクトと作成されたメッシュファイルパス。
    """
    geo_mesh_file = input_obj.create_mesh(mesh_file)
    return input_obj, geo_mesh_file


def create_input(input_obj: Any, tmp_input_dir: str, node_mesh_file: str, start_time: int, end_time: int) -> Any:
    """
    ネットワーク・パラメータファイルを作成し、入力オブジェクトを返す。

    Args:
        input_obj (Any): 入力オブジェクト。
        tmp_input_dir (str): 一時インプットディレクトリ。
        node_mesh_file (str): メッシュファイル。
        start_time (int): 開始時刻（秒）。
        end_time (int): 終了時刻（秒）。

    Returns:
        Any: 入力オブジェクト。
    """
    node_mesh_dict = pickle_load(node_mesh_file)
    input_obj.create_network(tmp_input_dir, node_mesh_dict, start_time, end_time)
    pd.read_csv(input_obj.input_dir + 'Activity_params.csv').to_csv(tmp_input_dir + 'Activity_params.csv')
    pd.read_csv(input_obj.input_dir + 'mfd_params3.csv').to_csv(tmp_input_dir + 'mfd_params3.csv')
    pd.read_csv(input_obj.input_dir + 'Route_params.csv').to_csv(tmp_input_dir + 'Route_params.csv')
    return input_obj

def copy_param_files(input_dir: str, output_dir: str) -> None:
    """
    パラメータファイルをコピーする。

    Args:
        input_dir (str): 入力ディレクトリ。
        output_dir (str): 出力ディレクトリ。
    """
    shutil.copy(input_dir + "Activity_params.csv", output_dir + "Activity_params.csv")
    shutil.copy(input_dir + "Route_params.csv", output_dir + "Route_params.csv")
    shutil.copy(input_dir + "mfd_params3.csv", output_dir + "mfd_params3.csv")


def pickle_dump(obj: Any, path: str) -> None:
    """
    オブジェクトをpickle形式で保存する。

    Args:
        obj (Any): 保存するオブジェクト。
        path (str): 保存先パス。
    """
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: str) -> Any:
    """
    pickleファイルからオブジェクトを読み込む。

    Args:
        path (str): 読み込みファイルパス。

    Returns:
        Any: 読み込まれたオブジェクト。
    """
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
        return obj


def cal_distance(latlon_a: tuple[float, float], latlon_b: tuple[float, float]) -> float:
    """
    2点間の緯度経度から距離（km）を計算する。

    Args:
        latlon_a (tuple[float, float]): 点Aの緯度・経度。
        latlon_b (tuple[float, float]): 点Bの緯度・経度。

    Returns:
        float: 2点間の距離（km）。
    """
    pole_radius = 6356752.314245  # 極半径
    equator_radius = 6378137.0  # 赤道半径

    # 緯度経度をラジアンに変換
    lat_a = math.radians(latlon_a[0])
    lon_a = math.radians(latlon_a[1])
    lat_b = math.radians(latlon_b[0])
    lon_b = math.radians(latlon_b[1])

    lat_difference = lat_a - lat_b  # 緯度差
    lon_difference = lon_a - lon_b  # 経度差
    lat_average = (lat_a + lat_b) / 2  # 平均緯度

    e2 = (math.pow(equator_radius, 2) - math.pow(pole_radius, 2)) \
         / math.pow(equator_radius, 2)  # 第一離心率^2

    w = math.sqrt(1 - e2 * math.pow(math.sin(lat_average), 2))

    m = equator_radius * (1 - e2) / math.pow(w, 3)  # 子午線曲率半径

    n = equator_radius / w  # 卯酉線曲半径

    distance = math.sqrt(math.pow(m * lat_difference, 2) \
                         + math.pow(n * lon_difference * math.cos(lat_average), 2))  # 距離計測
    distance = distance / 1000  # unit is km

    return distance


if __name__ == "__main__":
    sim = MFDRL("/Users/masudasatoki/Desktop/NEDO/simulation/MFD-RL/input/toyosu/359/",
                "/Users/masudasatoki/Desktop/NEDO/simulation/MFD-RL/output/toyosu/", 3600 * 10, 3600 * 11, 90)
    sim.read_data()
    sim.set_prob()



