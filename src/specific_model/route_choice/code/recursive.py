# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, csr_array
#import scipy.sparse as sp
#from scipy.sparse import linalg as spl
#from scipy.special import logsumexp
from scipy import integrate


class Recursive:
    ### Fosgerau et al. (2013)
    ## 割引率beta=1
    #def __init__(self, N, A, B, d_list, data_list,length,method,df,network,sbeta):
    def __init__(self, N: int, A: int, B: int, d_list: list[int], data_list: list, method: str, lrdummy: int, df, network, sbeta):
        self.N = N  # サンプル数＝リンク遷移回数
        self.A = A  # 全リンク数
        self.B = B  # 説明変数パラメータ数
        self.d_list = d_list  # 吸収状態のリスト
        self.data_list = data_list  # リンクの選択結果(k,a)の疎行列をdごとに格納するリスト
        self.D = len(self.d_list)  # 吸収状態の数
        self.A_tilde = self.A + self.D
        self.delta = lil_matrix((self.A, self.A_tilde))  # リンク隣接行列
        self.X = np.zeros(((self.B+1), self.A, self.A_tilde))  # リンク説明変数 #Bの+1はリターンダミー-10(固定値)
        self.BB = lil_matrix((self.A + 1, self.D))  # 吸収状態行列
        self.beta=0.5
        self.Z = np.ones((A+1,self.D))
        self.meth = method
        self.lrdummy = lrdummy
        self.df=df
        self.lam = np.ones((self.A,4))  # リンク説明変数_lambda
        self.network=network
        self.flow=np.zeros((A+1,1))
        self.sbeta=sbeta

    def log_likelihood(self, param: np.ndarray) -> float:
        ## 対数尤度関数
        theta = param[:]
        #beta = 1.0  # 将来効用の割引率
        #print('theta:',theta)

        M = lil_matrix((self.A + 1, self.A + 1))
        M[:-1, :-1] = self.util_matrix(theta)[:, :self.A]  # 遷移効用行列: shape=(A+1, A+1)
        M = M.tocsc()
        lamb = np.ones((self.A+1,1))

        #方法
        method=self.meth

        # ldrlの場合beta1が滞在割引(theta[-2])、beta2がリンク割引(theta[-1])

        # 逆行列価値関数
        if method == 'drl':
            beta = 1 / (1 + np.exp(theta[-1]))
        elif method == 'ndrl':
            beta = 1 / (1 + np.exp(theta[-1]))
        elif method == 'setdrl':
            beta = self.sbeta
        elif method == 'rl':
            beta = 1
        else:
            beta = 1

        self.beta=beta

        Z = self.value_update(M, beta,theta,method,lamb)
        #Z = np.dot(spl.inv(I - M).toarray(), self.BB.toarray())
        deno = np.dot(M.toarray(), Z).transpose()  # 分母: shape=(D, A+1)
        deno = np.where(deno == 0, 1, deno)  # zero除算回避


        # リンク選択確率: shape=(D, A+1, A+1)

        #P = (M.reshape(*M.shape, 1) * Z).transpose(2, 0, 1) / deno.reshape(*deno.shape, 1)
        #P = (M.toarray().reshape(*M.shape, 1) * Z).transpose(2, 0, 1) / deno.reshape(*deno.shape, 1)
        #P = np.where(P == 0, 1, P)  # log-zeroによる発散回避
        #lnP = np.log(P)
        ll=0
        if method == 'drl' or method == 'ldrl'or method == 'ndrl' or method == 'nldrl'or method == 'setdrl' or method == 'rl':
            for di in range(self.D):
                P = np.multiply(M[self.data_list[di].row, self.data_list[di].col], Z[self.data_list[di].col, di])
                #print('P/deno',P)
                P = P / deno[di, self.data_list[di].row]
                #print('P',P)
                P = np.where(P == 0, 1, P)
                lnP = np.log(P)
                ll += np.sum(self.data_list[di].data * lnP)

        print("Temp LL:", ll)

        if np.isnan(ll):
            print(np.any(np.isnan(M.data)))
            print(np.any(np.isnan(deno.data)))
            print(np.any(np.isnan(Z.data)))
            print(np.any(np.isnan(theta.data)))


        ## 戻り値は負の対数尤度
        return -1 * ll

    def logtest(self, param: np.ndarray) -> None:
        ## 対数尤度関数
        theta = param[:]
        #beta = 1.0  # 将来効用の割引率

        M = lil_matrix((self.A + 1, self.A + 1))
        M[:-1, :-1] = self.util_matrix(theta)[:, :self.A]  # 遷移効用行列: shape=(A+1, A+1)
        M = M.tocsc()

    def value_update(self, M: lil_matrix, beta, theta: np.ndarray, method: str, lamb: np.ndarray) -> np.ndarray:
        ## 価値反復による価値関数の更新
        #M = self.M
        val = np.zeros((self.A + 1,self.D))
        dl_z = np.ones(self.D) * 100
        tms=1
        if method =='drl' or method == 'ldrl' or method =='ndrl' or method == 'nldrl'or method =='rl' or method == 'setdrl':
            while any(dl_z > 0.1) or tms <= (np.sqrt(self.A_tilde) * 2):
                val_pre = val
                val_new = M @ (val ** beta) + self.BB.toarray()
                val_new = val_new * (np.ones((self.A + 1, self.D)) - self.BB.toarray()) + self.BB.toarray()
                val = val_new
                dl_z = np.linalg.norm(val_new - val_pre, axis=0)
                tms += 1

            self.Z = val
            return val ** beta
        else:
            return val



    def util_matrix(self, thet: np.ndarray, mu: float = 1) -> csr_matrix:
        ## 即時遷移効用行列Mを計算
        #util = np.tensordot(theta[:-1].reshape(1, 1, *theta[:-1].shape), self.X, 1).reshape(self.A, self.A_tilde)-theta[-1]
        #util = np.tensordot(theta[:-2].reshape(1, 1, *theta[:-2].shape), self.X, 1).reshape(self.A, self.A_tilde) - theta[-2]
        theta=thet
        method = self.meth
        if method == 'drl' or method == 'ndrl' or method == 'nldrl':
            theta=thet[:-1]
        elif method == 'ldrl':
            theta=thet[:-2]
        elif method == 'setdrl' or method == 'rl':
            theta=thet[:]
        theta = np.append(theta, -10)
        util = np.tensordot(theta[:].reshape(1, 1, *theta[:].shape), self.X, 1).reshape(self.A, self.A_tilde)
        '''
        if method == 'drl' or method == 'ndrl' or method == 'nldrl':
            util = np.tensordot(theta[:-1].reshape(1, 1, *theta[:-1].shape), self.X, 1).reshape(self.A, self.A_tilde)
        elif method == 'ldrl':
            util = np.tensordot(theta[:-2].reshape(1, 1, *theta[:-2].shape), self.X, 1).reshape(self.A, self.A_tilde)
        elif method == 'setdrl' or method == 'rl':
            util = np.tensordot(theta[:].reshape(1, 1, *theta[:].shape), self.X, 1).reshape(self.A, self.A_tilde)
        '''
        util[util>100] =100
        #print(np.any(np.isnan(self.X.data)))
        #print(np.any(np.isnan(util.data)))
        M = self.delta.toarray() * np.exp(mu * util)
        #print(np.any(np.isnan(M.data)))
        #print(util[np.isnan(M.data)])
        #self.M = M
        return csr_matrix(M)

    def set_matrices(self, network, link_adj: np.ndarray, f_name: list[str]) -> None:
        #print(self.A)
        self.delta[:self.A, :self.A] = link_adj  # 隣接行列

        for di, d in enumerate(self.d_list):
            end_node = d
            if self.meth !='ndrl' and self.meth != 'nldrl'and self.meth !='setdrl' and self.meth != 'rl':
                prevs = [l + network.n_link for l, v in enumerate(network.node_list) if v == end_node]
                for a in prevs:
                    self.delta[a , self.A + di] = 1
            prevs = [l for l, v in enumerate(network.link_end) if v == end_node]
            for a in prevs:
                self.delta[a , self.A + di] = 1

        self.BB[-1, :] = 1
        self.BB[:-1, :] = self.delta[:, self.A:]
        self.BB = self.BB.tocsc()

        for k, name in enumerate(f_name):
            x = network.attr[name]
            self.X[k, :self.A, :network.n_link] = np.tile(x, (self.A, 1))
            #if k==1 or k==2:
                #print(network.n_link)
                #print(self.lam.shape)
                #self.lam[:network.n_link,k]= x

        #リターンダミー、右左折ダミー、停止ダミー

        nins=len(f_name)
        if self.lrdummy ==1:
            for l in range(network.n_link):
                kakudo = network.link_kakudo[l]
                for k in range(network.n_link):
                    tkakudo = np.abs(network.link_kakudo[k] - kakudo)
                    if 60 < tkakudo < 179 or 181 < tkakudo < 300:
                        self.X[nins, l, k] = 1
            nins = nins + 1
        for l in range(network.n_link):
            stins=network.link_start[l]
            edins = network.link_end[l]
            for k in range(network.n_link):
                if network.link_start[k]==edins and network.link_end[k]==stins :
                    self.X[nins,l,k]=1

        '''

        if self.meth =='ndrl' or self.meth =='nldrl'or self.meth =='rl' or self.meth =='setdrl':
            nins = nins + 1
        else:
            nins=nins+1
            self.X[nins,:network.n_link,network.n_link:] =10
        self.lam[:network.n_link,3]=0
        '''


    def haibun(self, odmat: np.ndarray, par: np.ndarray) -> np.ndarray:
        theta = par[:]
        #beta = 1.0  # 将来効用の割引率

        M = lil_matrix((self.A + 1, self.A + 1))
        M[:-1, :-1] = self.util_matrix(theta)[:, :self.A]  # 遷移効用行列: shape=(A+1, A+1)
        M = M.tocsc()
        print(np.any(np.isnan(M.data)))
        #I = sp.identity(self.A + 1).tocsc()
        lamb = np.ones((self.A+1,1))

        #方法
        method=self.meth

        # beta1が滞在割引(theta[-2])、beta2がリンク割引(theta[-1])

        # 逆行列価値関数
        if method == 'drl':
            beta = 1 / (1 + np.exp(theta[-1]))
        else:
            beta = 1

        self.beta = beta

        Z = self.value_update(M, beta,theta,method,lamb)
        #Z = np.dot(spl.inv(I - M).toarray(), self.BB.toarray())
        deno = np.dot(M.toarray(), Z)  # 分母: shape=(D, A+1)
        deno = np.where(deno == 0, 1, deno)  # zero除算回避


        # リンク選択確率: shape=(D, A+1, A+1)

        #P = (M.reshape(*M.shape, 1) * Z).transpose(2, 0, 1) / deno.reshape(*deno.shape, 1)
        #P = (M.toarray().reshape(*M.shape, 1) * Z).transpose(2, 0, 1) / deno.reshape(*deno.shape, 1)
        #P = np.where(P == 0, 1, P)  # log-zeroによる発散回避
        #lnP = np.log(P)
        ll=0
        if method == 'drl' or method == 'ldrl':
            for di in range(self.D):
                P = M* (np.tile(Z[:, di].transpose(),(self.A+1,1)))
                #print(P.shape)
                P = P / np.tile(deno[:,di].reshape(self.A+1,1),(1,self.A+1))
                #print(P.shape)
                #print(odmat.shape)
                self.flow[:,0]+=np.dot(np.linalg.inv(np.eye(self.A+1)-P.T),(odmat[:,di].reshape(self.A+1,1)))
        return self.flow


    @classmethod
    def set_data(cls, network, df, A: int) -> tuple[list[int], np.ndarray]:
        ## 各吸収状態ごとに，リンク遷移結果をまとめる
        d_list = sorted(list(set(df['DestinationNodeID'])))
        D = len(d_list)
        group_df = df.groupby('DestinationNodeID')
        k_list = group_df['LinkID'].apply(list).to_list()
        a_list = group_df['NextLinkID'].apply(list).to_list()

        vals = [[1] * len(k_lst) for k_lst in k_list]
        k_list = [[network.link_list.index(k) if k in network.link_list else network.node_list.index(k)+network.n_link for k in k_lst] for k_lst in k_list]
        a_list = [[A if a==df['DestinationNodeID'].values.tolist()[a_num] else network.link_list.index(a) if a in network.link_list else network.node_list.index(a)+network.n_link for a in a_lst] for a_num, a_lst in enumerate(a_list)]

        data_list = np.array([coo_matrix((val, (k_lst, a_lst)), shape=(A + 1, A + 1))
                              for val, k_lst, a_lst in zip(vals, k_list, a_list)])
        return d_list, data_list

