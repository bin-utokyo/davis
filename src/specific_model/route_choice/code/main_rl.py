# -*- coding: utf-8 -*-
# 手順:最初にsetdrlで安定する説明変数の組み合わせやオーダーの見当をつけてからbetaも推定するndrlで推定
### 書き換えが必要な行 ###
# 25行目:method/28行目:年代/31行目:識別番号/39行目:集合の取り方/説明変数のところ/43行目:年齢

import time
import random
#import math
import numpy as np
import pandas as pd
import os
import sys
#from scipy.linalg import norm
from scipy.optimize import minimize
from scipy.sparse.csgraph import shortest_path
#from tqdm import tqdm, trange
from sklearn.utils import resample
#from multiprocessing import Pool
from joblib import Parallel, delayed

from setting import Network
from recursive import Recursive


def main(mode, input_dir, output_dir):
    """
    Main function to run the route choice estimation.

    Args:
        mode (str): The mode of transportation ('walk').
        input_dir (str): The directory containing input files.
        output_dir (str): The directory for output files.
    """

    time0 = time.time()
    print('Program Start!')

    # (1)推定方法
    method ='ndrl' #rl=RL,ndrl=時間割引率RL,setdrl=時間割引率RL(β固定),nldrl=一般化割引率RL
    lrdummy = 0 #0=右左折ダミー無し，1=有り

    print('【',method, input_dir,':','出力ディレクトリ',output_dir,'】')

    ### 時間割引率(β固定のsetdrlで使用)
    sbeta=0.8

    # データ取得
    pls3 = os.path.join(input_dir, 'transition.csv')
    df3 = pd.read_csv(pls3)

    network, f_name = make_network(input_dir)

    if mode == 'walk':
        ########## データの分割 ##########
        df_train, df_valid = split_data(df3, 1)
        del df3
        print('Data Split and Sample: Done')

        #パラメータ初期値
        # パラメータ初期値
        #自動(ゼロ行列)
        if lrdummy == 1:
            B = len(f_name) + 1
        else:
            B = len(f_name)
        if method == 'ndrl':
            par_init = np.zeros(B + 1)
        elif method == 'nldrl':
            par_init = np.zeros(B + 1)
        else :
            par_init = np.zeros(B )
        #手動
        # par_init = np.array([-1,0,0,0,0,0,0])
        # par_init[0]=-2
        # par_init[-1]=-1
        rl(output_dir,network, f_name,df_train, df_valid,method,lrdummy,par_init,sbeta)


def make_network(place):
    # データcsvの読み込み
    node_file, link_file = readfile(place)
    network = Network(node_file, link_file)

    # Dijkstra法で各ノードから各ノードへの最短経路距離を計算
    # adj = network.adj_matrix()
    # network.sp_mat = shortest_path(adj)

    # リンク属性の準備
    f_name = network.f_name  # リンク属性の名前

    print('Ready')
    del node_file, link_file
    return network, f_name


def rl(output_dir, network, f_name,df_train, df_valid,method,lrdummy,par_init,sbeta):

    ########## パラメータ推定 ##########
    t4 = time.time()
    print('Start Estimation...')


    #model = RecursiveSampling
    model = Recursive
    res, fr, B, N = estimate(model, df_train, network, f_name,method,lrdummy,par_init,sbeta)

    show_estimation_result(res, fr, B, N, f_name, output_dir, method,lrdummy,sbeta)
    #print('Time for Estimation: {} sec'.format(t_end - t_start))

    t5 = time.time()
    print('Time for Total Estimation: {} sec'.format(t5 - t4))

    ########### Hold-Out Validation ##########
    par_opt = res.x
    if len(df_valid)!=0:
        calc_validation(model, df_valid, network, f_name, par_opt, method,lrdummy,sbeta)

'''
def bagging(network, f_name, df3, method,svfld, times):

    df3 = data_cleaning(df3)
    trip_list = df3['TripID'].unique().tolist()
    #df_train, df_valid = None, None
    trip_list = resample(trip_list)
    df_train = pd.DataFrame([])
    for trip in trip_list:
        df_temp = df3[df3['TripID']==trip]
        df_train=pd.concat([df_train, df_temp])
    ########## パラメータ推定 ##########
    print('Start Estimation...')

    # model = RecursiveSampling
    model = Recursive

    res, fr, B, N = estimate(model, df_train, network, f_name, method)
    print("done")

    if method=='drl':
        par0 = np.zeros(B + 1)
        f_name2 = f_name + ['return', 'lrturn', 'nodedummy', 'discount']
    elif method=='ldrl':
        par0 = np.zeros(B + 2)
        f_name2 = f_name + ['return', 'lrturn', 'nodedummy', 'discount', 'discount2']
    par_opt = res.x
    h_inv = res.hess_inv
    tval = par_opt / np.sqrt(np.diag(h_inv))  # 各パラメータのt値

    L0 = -1 * fr(par0)  # 初期尤度，関数frが勝手にprintする
    LL = -1 * res.fun  # 最終尤度
    rho2_adj = (L0 - (LL - len(par_opt))) / L0  # 修正済み尤度比
    aic = -2 * (LL - len(par_opt))  # 赤池情報量規準AIC
    resultl = pd.DataFrame({'val': par_opt,'tval': tval},index=f_name2)
    #resultl=np.array([f_name2,par_opt,tval])
    #resultl=np.transpose(resultl)
    resultl2 =pd.DataFrame({'val': [L0,LL,rho2_adj,aic,N],'tval': [0,0,0,0,0]},index=['初期尤度','修正尤度','尤度比','AIC', 'N'])
    resultl=resultl.append(resultl2)
    #print(resultl)
    #print(times)
    svpls=svfld+'/'+str(times)+'.csv'
    np.savetxt(svpls,resultl,fmt="%s")
    return resultl, times
'''

def calc_validation(model, df_valid, network, f_name, par_opt,method,lrdummy,sbeta):
    t6 = time.time()
    err = validation(model, df_valid, network, f_name, par_opt,method,lrdummy,sbeta)
    print('---------- Validation Results ----------')
    print('Test Error Value = {}'.format(err))
    t7 = time.time()
    print('Time for Total Validation: {} sec'.format(t7 - t6))



def readfile(place):
    pls1 = os.path.join(place, 'node.csv')
    pls2 = os.path.join(place, 'link.csv')
    df1 = pd.read_csv(pls1)
    df2 = pd.read_csv(pls2)

    return df1, df2


def data_cleaning(df):
    # 経由リンク数が1しかない経路(トリップ)を除外
    dfs = df.groupby('TripID').count().reset_index()[['TripID', 'LinkID']]
    lst = dfs[dfs['LinkID'] > 1]['TripID'].values.tolist()
    #trip_list = df['TripID'].unique().tolist()
    df_sampled = df[df['TripID'].isin(lst)]
    return df_sampled


def split_data(df, ratio):
    ## データを学習用(training set)と検証用(validation set)に分割
    df = data_cleaning(df)
    trip_list = df['TripID'].unique().tolist()
    #df_train, df_valid = None, None
    num = len(trip_list)
    list_train = random.sample(trip_list, int(num * ratio))
    df_train = df[df['TripID'].isin(list_train)]
    df_valid = df[~df['TripID'].isin(list_train)]


    return df_train, df_valid


def estimate(model, df, network, f_name,method,lrdummy,par_init,sbeta):
    #t_start = time.time()
    N = len(df)
    if method == 'ndrl' or method == 'nldrl'or method == 'setdrl' or method == 'rl':
        A = network.n_link
        B = len(f_name)
    else:
        A = network.n_link+network.n_node
        B = len(f_name) + 1
    if lrdummy == 1:
        B = B + 1

    d_list, data_list = model.set_data(network ,df, A)  # データを終点dごとにまとめ，行列化
    print('No. of Link Transition: {}'.format(N))
    print('No. of Trips: {}'.format(len(df.groupby('TripID'))))
    #print('No. of O-D pairs: {}'.format(len(df.groupby(['OriNode', 'DesNode']))))
    print('No. of Destination: {}'.format(len(d_list)))

    #length=getattr(network, 'Length')

    m = model(N, A, B, d_list, data_list,method,lrdummy,df,network,sbeta)
    #print('Model: {}'.format(m.name))
    link_adj = network.adj_mat(method)

    m.set_matrices(network, link_adj, f_name)  # 選択結果，説明変数等を行列(np.array)に格納

    # パラメータ初期値

    fr = m.log_likelihood  # 目的関数：負の対数尤度関数

    res = minimize(fr, par_init, method='BFGS') #さいぱい cypy
    # mini = args が出てくる


    #t_end = time.time()
    if np.isnan(res.fun):
        res = minimize(fr, par_init, method='BFGS')


    return res, fr, B, N


def validation(model, df, network, f_name, par,method,lrdummy,sbeta):
    N = len(df)
    #A = network.n_link+network.n_node
    #B = len(f_name)+3
    if lrdummy ==0:
        B = len(f_name)
    else :
        B = len(f_name)+1

    if method == 'ndrl' or method == 'nldrl'or method == 'setdrl' or method == 'rl':
        A = network.n_link
    else:
        A = network.n_link+network.n_node
        B = B+1

    d_list, data_list = model.set_data(network,df, A)  # データを終点dごとにまとめ，行列化
    print('No. of Link Transition: {}'.format(N))
    print('No. of Trips: {}'.format(len(df.groupby('TripID'))))
    #print('No. of O-D pairs: {}'.format(len(df.groupby(['OriNode', 'DesNode']))))
    print('No. of Destination: {}'.format(len(d_list)))
    #length = getattr(network, 'Length')
    m = model(N, A, B, d_list, data_list,method,lrdummy,df,network,sbeta)
    # print('Model: {}'.format(m.name))
    link_adj = network.adj_mat(method)
    m.set_matrices(network, link_adj, f_name)  # 選択結果，説明変数等を行列(np.array)に格納
    #m.logtest(par)

    loss_likelihood = m.log_likelihood(par) / N
    return loss_likelihood


def show_estimation_result(result, fr, B, N, f_name, output_dir, method,lrdummy,sbeta):
    if lrdummy==0:
        f_name2 = f_name
    elif lrdummy ==1:
        f_name2 = f_name+['lrturn']
    else:
        f_name2 = f_name

    if method=='nldrl':
        par0 = np.zeros(B + 1)
        f_name2 = f_name2 + ['discount']
    elif method=='ndrl':
        par0 = np.zeros(B + 1)
        f_name2 = f_name2 + ['discount']
    else:
        par0 = np.zeros(B)
        f_name2 = f_name2

    par_opt = result.x  # パラメータの推定値

    h_inv = result.hess_inv
    tval = par_opt / np.sqrt(np.diag(h_inv))  # 各パラメータのt値

    L0 = -1 * fr(par0)  # 初期尤度，関数frが勝手にprintする
    LL = -1 * result.fun  # 最終尤度
    rho2 = (L0 - LL) / L0  # 尤度比
    rho2_adj = (L0 - (LL - len(par_opt))) / L0  # 修正済み尤度比
    aic = -2 * (LL - len(par_opt))  # 赤池情報量規準AIC
    if method=='nldrl' or method=='ndrl':
        beta = 1 / (1 + np.exp(par_opt[-1]))
    elif method=='setdrl':
        beta=sbeta
    else:
        beta=1

    '''
    # これやるとバグるのでやめた方がいい.違うとわかってればいい.
    if method != 'setdrl':
        beta = 1 / (1 + np.exp(par_opt[-1])) # γが0から有意にずれいてるかがt値(betaだと0.5)
    else:
        beta = sbeta
    '''

    print('---------- Estimation Results ----------')
    print('sample number = {}'.format(N))
    print('    variables = {}'.format(f_name2))
    print('    parameter = {}'.format(par_opt))
    print('      t value = {}'.format(tval))
    print('           L0 = {}'.format(L0))
    print('           LL = {}'.format(LL))
    print('         rho2 = {}'.format(rho2))
    print('adjusted rho2 = {}'.format(rho2_adj))
    print('          AIC = {}'.format(aic))
    print('         beta = {}'.format(beta))

    make_csv(par_opt, tval,rho2_adj, f_name2, output_dir, method)

def make_csv(par_opt, tval,rho2_adj, f_name, output_dir, method):
    """
    Save the estimation results to a CSV file. Formmat:
    - Feature names
    - Estimated parameters
    - t-values of the parameters
    - Adjusted rho-squared

    Args:
        par_opt (np.ndarray): Estimated parameters.
        tval (np.ndarray): t-values of the parameters.
        rho2_adj (float): Adjusted rho-squared.
        f_name (list): List of feature names.
        output_dir (str): Directory to save the output CSV file.
        method (str): Estimation method used.
    """
    resultl=np.array([f_name,par_opt,tval])
    resultl=np.transpose(resultl)
    resultl=np.vstack((resultl,['修正尤度',rho2_adj,0]))
    svpls=os.path.join(output_dir, "{}.csv".format(method))
    np.savetxt(svpls,resultl,fmt="%s", delimiter=',')


## 実質これ以下だけで命令している.他はひたすら関数の定義
if __name__ == '__main__':
    # コマンドライン引数の取得
    args = sys.argv
    estimate_mode = bool(args[1])
    place = args[2] 
    nmm = args[3]  #識別番号

    if estimate_mode:
        main('walk', place, nmm)  #mode == 'walk'だとして推定してください
    
