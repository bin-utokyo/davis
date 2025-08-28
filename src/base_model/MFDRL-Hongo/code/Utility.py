#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 01:47:01 2021

@author: dogawa
"""
from typing import Any, Optional, Union
import numpy as np

def calcAngle(ox: float, oy: float, dx: float, dy: float) -> int:
    """
    x軸から見たベクトル(dx-ox,dy-oy)の角度を計算します。

    Args:
        ox (float): 原点x座標。
        oy (float): 原点y座標。
        dx (float): 終点x座標。
        dy (float): 終点y座標。

    Returns:
        int: x軸から見た角度（度単位）。
    """
    n = (dx - ox, dy - oy)
    if n[0] == 0 and n[1] == 0:
        return 0
    if n[0] > 0 and n[1] >= 0:
        return int(np.arctan(n[1] / n[0]) * 180 / np.pi)
    elif n[0] < 0 and n[1] > 0:
        return 180 - int(np.arctan(-n[1] / n[0]) * 180 / np.pi)
    elif n[0] < 0 and n[1] <= 0:
        return 180 + int(np.arctan(n[1] / n[0]) * 180 / np.pi)
    elif n[0] > 0 and n[1] < 0:
        return 360 - int(np.arctan(-n[1] / n[0]) * 180 / np.pi)
    elif n[1] > 0:
        return 90
    else:
        return 270
    
def calcAngle0to180(upAngle: float, dnAngle: float) -> float:
    """
    upAngleとdnAngleの成す角（0度以上180度以下）を計算します。

    Args:
        upAngle (float): 上側の角度。
        dnAngle (float): 下側の角度。

    Returns:
        float: 0度以上180度以下の角度。
    """
    tmpAngle = (dnAngle - upAngle) % 360
    if tmpAngle > 180:
        tmpAngle = 360 - tmpAngle
    return tmpAngle

def calcAngleMinus180to180(upAngle: float, dnAngle: float) -> float:
    """
    upAngleから見たdnAngleの角度（-180度より大きく180度以下）を計算します。

    Args:
        upAngle (float): 基準となる角度。
        dnAngle (float): 比較する角度。

    Returns:
        float: -180度より大きく180度以下の角度。
    """
    tmpAngle = (dnAngle - upAngle) % 360
    if tmpAngle > 180:
        tmpAngle = tmpAngle - 360
    return tmpAngle


class PointMgr:
    """近傍探索用の点管理クラス
    """
    def __init__(self, distFun: Optional[Any] = None) -> None:
        """
        近傍探索用の点管理クラスの初期化。

        Args:
            distFun (Optional[Any], optional): 距離関数。デフォルトはNone。
        """
        self.pointList: list[list[float]] = list()  # [[x, y]]
        self.sortx: list[int] = list()  # x座標でソートしたポイントID
        self.pointCount: int = 0

        self.thresh: float = 0.0000001  # 丸誤差対応のための閾値

        self.distFun: Optional[Any] = distFun
    
    def clear(self) -> None:
        """
        全ての点情報をクリアし、内部状態をリセットします。

        Returns:
            None
        """
        self.pointList.clear()
        self.sortx.clear()
        self.pointCount = 0
        
    def distance(self, p1: int, p2: int) -> float:
        """
        2点間の距離を計算します。

        Args:
            p1 (int): 1つ目の点のインデックス。
            p2 (int): 2つ目の点のインデックス。

        Returns:
            float: 2点間の距離。インデックスが範囲外の場合は-1。
        """
        if self.distFun is not None:
            return self.distFun(self.pointList[p1], self.pointList[p2])
        if (p1 < len(self.pointList)) and (p2 < len(self.pointList)):
            return ((self.pointList[p1][0] - self.pointList[p2][0])**2 + (self.pointList[p1][1] - self.pointList[p2][1])**2)**0.5 
        else:
            return -1
        
    def linkDistance(self, p: int, p1: int, p2: int) -> float:
        """
        点pと線分(p1,p2)の距離を計算します。

        Args:
            p (int): 対象点のインデックス。
            p1 (int): 線分の始点インデックス。
            p2 (int): 線分の終点インデックス。

        Returns:
            float: 点pと線分(p1,p2)の距離。
        """
        if (p == p1) or (p == p2):
            return 0
        a1 = self.calcAngle0toPi(p1, p1, p, p2)
        a2 = self.calcAngle0toPi(p2, p2, p, p1)
        
        if (a1 <= np.pi/2) and (a2 <= np.pi/2):#pからリンク(p1,p2)へおろした垂線がリンク内で交わる
            return self.distance(p, p1) * np.sin(a1)
        elif a1 > np.pi/2:#p1側にはみ出る時
            return self.distance(p, p1)
        else:#p2側にはみ出る時
            return self.distance(p, p2)
    
    def pointLocInLink(self, p: int, p1: int, p2: int) -> tuple[float, float]:
        """
        点pが線分(p1,p2)上のどの位置にあるかを返します。

        Args:
            p (int): 対象点のインデックス。
            p1 (int): 線分の始点インデックス。
            p2 (int): 線分の終点インデックス。

        Returns:
            tuple[float, float]: 線分上の位置（始点からの距離, 終点からの距離）。
        """
        if p == p1:
            return (0, self.distance(p, p2))
        if p == p2:
            return (self.distance(p, p1), 0)
        a1 = self.calcAngle0toPi(p1, p1, p, p2)
        a2 = self.calcAngle0toPi(p2, p2, p, p1)
        if (a1 <= np.pi/2) and (a2 <= np.pi/2):
            return (self.distance(p, p1) * np.cos(a1), self.distance(p, p2) * np.cos(a2))
        elif a1 > np.pi/2:
            return (0, self.distance(p1, p2))
        else:
            return (self.distance(p1, p2), 0)
        
    def calcAngle(self, p11: int, p12: int, p21: int, p22: int) -> float:
        """
        ベクトルp11→p21から見たベクトルp12→p22の角度を計算します。

        Args:
            p11 (int): ベクトル1の始点。
            p12 (int): ベクトル2の始点。
            p21 (int): ベクトル1の終点。
            p22 (int): ベクトル2の終点。

        Returns:
            float: 角度（ラジアン）。
        """
        t1 = self.calcAngleOfLine(p11, p21)
        t2 = self.calcAngleOfLine(p12, p22)
        return (t2 - t1) % (2*np.pi)
    
    def calcAngle0toPi(self, p11: int, p12: int, p21: int, p22: int) -> float:
        """
        ベクトルp11→p21から見たベクトルp12→p22の角度（0以上Pi以下）を計算します。

        Args:
            p11 (int): ベクトル1の始点。
            p12 (int): ベクトル2の始点。
            p21 (int): ベクトル1の終点。
            p22 (int): ベクトル2の終点。

        Returns:
            float: 角度（ラジアン、0以上Pi以下）。
        """
        tmpAngle = self.calcAngle(p11, p12, p21, p22)
        if tmpAngle > np.pi:
            tmpAngle = 2*np.pi - tmpAngle
        return tmpAngle
    
    def calcAngleOfLine(self, p1: int, p2: int) -> float:
        """
        ベクトルp1→p2の角度を計算します。

        Args:
            p1 (int): 始点インデックス。
            p2 (int): 終点インデックス。

        Returns:
            float: 角度（ラジアン）。
        """
        delx = self.pointList[p2][0] - self.pointList[p1][0]
        dely = self.pointList[p2][1] - self.pointList[p1][1]
        
        if (delx == 0) and (dely == 0):
            return 0
        elif (delx > 0) and (dely >= 0):
            return np.arctan(dely/delx)
        elif (delx < 0) and (dely > 0):
            return np.pi - np.arctan(-dely/delx)
        elif (delx < 0) and (dely <= 0):
            return np.pi + np.arctan(dely/delx)
        elif (delx > 0) and (dely < 0):
            return 2 * np.pi - np.arctan(-dely/delx)
        elif (dely > 0):
            return np.pi / 2
        else:
            return np.pi * 3 / 2
        
    def addPoint(self, x: float, y: float) -> int:
        """
        点（x, y）を追加します。

        Args:
            x (float): x座標。
            y (float): y座標。

        Returns:
            int: 追加した点のID。
        """
        if self.pointCount == 0:
            self.pointCount += 1
            self.pointList.append([x, y])
            self.sortx.append(self.pointCount - 1)
            return self.pointCount - 1
            
        u = self.pointCount - 1
        le = 0
        preu = -1
        prel = -1
        
        while (preu != u) or (prel != le):##self.sortxを使ってx座標の近い点を二分探索で探す
            preu = u
            prel = le
            
            tmp = (u + le) // 2
            tmpx = self.pointList[self.sortx[tmp]][0]
            if tmpx <= x-self.thresh:
                le = tmp
                continue
            elif tmpx >= x+self.thresh:
                u = tmp
                continue
            if self.pointList[self.sortx[le]][0] <= x-self.thresh:
                for i in range(tmp, le, -1):
                    if self.pointList[self.sortx[i]][0] <= x-self.thresh:
                        le = i
                        break
            if self.pointList[self.sortx[u]][0] >= x+self.thresh:
                for i in range(tmp,u):
                    if self.pointList[self.sortx[i]][0] >= x+self.thresh:
                        u = i
                        break
        ind = -1
        for i in range(le, u+1):
            if (self.pointList[self.sortx[i]][0] > x-self.thresh) and (self.pointList[self.sortx[i]][0] < x+self.thresh) and (self.pointList[self.sortx[i]][1] > y-self.thresh) and (self.pointList[self.sortx[i]][1] < y+self.thresh):
                return self.sortx[i]#同一の点がすでに追加されていた場合
            if (ind == -1) and (self.pointList[self.sortx[i]][0] > x):
                ind = i
                
        self.pointCount += 1
        self.pointList.append([x,y])
        if ind == -1:
            if u+1 == self.pointCount - 1:
                self.sortx.append(self.pointCount - 1)
            else:
                self.sortx.insert(u+1, self.pointCount - 1)
        else:
            self.sortx.insert(ind, self.pointCount - 1)
        return self.pointCount - 1

class LineMgr:
    def __init__(self) -> None:
        """
        線分管理クラスの初期化。
        """
        self.lineList: list[list[Union[int, float]]] = list()#[[p1, p2 ,fin1, fin2, angle]]
        self.lineCount = 0
        
        self.pointMgr = PointMgr()
        
    def clear(self) -> None:
        """
        全ての線分・点情報をクリアします。

        Returns:
            None
        """
        self.lineList.clear()
        self.lineCount = 0
        self.pointMgr.clear()
        
    def addPoint(self, x: float, y: float) -> int:
        """
        点（x, y）を追加します。

        Args:
            x (float): x座標。
            y (float): y座標。

        Returns:
            int: 追加した点のID。
        """
        p = self.pointMgr.addPoint(x, y)
        return p
    
    def calcPerpendicularBisector(self, p1: int, p2: int) -> int:
        """
        線分(p1,p2)の垂直二等分線を作成します。

        Args:
            p1 (int): 始点インデックス。
            p2 (int): 終点インデックス。

        Returns:
            int: 垂直二等分線のID。
        """
        assert p1 != p2, "p1 == p2 "
        midx = (self.pointMgr.pointList[p1][0] + self.pointMgr.pointList[p2][0]) / 2
        midy = (self.pointMgr.pointList[p1][1] + self.pointMgr.pointList[p2][1]) / 2
        
        delx = self.pointMgr.pointList[p2][0] - self.pointMgr.pointList[p1][0]
        dely = self.pointMgr.pointList[p2][1] - self.pointMgr.pointList[p1][1]
        line: list[Union[int, float]] = [self.addPoint(midx, midy), self.addPoint(midx - dely, midy + delx), 0, 0, 0]
        line[4] = self.pointMgr.calcAngleOfLine(int(line[0]), int(line[1]))
        self.lineCount += 1
        self.lineList.append(line)
        
        return self.lineCount - 1
    
    def addLine(self, p1: int, p2: int, fin1: int = 0, fin2: int = 0) -> int:
        """
        線分を追加します。

        Args:
            p1 (int): 始点インデックス。
            p2 (int): 終点インデックス。
            fin1 (int, optional): 始点の端点フラグ。デフォルトは0。
            fin2 (int, optional): 終点の端点フラグ。デフォルトは0。

        Returns:
            int: 追加した線分のID。
        """
        assert p1 != p2, "too match degree of freedom."
        line: list[Union[int, float]] = [p1, p2, fin1, fin2, 0]
        line[4] = self.pointMgr.calcAngleOfLine(int(line[0]), int(line[1]))
        self.lineCount += 1
        self.lineList.append(line)
        return self.lineCount - 1
    
    def cutLine(self, line: int, cutPoint: int, direction: int) -> None:
        """
        指定した線分をcutPointで切断し、direction側を取り除きます。

        Args:
            line (int): 線分ID。
            cutPoint (int): 切断点のインデックス。
            direction (int): 1または2。どちら側を取り除くか。

        Returns:
            None
        """
        assert self.containLoc(line,self.pointMgr.pointList[cutPoint][0], self.pointMgr.pointList[cutPoint][1]), "cutPoint is not contained in the line."
        
        lineInfo = self.lineList[line]
        
        if direction == 1:#p1側を取り除く場合
            delx = self.pointMgr.pointList[int(lineInfo[1])][0] - self.pointMgr.pointList[int(lineInfo[0])][0]
            dely = self.pointMgr.pointList[int(lineInfo[1])][1] - self.pointMgr.pointList[int(lineInfo[0])][1]
            self.lineList[line][0] = cutPoint
            self.lineList[line][2] = 1
            if lineInfo[3] == 0:#p2側が無限に長い場合は、cutPoint-p1分だけp2も平行移動させる
                self.lineList[line][1] = self.addPoint(self.pointMgr.pointList[cutPoint][0] + delx, self.pointMgr.pointList[cutPoint][1] + dely)
        else:#p2側を取り除く場合
            delx = self.pointMgr.pointList[int(lineInfo[0])][0] - self.pointMgr.pointList[int(lineInfo[1])][0]
            dely = self.pointMgr.pointList[int(lineInfo[0])][1] - self.pointMgr.pointList[int(lineInfo[1])][1]
            self.lineList[line][1] = cutPoint
            self.lineList[line][3] = 1
            if lineInfo[2] == 0:#p1側が無限に長い場合は、cutPoint-p2分だけp1も平行移動させる
                self.lineList[line][0] = self.addPoint(self.pointMgr.pointList[cutPoint][0] + delx, self.pointMgr.pointList[cutPoint][1] + dely)
        
        
    def containLoc(self, line: int, x: float, y: float) -> bool:
        """
        点（x,y）が指定した線分の範囲に入っているか判定します。

        Args:
            line (int): 線分ID。
            x (float): x座標。
            y (float): y座標。

        Returns:
            bool: 範囲内ならTrue、そうでなければFalse。
        """
        lineInfo = self.lineList[line]
        if (lineInfo[2] == 0) and (lineInfo[3] == 0):#直線
             return True
        elif (lineInfo[2] != 0) and (lineInfo[3] == 0):#p1までの半直線
            if (lineInfo[4] + np.pi / 2) % np.pi != 0:#y軸に平行でない
                p1x = self.pointMgr.pointList[int(lineInfo[0])][0]
                if p1x < self.pointMgr.pointList[int(lineInfo[1])][0]:
                    return x >= p1x
                else:
                    return x <= p1x
            else:
                p1y = self.pointMgr.pointList[int(lineInfo[0])][1]
                if p1y < self.pointMgr.pointList[int(lineInfo[1])][1]:
                    return y >= p1y
                else:
                    return y <= p1y
        elif (lineInfo[2] == 0) and (lineInfo[3] != 0):#p2までの半直線
            if (lineInfo[4] + np.pi / 2) % np.pi != 0:#y軸に平行でない
                p2x = self.pointMgr.pointList[int(lineInfo[1])][0]
                if p2x < self.pointMgr.pointList[int(lineInfo[0])][0]:
                    return x >= p2x
                else:
                    return x <= p2x
            else:
                p2y = self.pointMgr.pointList[int(lineInfo[1])][1]
                if p2y < self.pointMgr.pointList[int(lineInfo[0])][1]:
                    return y >= p2y
                else:
                    return y <= p2y
        else:#線分
            if (lineInfo[4] + np.pi / 2) % np.pi != 0:#y軸に平行でない
                p1x = self.pointMgr.pointList[int(lineInfo[0])][0]
                p2x = self.pointMgr.pointList[int(lineInfo[1])][0]
                
                if p1x < p2x:
                    return (x >= p1x) and (x <= p2x)
                else:
                    return (x <= p1x) and (x >= p2x)
            else:
                p1y = self.pointMgr.pointList[int(lineInfo[0])][1]
                p2y = self.pointMgr.pointList[int(lineInfo[1])][1]
                if p1y < p2y:
                    return (y >= p1y) and (y <= p2y)
                else:
                    return (y <= p1y) and (y >= p2y)
                
    def calcIntersection(self, line1: int, line2: int) -> Optional[int]:
        """
        2つの線分の交点を返します。交点がない場合はNone、交点が無限遠点の場合は-1を返します。

        Args:
            line1 (int): 線分1のID。
            line2 (int): 線分2のID。

        Returns:
            Optional[int]: 交点の点ID、またはNone、または-1（無限遠点）。
        """
        line1Info = self.lineList[line1]
        line2Info = self.lineList[line2]
        
        o1 = self.pointMgr.pointList[int(line1Info[0])]
        o2 = self.pointMgr.pointList[int(line2Info[0])]
        
        d1 = self.pointMgr.pointList[int(line1Info[1])]
        d2 = self.pointMgr.pointList[int(line2Info[1])]
        
        delx1 = d1[0] - o1[0]
        dely1 = d1[1] - o1[1]
        delx2 = d2[0] - o2[0]
        dely2 = d2[1] - o2[1]
        a1 = [-dely1,delx1,0]
        a2 = [-dely2,delx2,0]
        
        a1[2] = -(a1[0] * o1[0] + a1[1] * o1[1])
        a2[2] = -(a2[0] * o2[0] + a2[1] * o2[1])
        d = a1[0] * a2[1] - a1[1] * a2[0]
        
        if d != 0:#交点存在
            tmpx = (a1[1] * a2[2] - a1[2] * a2[1]) / d
            tmpy = (a1[2] * a2[0] - a1[0] * a2[2]) / d
            addP = self.addPoint(tmpx, tmpy)
            if self.containLoc(line1, self.pointMgr.pointList[addP][0],self.pointMgr.pointList[addP][1]) and self.containLoc(line2, self.pointMgr.pointList[addP][0],self.pointMgr.pointList[addP][1]):#線分がある範囲で交わるか
                return addP#index of point
            else:
                return None
        elif (a1[1] * a2[2] - a1[2] * a2[1] == 0) and (a1[2] * a2[0] - a1[0] * a2[2] == 0):#2直線一致
            return None
        else:#交点なし、-1は無限遠点
            return -1