//
//  RL.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/28.
//

#ifndef RL_hpp
#define RL_hpp

#include <float.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Hongo {

class Link;
class Node;
class Lane;
class NodeLane;
class ChoiceSet;
class Station;
class RL : public std::enable_shared_from_this<RL> {
  std::shared_ptr<ChoiceSet> choiceSet;
  std::unordered_map<int, std::shared_ptr<Link>> linkMap;

  int mode;
  bool bus = false;  // バス車両か否か
  int oNodeId;
  int dNodeId;  // 目的地ノードのId
  std::shared_ptr<Lane>
      dLane;  // 目的地に到着するために通る最後の車線が指定されている場合（ex.バスの停留所）
  std::unordered_map<int, std::pair<double, double>>
      V;  // key:linkID, value:log(sum(exp(sum(V下流リンク))))(first:
          // oNodeからdNodeに行く場合, second: 逆方向）無向リンクのため
  double beta = 0.9;  // 時間割引率
  double mu = 1.0;    // スケールファクター

  std::vector<std::shared_ptr<Link>> path;
  std::unordered_map<int, int> pathMap;  // key: linkID, value: index

  std::unordered_map<int, std::vector<std::shared_ptr<Link>>> odLinkMap;

 public:
  RL(std::unordered_map<int, std::shared_ptr<Link>> _linkMap,
     int _mode);  // mode=-1は回遊モデル

  double getLaneUtility(const std::shared_ptr<Lane> &lane,
                        int nodeId);  // nodeIdは下流側ノードのID
  std::shared_ptr<Link> chooseFirstLink(int nodeId);  // nodeから出発する場合
  std::shared_ptr<Lane> chooseFirstLane(const std::shared_ptr<Link> &nextLink,
                                        int nodeId);
  std::shared_ptr<Link> chooseDnLink(
      const std::shared_ptr<Link> &link, const std::shared_ptr<Lane> &lane,
      int nodeId);  // 下流リンクを選択,
                    // nodeIdは今いるノードのID(linkの下流側のノード)
  std::shared_ptr<NodeLane> chooseDnLane(
      const std::shared_ptr<Lane> &lane, const std::shared_ptr<Link> &nextLink,
      int nodeId);  // 次にいくリンクが決まった状態で移動先レーンを選択する
  std::shared_ptr<Lane> chooseSideLane(
      const std::shared_ptr<Lane> &lane, int nodeId,
      double
          position);  // nodeIdは下流側ノードのID,
                      // positionはlane内に入った直後なら0でlaneから出る直前なら1
  void setDNodeId(int _oNodeId, int _dnodeId);
  void setDNodeId(const std::shared_ptr<Lane> &_oLane, int _dnodeId);
  void setDLane(int _oNodeId, const std::shared_ptr<Lane> &_dLane);
  void setDLane(const std::shared_ptr<Lane> &_oLane,
                const std::shared_ptr<Lane> &_dLane);

  std::vector<std::shared_ptr<Station>> getNearStations(int oNodeId);
  std::pair<double, std::vector<std::shared_ptr<Link>>> getShortestPath(
      int oNodeId, int dNodeId);

  void resetDijkstra();
  void trimDnLinks();  // linkMapに入っていないdnLinkを削除する

  void setPath(std::vector<std::shared_ptr<Link>> _path);
  std::vector<std::shared_ptr<Link>> getPath() { return path; }
  std::shared_ptr<Link> getNextLink(int linkId);
  int getNextNodeId(int tmpNodeId);

  void setK(int _k);

  // リンク列挙→下流リンクの効用計算→次のリンクを選択 の場合
  void calcV_dn(int oNodeId);  // nodeIdは出発地点のノードID

  void recursive_dn(const std::shared_ptr<Link> &link, int nodeId,
                    std::unordered_map<int, std::pair<bool, bool>> &active,
                    double &dL);  // nodeIdはlinkの上流側のノードID

  // ベルマン方程式解法
  void calcV_bellman();

  // ベルマン方程式＋階層化
  void calcV_bellmanLayer();

  double getVValue(const std::shared_ptr<Link> &link,
                   int nodeId);  // nodeIdはlinkの上流側のノードのID
  void setVValue(const std::shared_ptr<Link> &link, int nodeId,
                 double value);  // nodeIdはlinkの上流側のノードのID

  std::unordered_map<int, std::pair<double, double>> getV() { return V; }
  double getBeta() { return beta; }
  void setBeta(double _beta) { beta = _beta; }
  void setBus(bool _bus) { bus = _bus; }
  void clearV() { V.clear(); }
  void setMu(double _mu) { mu = _mu; }
};
}  // namespace Hongo

#endif /* RL_hpp */
