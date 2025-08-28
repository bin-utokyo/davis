//
//  Signal.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/18.
//

#ifndef Signal_hpp
#define Signal_hpp

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Hongo {

class Node;
class Link;
class Lane;
struct SignalProp {
  // 信号のデフォルト値設定用
  const double minRed = 60.0;
  double blue = 57.0;
  double pedBlue = 47.0;
  double yellow = 3.0;
  double allRed = 5.0;
  double rightBlue = 15.0;
  double red, cycle;

  SignalProp() {}

  void setRed(int numOfDirection, bool rightTurn) {
    if (numOfDirection <= 1) allRed = minRed;
    if (!rightTurn) {
      cycle = (blue + yellow + allRed) * numOfDirection;
      red = cycle - (blue + yellow + allRed);
    } else {
      cycle = (blue + yellow + rightBlue + yellow + allRed) * numOfDirection;
      red = cycle - (blue + yellow + rightBlue + yellow + allRed);
    }
  }
  double getOffset(int tmpDirection) {
    // tmpDirectionは0以上numOfDirection未満
    return (cycle - red) * tmpDirection;
  }
};

class Signal : public std::enable_shared_from_this<Signal> {
  int id;
  std::unordered_map<int, std::weak_ptr<Node>> nodeMap;
  SignalProp signalProp;

  bool rightTurn;
  std::vector<std::shared_ptr<Lane>>
      laneVec;  // 信号に接続されている車線のベクトル
  std::vector<int>
      dLinkVec;  // 信号で制御される動線の下流リンク（前リンクを制御対象とする場合は，使わない）
  std::vector<double> offsetVec;  // laneVec順のoffset
  std::vector<double> blueVec;    // laneVec順の青時間長さ

 public:
  Signal(int _id, std::vector<std::shared_ptr<Node>> _nodeVec, bool _rightTurn);
  Signal(int _id, std::vector<std::shared_ptr<Node>> _nodeVec, double blue,
         double yellow, double rightBlue);  // 信号周期を設定する場合
  Signal(int _id, std::vector<std::shared_ptr<Node>> _nodeVec,
         std::vector<std::shared_ptr<Lane>> _laneVec,
         std::vector<int> _dLinkVec, double cycle, std::vector<double> _blueVec,
         std::vector<double> _offsetVec);  // レーンごとに信号現示を設定する場合

  bool isPassible(const std::shared_ptr<Lane> &lane, int mode, int simTime);
  bool isPassible(const std::shared_ptr<Lane> &lane, int mode, int simTime,
                  int dLinkId);  // 行き先指定の場合（dLinkId）

  std::shared_ptr<Node> getNode(int nodeId) {
    if (nodeMap.find(nodeId) != nodeMap.end()) {
      return nodeMap.at(nodeId).lock();
    }
    return nullptr;
  }
};
}  // namespace Hongo

#endif /* Signal_hpp */
