//
//  Signal.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/18.
//

#include "Signal.hpp"

#include "Lane.hpp"
#include "Link.hpp"
#include "Node.hpp"
#include "Utility.hpp"

namespace Hongo {

Signal::Signal(int _id, std::vector<std::shared_ptr<Node>> _nodeVec,
               bool _rightTurn)
    : id(_id), rightTurn(_rightTurn) {
  int tmpDirectionNum = 0;
  for (const auto &node : _nodeVec) {
    nodeMap.emplace(node->getId(), node);
  }
  std::unordered_map<int, int> linkDirectionMap;
  for (const auto &node : _nodeVec) {
    const std::map<double, std::shared_ptr<Link>> &linkDirection =
        node->getLinkDirection();  // linkのONodeからDNodeの方向
    for (const auto &pair : linkDirection) {
      if (nodeMap.find(pair.second->getONodeId()) != nodeMap.end() &&
          nodeMap.find(pair.second->getDNodeId()) != nodeMap.end()) {
        continue;  // リンクの両端が交差点ノードになっている場合
      }
      if (linkDirectionMap.find(pair.second->getId()) ==
          linkDirectionMap
              .end()) {  // ノードから伸びているリンクがまだmapに入っていない場合（信号の同じフェーズのリンクは次のforループで追加されているため新しいフェーズとして追加される）
        linkDirectionMap[pair.second->getId()] = tmpDirectionNum;
        int sign = 1;
        if (pair.second->getDNodeId() != node->getId()) {
          sign = -1;
        }
        for (const auto &lane : pair.second->getLanes(sign)) {
          laneVec.emplace_back(lane);
        }
        for (const auto &lane : pair.second->getLanes(0)) {
          laneVec.emplace_back(lane);
        }
        for (const auto &pair2 :
             linkDirection) {  // pair.secondと同じフェーズのリンクを抽出
          if (linkDirectionMap.find(pair2.second->getId()) ==
              linkDirectionMap.end()) {
            double angle = Utility::angle(pair.first, pair2.first);
            if (angle < 40.0 / 180.0 * M_PI || angle > 140.0 / 180.0 * M_PI) {
              linkDirectionMap[pair2.second->getId()] = tmpDirectionNum;
              int sign = 1;
              if (pair2.second->getDNodeId() != node->getId()) {
                sign = -1;
              }
              for (const auto &lane : pair2.second->getLanes(sign)) {
                laneVec.emplace_back(lane);
              }
              for (const auto &lane : pair2.second->getLanes(0)) {
                laneVec.emplace_back(lane);
              }
            }
          }
        }
        tmpDirectionNum++;
      }
    }
  }
  dLinkVec = std::vector<int>(laneVec.size(), -1);

  // 信号周期・offsetの設定
  signalProp.setRed(tmpDirectionNum, rightTurn);
  for (const auto &lane : laneVec) {
    int tmpDirectionNum = linkDirectionMap.at(lane->getLink()->getId());
    offsetVec.emplace_back(signalProp.getOffset(tmpDirectionNum));
    blueVec.emplace_back(signalProp.blue);
  }
}

Signal::Signal(int _id, std::vector<std::shared_ptr<Node>> _nodeVec,
               double blue, double yellow, double rightBlue)
    : id(_id), rightTurn(rightBlue > 0) {
  int tmpDirectionNum = 0;
  for (const auto &node : _nodeVec) {
    nodeMap.emplace(node->getId(), node);
  }
  std::unordered_map<int, int> linkDirectionMap;
  for (const auto &node : _nodeVec) {
    const std::map<double, std::shared_ptr<Link>> &linkDirection =
        node->getLinkDirection();
    for (const auto &pair : linkDirection) {
      if (nodeMap.find(pair.second->getONodeId()) != nodeMap.end() &&
          nodeMap.find(pair.second->getDNodeId()) != nodeMap.end()) {
        continue;  // リンクの両端が交差点ノードになっている場合
      }
      if (linkDirectionMap.find(pair.second->getId()) ==
          linkDirectionMap.end()) {
        linkDirectionMap[pair.second->getId()] = tmpDirectionNum;
        int sign = 1;
        if (pair.second->getDNodeId() != node->getId()) {
          sign = -1;
        }
        for (const auto &lane : pair.second->getLanes(sign)) {
          laneVec.emplace_back(lane);
        }
        for (const auto &lane : pair.second->getLanes(0)) {
          laneVec.emplace_back(lane);
        }
        for (const auto &pair2 : linkDirection) {
          if (linkDirectionMap.find(pair2.second->getId()) ==
              linkDirectionMap.end()) {
            double angle = Utility::angle(pair.first, pair2.first);
            if (angle < 40.0 / 180.0 * M_PI || angle > 140.0 / 180.0 * M_PI) {
              linkDirectionMap[pair2.second->getId()] = tmpDirectionNum;
              int sign = 1;
              if (pair2.second->getDNodeId() != node->getId()) {
                sign = -1;
              }
              for (const auto &lane : pair2.second->getLanes(sign)) {
                laneVec.emplace_back(lane);
              }
              for (const auto &lane : pair2.second->getLanes(0)) {
                laneVec.emplace_back(lane);
              }
            }
          }
        }
        tmpDirectionNum++;
      }
    }
  }
  dLinkVec = std::vector<int>(laneVec.size(), -1);

  // 信号周期・offsetの設定
  signalProp.blue = (blue > 0) ? blue : signalProp.blue;
  signalProp.yellow = (yellow > 0) ? yellow : signalProp.yellow;
  signalProp.rightBlue = (rightBlue > 0) ? rightBlue : signalProp.rightBlue;

  signalProp.setRed(tmpDirectionNum, rightTurn);
  for (const auto &lane : laneVec) {
    int tmpDirectionNum = linkDirectionMap.at(lane->getLink()->getId());
    offsetVec.emplace_back(signalProp.getOffset(tmpDirectionNum));
    blueVec.emplace_back(signalProp.blue);
  }
}
Signal::Signal(int _id, std::vector<std::shared_ptr<Node>> _nodeVec,
               std::vector<std::shared_ptr<Lane>> _laneVec,
               std::vector<int> _dLinkVec, double cycle,
               std::vector<double> _blueVec, std::vector<double> _offsetVec) {
  for (const auto &node : _nodeVec) {
    nodeMap.emplace(node->getId(), node);
  }
  id = _id;
  laneVec = _laneVec;
  dLinkVec = _dLinkVec;

  signalProp.cycle = cycle;
  blueVec = _blueVec;
  offsetVec = _offsetVec;

  signalProp.blue = 0;
  signalProp.pedBlue = 0;
  signalProp.yellow = 0;
  signalProp.allRed = 0;
  signalProp.rightBlue = 0;
  signalProp.red = 0;
}

bool Signal::isPassible(const std::shared_ptr<Lane> &lane, int mode,
                        int simTime) {
  double tmpTime = std::fmod((double)simTime, signalProp.cycle);
  for (size_t i = 0; i < laneVec.size(); i++) {
    if (lane == laneVec.at(i)) {  // laneVecに登録されているレーンの場合
      if (tmpTime > offsetVec.at(i) &&
          tmpTime <=
              offsetVec.at(i) + blueVec.at(i)) {  // 青信号の時間だった場合
        return true;
      } else {
        return false;
      }
    }
  }
  return true;  // laneVecに登録されていないレーンの場合
}

bool Signal::isPassible(const std::shared_ptr<Lane> &lane, int mode,
                        int simTime, int dLinkId) {
  double tmpTime = std::fmod((double)simTime, signalProp.cycle);
  for (size_t i = 0; i < laneVec.size(); i++) {
    if (lane == laneVec.at(i)) {  // laneVecに登録されているレーンの場合
      int tmpDLinkId = dLinkVec.at(i);
      if (tmpDLinkId > 0 &&
          tmpDLinkId != dLinkId) {  // 行き先のリンクが信号制御の対象でない場合
        continue;
      }
      if (tmpTime > offsetVec.at(i) &&
          tmpTime <=
              offsetVec.at(i) + blueVec.at(i)) {  // 青信号の時間だった場合
        return true;
      } else {
        return false;
      }
    }
  }
  return true;  // laneVecに登録されていないレーンの場合
}
}  // namespace Hongo
