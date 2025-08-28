//
//  Agent.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/08.
//

#include "Agent.hpp"

#include "Bus.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "Node.hpp"
#include "Pedestrian.hpp"
#include "RL.hpp"
#include "Station.hpp"
#include "Utility.hpp"
#include "Vehicle.hpp"

namespace Hongo {
double MigrationAttribute::getUtility(const std::shared_ptr<Node> &currentNode,
                                      int mode, int currentTime,
                                      const MigrationCoefficient &coeff) {
  double ox, oy, dx, dy;
  ox = currentNode->getX();
  oy = currentNode->getY();
  dx = node->getX();
  dy = node->getY();
  if (currentTime >= startTime && currentTime <= endTime) {
    return coeff.time * getTime(Utility::euclidean(ox, oy, dx, dy), mode) +
           coeff.spa * spa + coeff.shop * shop + coeff.famous * famous +
           coeff.art * art;
  } else {
    return 0.0;
  }
}
double MigrationAttribute::getUtility(int currentTime,
                                      const MigrationCoefficient &coeff) {
  if (currentTime >= startTime && currentTime <= endTime) {
    return coeff.spa * spa + coeff.shop * shop + coeff.famous * famous +
           coeff.art * art;
  } else {
    return 0.0;
  }
}

Agent::Agent(int _id, int _startTime, int _endTime, std::string _purpose,
             int _mode, const std::shared_ptr<Node> &oNode,
             const std::shared_ptr<Node> &dNode,
             const std::unordered_map<int, std::shared_ptr<MigrationAttribute>>
                 &_facilityMap)
    : id(_id),
      mode(_mode),
      oNodeId(oNode->getId()),
      dNodeId(dNode->getId()),
      tmpDNodeId(oNode->getId()),
      startTime(_startTime),
      endTime(_endTime),
      purpose(_purpose) {
  facilityMap = _facilityMap;

  std::unordered_map<int, std::shared_ptr<Link>> actLinkMap;
  std::shared_ptr<Node> oActNode =
      std::make_shared<Node>(0, oNode->getLon(), oNode->getLat());
  std::shared_ptr<Node> dActNode =
      std::make_shared<Node>(1, dNode->getLon(), dNode->getLat());
  tmpActNodeId = 0;

  if (!ecursion) {  // 回遊モデルを使用しない場合
    actLinkMap[1] = std::make_shared<Link>(1, oActNode, dActNode);
    actLinkMap.at(1)->setActUtility(1.0);
    actNodeIdMap[1] = dNode->getId();
    upperRL = std::make_shared<RL>(actLinkMap, -1);
    upperRL->setDNodeId(0, 1);
    return;
  }

  int depth = std::max(2, (_endTime - _startTime) / activityInterval + 1);
  std::vector<std::shared_ptr<Node>> prevActNodes;
  std::unordered_map<int, std::shared_ptr<Link>>
      prevActLinkMap;  // key:nodeID, value:ノードに接続するリンク
  std::vector<std::shared_ptr<Node>> tmpActNodes;

  int tmpLinkId = 1;
  int tmpNodeId = 2;
  double util;
  prevActNodes.emplace_back(oActNode);
  for (int i = 1; i < depth; i++) {
    for (const auto &pair : facilityMap) {
      if (pair.first == dNodeId) {
        continue;
      }
      std::shared_ptr<Node> tmpNode1 =
          std::make_shared<Node>(tmpNodeId, pair.second->getNode()->getLon(),
                                 pair.second->getNode()->getLat());
      tmpNodeId++;
      std::shared_ptr<Node> tmpNode2 =
          std::make_shared<Node>(tmpNodeId, pair.second->getNode()->getLon(),
                                 pair.second->getNode()->getLat());
      tmpLinkId++;

      tmpActNodes.emplace_back(tmpNode2);
      std::shared_ptr<Link> nodeLink = std::make_shared<Link>(
          tmpLinkId, tmpNode1,
          tmpNode2);  // dnLinkの追加回数を減らすため、nodeのところを一本のリンクで仲介させる
      actLinkMap[tmpLinkId] = nodeLink;
      nodeLink->setActUtility(1.0);
      prevActLinkMap[tmpNode2->getId()] = nodeLink;
      tmpLinkId++;

      for (const auto &prevNode : prevActNodes) {
        if (tmpNode1->getX() == prevNode->getX() &&
            tmpNode1->getY() ==
                prevNode->getY()) {  // 前のノードと同じノードは目的地にしない
          continue;
        }
        actLinkMap[tmpLinkId] =
            std::make_shared<Link>(tmpLinkId, prevNode, tmpNode1);
        util = pair.second->getUtility(
            prevNode, mode,
            _startTime + activityInterval * (i - 1) +
                pair.second->getTime(
                    Utility::euclidean(prevNode->getX(), prevNode->getY(),
                                       tmpNode1->getX(), tmpNode1->getY()),
                    mode),
            migrationCoefficient);
        actLinkMap.at(tmpLinkId)->setActUtility(util);

        actLinkMap.at(tmpLinkId)->addDnLink(tmpNode1->getId(),
                                            nodeLink);  // 接続情報
        if (prevActLinkMap.find(prevNode->getId()) != prevActLinkMap.end()) {
          prevActLinkMap.at(prevNode->getId())
              ->addDnLink(prevNode->getId(),
                          actLinkMap.at(tmpLinkId));  // 接続情報
        }

        actNodeIdMap[tmpLinkId] =
            pair.first;  // tmpLinkの終点ノード（アクティビティ用のノードでなく、実ネットワークのノード）
        tmpLinkId++;
      }
      tmpNodeId++;
    }
    prevActNodes = tmpActNodes;
    tmpActNodes.clear();
  }
  for (const auto &prevNode : prevActNodes) {
    actLinkMap[tmpLinkId] =
        std::make_shared<Link>(tmpLinkId, prevNode, dActNode, 1.0);
    actLinkMap.at(tmpLinkId)->setActUtility(1.0);
    actNodeIdMap[tmpLinkId] = dNode->getId();
    if (prevActLinkMap.find(prevNode->getId()) != prevActLinkMap.end()) {
      prevActLinkMap.at(prevNode->getId())
          ->addDnLink(prevNode->getId(), actLinkMap.at(tmpLinkId));
    }
    tmpLinkId++;
  }
  upperRL = std::make_shared<RL>(actLinkMap, -1);
  upperRL->setBeta(0.1);
  upperRL->setDNodeId(
      0,
      1);  // linkMapはここでセットされる。ChoiceSet::dijkstraが大きなlinkMapを持っているので、
  upperRL->resetDijkstra();  // リセットする。
  upperRL->trimDnLinks();    // 必要のないリンクを削除
}

Agent::Agent(int _id, int _startTime, int _mode,
             const std::shared_ptr<Node> &oNode,
             const std::shared_ptr<Node> &dNode)
    : id(_id),
      mode(_mode),
      oNodeId(oNode->getId()),
      dNodeId(dNode->getId()),
      tmpDNodeId(oNode->getId()),
      startTime(_startTime) {
  transportation = true;
}

void Agent::initializeLowerRL(
    const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap) {
  if (transportation && mode == 6) {
    bus->initializeLowerRL(_linkMap);
    return;
  }
  if (mode == 0) {  // 車
    vehicle = std::make_shared<Vehicle>(shared_from_this());
    lowerRL = std::make_shared<RL>(_linkMap, 0);
  } else if (mode == 3) {  // 歩行者
    pedestrian = std::make_shared<Pedestrian>(shared_from_this());
    lowerRL = std::make_shared<RL>(_linkMap, 3);
  } else if (mode == 6) {
    pedestrian = std::make_shared<Pedestrian>(shared_from_this());
    lowerRL = std::make_shared<RL>(_linkMap, 3);
  }
}

void Agent::initializeTransRL(
    const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap) {
  if (transportation || mode != 6) {
    return;
  }
  transRL =
      std::make_shared<RL>(_linkMap, 6);  // 公共交通を利用するエージェント用
}

void Agent::setLowerRL(double _simTime) {
  simTime = _simTime;
  if (transportation && mode == 6) {
    bus->setLowerRL(simTime);
    return;
  }
  lowerRL->setDNodeId(oNodeId, tmpDNodeId);
  link = lowerRL->chooseFirstLink(oNodeId);
  if (link == nullptr) {
    nodeLane = nullptr;
    lane = nullptr;
    direction = 0;
    return;
  }
  lane = lowerRL->chooseFirstLane(link, oNodeId);

  if (lane != nullptr) {
    lane->addAgent(shared_from_this(), simTime);
    if (link->getDNodeId() == oNodeId) {
      position.x = link->getLength() - link->getOffset(oNodeId);
      position.endx = link->getOffset(link->getAnotherNodeId(oNodeId));
      direction = -1;
    } else {
      position.x = link->getOffset(oNodeId);
      position.endx =
          link->getLength() - link->getOffset(link->getAnotherNodeId(oNodeId));
      direction = 1;
    }
    position.endy = lane->getWidth();
  }
}

void Agent::chooseDestination() {
  if (transportation) {  // バス
    return;
  }
  actLink = upperRL->chooseFirstLink(tmpActNodeId);
  if (actLink->getXY(actLink->getONodeId()) ==
      actLink->getXY(actLink->getDNodeId())) {  // nodeLink
    actLink = upperRL->chooseFirstLink(actLink->getDNodeId());
  }
  if (actLink == nullptr) {  // oNode,dNodeが一緒の場合で他に行き先がない場合
    oNodeId = tmpDNodeId;
    return;
  }
  oNodeId = tmpDNodeId;
  tmpActNodeId = actLink->getDNodeId();
  changeDestination = true;
  if (mode == 6) {  // tmpDNodeIdを次のバス停とする。
    chooseDestinationTransport();
  } else {
    tmpDNodeId = actNodeIdMap.at(actLink->getId());
  }
}

void Agent::chooseDestinationTransport() {
  std::vector<std::shared_ptr<Station>> oStations, dStations;
  oStations = lowerRL->getNearStations(oNodeId);
  dStations = lowerRL->getNearStations(actNodeIdMap.at(actLink->getId()));
  std::pair<double, std::vector<std::shared_ptr<Link>>> path =
      std::make_pair(std::numeric_limits<double>::infinity(),
                     std::vector<std::shared_ptr<Link>>());
  std::pair<double, std::vector<std::shared_ptr<Link>>> tmpPath;
  std::unordered_set<int> oModeSet, dModeSet;
  for (const auto &oSt : oStations) {
    oModeSet = oSt->getModeSet();
    if (oModeSet.find(mode) == oModeSet.end()) {
      continue;
    }
    for (const auto &dSt : dStations) {
      dModeSet = dSt->getModeSet();
      if (dModeSet.find(mode) == dModeSet.end()) {
        continue;
      }
      tmpPath = transRL->getShortestPath(oSt->getId(), dSt->getId());
      if (tmpPath.first < path.first) {
        path = tmpPath;
        oStation = oSt;
      }
    }
  }
  if (path.second.size() > 0) {
    transRL->setPath(path.second);
    std::shared_ptr<Lane> dLane = oStation->getLane();
    double length = dLane->getLink()->getLength();
    if (oStation->getX() < length / 2) {  // 近い方のノードをゴールとする
      tmpDNodeId = dLane->getLink()->getONodeId();
    } else {
      tmpDNodeId = dLane->getLink()->getDNodeId();
    }
  } else {
    tmpDNodeId = actNodeIdMap.at(actLink->getId());  // バスに乗らない
  }
}

void Agent::setDestinationTransport() {
  // すでにlinkとlane, positionはセットされている
  int tmpONodeId = link->getDNodeId();
  if (position.x < link->getLength() / 2) {
    tmpONodeId = link->getONodeId();
  }
  lowerRL->setDNodeId(tmpONodeId, tmpDNodeId);
  tmpDNodeId = actNodeIdMap.at(actLink->getId());

  transRL->setPath(std::vector<std::shared_ptr<Link>>());
}

std::shared_ptr<NodeLane> Agent::planDnLane() {
  int nodeId;
  std::shared_ptr<Link> dnLink;
  std::shared_ptr<NodeLane> dnNodeLane;
  int dnDirection;
  if (direction == 1) {
    nodeId = link->getDNodeId();
  } else {
    nodeId = link->getONodeId();
  }
  dnLink = lowerRL->chooseDnLink(link, lane, nodeId);
  if (dnLink != nullptr) {
    dnNodeLane = lowerRL->chooseDnLane(lane, dnLink, nodeId);
  } else {
    dnNodeLane = nullptr;
  }
  return dnNodeLane;
}

void Agent::setDnLane(const std::shared_ptr<NodeLane> &dnNodeLane) {
  if (dnNodeLane == nullptr) {
    link = nullptr;
    lane = nullptr;
    nodeLane = nullptr;
    direction = 0;
    return;
  }
  int nodeId;
  if (direction == 1) {
    nodeId = link->getDNodeId();
  } else {
    nodeId = link->getONodeId();
  }

  nodeLane = dnNodeLane;
  lane = nodeLane->getDnLane();
  link = lane->getLink();

  if (link->getONodeId() == nodeId) {
    direction = 1;
  } else {
    direction = -1;
  }
}

int Agent::getNextStation(int tmpStationId) {
  return transRL->getNextNodeId(tmpStationId);
}

std::shared_ptr<Lane> Agent::planLaneChange() {
  std::shared_ptr<Lane> sideLane = nullptr;
  if (inNodeLane()) {
    return sideLane;
  }
  int dNodeId;
  if (direction == 1) {  // linkのONodeからDNodeへ進んでいる場合
    dNodeId = link->getDNodeId();
  } else {
    dNodeId = link->getONodeId();
  }
  double pos =
      1.0 - (link->getOffset(dNodeId) + std::abs(position.endx - position.x)) /
                link->getLength();  // レーンのどれくらいの割合を進んだか

  sideLane = lowerRL->chooseSideLane(lane, dNodeId, pos);
  return sideLane;
}

int Agent::chooseStayTime(int timestep, double simTime) {
  const auto facility = facilityMap.at(tmpDNodeId);
  double stayUtility = facility->getUtility(simTime, migrationCoefficient);
  double moveUtility =
      upperRL->getVValue(actLink, actLink->getAnotherNodeId(tmpActNodeId));

  double maxUtility = std::max(stayUtility, moveUtility);

  return 60 * 60 * std::exp(stayUtility - maxUtility) /
         (timestep * (std::exp(stayUtility - maxUtility) +
                      std::exp(moveUtility - maxUtility)));
}

void Agent::updateAcceleration(double timestep, double _simTime) {
  simTime = _simTime;
  if (stayStep > 0 || bord) {
    return;
  }
  if (mode == 3) {
    pedestrian->updateAcceleration(timestep);
  } else if (mode == 0) {
    vehicle->updateAcceleration(timestep);
  } else if (transportation && mode == 6) {
    bus->updateAcceleration(timestep);
  } else if (mode == 6) {
    pedestrian->updateAcceleration(timestep);
  }
}
void Agent::updateVelocity(double timestep) {
  if (stayStep > 0 || bord) {
    return;
  }
  if (mode == 3) {
    pedestrian->updateVelocity(timestep);
  } else if (mode == 0) {
    vehicle->updateVelocity(timestep);
  } else if (transportation && mode == 6) {
    bus->updateVelocity(timestep);
  } else if (mode == 6) {
    pedestrian->updateVelocity(timestep);
  }
}

std::shared_ptr<Agent> Agent::moveAgent(double timestep, double simTime) {
  std::shared_ptr<Agent> finishAgent = nullptr;
  if (lane == nullptr) {
    return shared_from_this();
  }
  if (stayStep > 0 || bord) {
    return nullptr;
  }
  if (mode == 0) {
    finishAgent = vehicle->moveVehicle(timestep, simTime);
  } else if (mode == 3) {  // 歩行者
    finishAgent = pedestrian->movePedestrian(timestep, simTime);
  } else if (transportation && mode == 6) {
    finishAgent = bus->moveBus(
        timestep, simTime);  // すべての駅を回るまでnullptrが返り続ける。
  } else if (mode == 6) {
    finishAgent = pedestrian->movePedestrian(timestep, simTime);
    if (finishAgent != nullptr &&
        transRL->getPath().size() > 0) {  // アクセス終了
      oStation->addAgent(shared_from_this());
      bord = true;
      finishAgent = nullptr;
    }
  }
  double x, y;
  std::tie(x, y) = getGlobalCoord();
  if (!transportation && x != 0.0 && y != 0.0) {
    double dx, dy;
    std::tie(dx, dy) = actLink->getXY(tmpDNodeId);
    destinationDistance = Utility::euclidean(x, y, dx, dy);
  } else {
    destinationDistance = std::numeric_limits<double>::infinity();
  }
  if (finishAgent != nullptr) {
    if (tmpDNodeId == dNodeId) {
      return shared_from_this();
    } else {
      stayStep = chooseStayTime(timestep, simTime);
      chooseDestination();
      setLowerRL(simTime +
                 stayStep * timestep);  // 実際に移動を開始する時間をセット
      if (lane == nullptr) {
        return shared_from_this();
      }
    }
  }
  return nullptr;
}

void Agent::setPrevPosition() {
  prevPosition = Position(position);  // コピー代入
}

bool Agent::isStop() {
  if (transportation && mode == 6) {
    return bus->isStop();
  }
  return false;
}

bool Agent::isCarFollwing() {
  if (mode != 0) return false;  // 自動車以外
  return vehicle->isCarFollowing();
}

bool Agent::isSignalStop() {
  if (mode == 0) {  // 車
    return vehicle->isSignalStop();
  } else if (mode == 3) {  // 歩行者
    return pedestrian->isSignalStop();
  } else if (mode == 6) {
    if (transportation) {  // バス
      if (bus != nullptr) {
        return bus->isSignalStop();
      } else {
        return false;
      }
    } else if (!bord) {  // バス乗車中以外
      return pedestrian->isSignalStop();
    }
  }
  return false;
}

bool Agent::isLastTrip() {
  return dNodeId == actNodeIdMap.at(actLink->getId());
}

std::shared_ptr<MigrationAttribute> Agent::getDestinationAttribute() {
  std::shared_ptr<MigrationAttribute> ma = nullptr;
  int _tmpDNodeId = actNodeIdMap.at(actLink->getId());
  if (facilityMap.find(_tmpDNodeId) != facilityMap.end()) {
    ma = facilityMap.at(_tmpDNodeId);
  }
  return ma;
}

std::pair<double, double> Agent::getGlobalCoord() {
  double x = 0.0;
  double y = 0.0;
  if (lane == nullptr) {
    return std::make_pair(0.0, 0.0);
  }

  if (inNodeLane()) {
    if (nodeLane->getR() == 0.0) {  // 直線
      double ox, oy, dx, dy, ix, iy;
      std::pair<double, double> xe, ye;
      std::tie(ox, oy) = nodeLane->getOXY();
      std::tie(dx, dy) = nodeLane->getDXY();
      std::tie(ix, iy) = nodeLane->getIXY();
      if (ix == 0.0 && iy == 0.0) {  // ただの直線
        xe = std::make_pair((dx - ox) / nodeLane->getLength(),
                            (dy - oy) / nodeLane->getLength());
        ye = std::make_pair((dy - oy) / nodeLane->getLength(),
                            -(dx - ox) / nodeLane->getLength());
        if (lane->getONodeId() != nodeLane->getNodeId()) {
          ye.first = -ye.first;
          ye.second = -ye.second;
        }

        x = ox + xe.first * position.x + ye.first * position.y;
        y = oy + xe.second * position.x + ye.second * position.y;
      } else {  // 折線
        double length1 = Utility::euclidean(ox, oy, ix, iy);
        if (position.x < length1) {
          double angle = Utility::angle(Utility::direction(ox, oy, ix, iy),
                                        link->getDirection());
          bool turn = angle > 40. / 180. * M_PI && angle < 150. / 180. * M_PI;
          xe = std::make_pair((ix - ox) / length1, (iy - oy) / length1);
          ye = std::make_pair((iy - oy) / length1, -(ix - ox) / length1);
          if ((turn && lane->getONodeId() != nodeLane->getNodeId()) ||
              (!turn && lane->getONodeId() == nodeLane->getNodeId())) {
            ye.first = -ye.first;
            ye.second = -ye.second;
          }

          x = ox + xe.first * position.x + ye.first * position.y;
          y = oy + xe.second * position.x + ye.second * position.y;
        } else {
          double length2 = Utility::euclidean(ix, iy, dx, dy);
          xe = std::make_pair((dx - ix) / length2, (dy - iy) / length2);
          ye = std::make_pair((dy - iy) / length2, -(dx - ix) / length2);
          if (lane->getONodeId() != nodeLane->getNodeId()) {
            ye.first = -ye.first;
            ye.second = -ye.second;
          }

          x = ix + xe.first * (position.x - length1) + ye.first * position.y;
          y = iy + xe.second * (position.x - length1) + ye.second * position.y;
        }
      }
    } else {  // 曲線
      double cx, cy, oTheta, dTheta, theta, r;
      std::tie(oTheta, dTheta) = nodeLane->getTheta();
      theta = (oTheta * (nodeLane->getLength() - position.x) +
               dTheta * position.x) /
              nodeLane->getLength();
      std::tie(cx, cy) = nodeLane->getCenterXY();

      if (dTheta < oTheta) {
        dTheta = dTheta + 2 * M_PI;
      }
      if (dTheta - oTheta < M_PI) {
        r = nodeLane->getR() - position.y;
      } else {
        r = nodeLane->getR() + position.y;
      }

      x = cx + std::cos(theta) * r;
      y = cy + std::sin(theta) * r;
    }
  } else {
    std::pair<double, double> coord, xe, ye;  // coord=laneの原点の(x,y)
    coord = link->getXY(link->getONodeId());
    xe = link->getE();
    ye = link->getN();

    coord.first += ye.first * lane->getCenterOffset();
    coord.second += ye.second * lane->getCenterOffset();
    if (lane->getId() < 0) {
      ye.first = -ye.first;
      ye.second = -ye.second;
    }
    x = coord.first + position.x * xe.first + position.y * ye.first;
    y = coord.second + position.x * xe.second + position.y * ye.second;
  }
  return Utility::xy2LonLat(x, y);
}

Position Agent::getGlobalPosition() {
  Position pos;
  if (lane == nullptr) {
    return pos;
  }

  if (inNodeLane()) {
    if (nodeLane->getR() == 0.0) {  // 直線
      double ox, oy, dx, dy, ix, iy;
      std::pair<double, double> xe, ye;
      std::tie(ox, oy) = nodeLane->getOXY();
      std::tie(dx, dy) = nodeLane->getDXY();
      std::tie(ix, iy) = nodeLane->getIXY();
      if (ix == 0.0 && iy == 0.0) {  // ただの直線
        xe = std::make_pair((dx - ox) / nodeLane->getLength(),
                            (dy - oy) / nodeLane->getLength());
        ye = std::make_pair((dy - oy) / nodeLane->getLength(),
                            -(dx - ox) / nodeLane->getLength());
        if (lane->getONodeId() != nodeLane->getNodeId()) {
          ye.first = -ye.first;
          ye.second = -ye.second;
        }

        pos.x = ox + xe.first * position.x + ye.first * position.y;
        pos.y = oy + xe.second * position.x + ye.second * position.y;
        pos.dx = xe.first * position.dx + ye.first * position.dy;
        pos.dy = xe.second * position.dx + ye.second * position.dy;
      } else {  // 折線
        double length1 = Utility::euclidean(ox, oy, ix, iy);
        if (position.x < length1) {
          double angle = Utility::angle(Utility::direction(ox, oy, ix, iy),
                                        link->getDirection());
          bool turn = angle > 40. / 180. * M_PI && angle < 150. / 180. * M_PI;
          xe = std::make_pair((ix - ox) / length1, (iy - oy) / length1);
          ye = std::make_pair((iy - oy) / length1, -(ix - ox) / length1);
          if ((turn && lane->getONodeId() != nodeLane->getNodeId()) ||
              (!turn && lane->getONodeId() == nodeLane->getNodeId())) {
            ye.first = -ye.first;
            ye.second = -ye.second;
          }

          pos.x = ox + xe.first * position.x + ye.first * position.y;
          pos.y = oy + xe.second * position.x + ye.second * position.y;
          pos.dx = xe.first * position.dx + ye.first * position.dy;
          pos.dy = xe.second * position.dx + ye.second * position.dy;
        } else {
          double length2 = Utility::euclidean(ix, iy, dx, dy);
          xe = std::make_pair((dx - ix) / length2, (dy - iy) / length2);
          ye = std::make_pair((dy - iy) / length2, -(dx - ix) / length2);
          if (lane->getONodeId() != nodeLane->getNodeId()) {
            ye.first = -ye.first;
            ye.second = -ye.second;
          }

          pos.x =
              ix + xe.first * (position.x - length1) + ye.first * position.y;
          pos.y =
              iy + xe.second * (position.x - length1) + ye.second * position.y;
          pos.dx = xe.first * position.dx + ye.first * position.dy;
          pos.dy = xe.second * position.dx + ye.second * position.dy;
        }
      }
    } else {  // 曲線
      double cx, cy, oTheta, dTheta, theta;
      std::tie(oTheta, dTheta) = nodeLane->getTheta();
      theta = (oTheta * (nodeLane->getLength() - position.x) +
               dTheta * position.x) /
              nodeLane->getLength();
      std::tie(cx, cy) = nodeLane->getCenterXY();

      pos.x = cx + std::cos(theta) * nodeLane->getR();
      pos.y = cy + std::sin(theta) * nodeLane->getR();
      pos.dx = -position.dx * std::sin(theta) * (dTheta - oTheta) /
               std::abs(dTheta - oTheta);
      pos.dy = position.dx * std::cos(theta) * (dTheta - oTheta) /
               std::abs(dTheta - oTheta);
    }
  } else {
    std::pair<double, double> coord, xe, ye;  // coord=laneの原点の(x,y)
    coord = link->getXY(link->getONodeId());
    xe = link->getE();
    ye = link->getN();

    coord.first += ye.first * lane->getCenterOffset();
    coord.second += ye.second * lane->getCenterOffset();
    if (lane->getId() < 0) {
      ye.first = -ye.first;
      ye.second = -ye.second;
    }
    pos.x = coord.first + position.x * xe.first + position.y * ye.first;
    pos.y = coord.second + position.x * xe.second + position.y * ye.second;
    pos.dx = xe.first * position.dx + ye.first * position.dy;
    pos.dy = xe.second * position.dx + ye.second * position.dy;
  }
  return pos;
}

double Agent::getLength() {
  if (mode == 0 || (mode == 6 && transportation)) {
    return vehicle->getLength();
  } else {
    return 1.0;
  }
}

}  // namespace Hongo
