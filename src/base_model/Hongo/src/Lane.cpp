//
//  Lane.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/29.
//

#include "Lane.hpp"

#include "Agent.hpp"
#include "Link.hpp"
#include "Node.hpp"

namespace Hongo {

Lane::Lane(int _laneId, const std::shared_ptr<Link> &_link, std::string _type,
           double _centerOffset, double _width)
    : laneId(_laneId),
      centerOffset(_centerOffset),
      width(_width),
      type(_type),
      link(_link) {
  if (_laneId > 0) {
    direction = _link->getDirection();
  } else {
    direction = Utility::oppositeDirection(_link->getDirection());
  }
}
int Lane::getONodeId() {
  if (laneId > 0) {
    return link.lock()->getONodeId();
  } else {
    return link.lock()->getDNodeId();
  }
}

int Lane::getDNodeId() {
  if (laneId > 0) {
    return link.lock()->getDNodeId();
  } else {
    return link.lock()->getONodeId();
  }
}
void Lane::addDnLane(int nodeId, const std::shared_ptr<Lane> &dnLane) {
  if (type != dnLane->getType() && (type + dnLane->getType() != "carbus") &&
      (type + dnLane->getType() != "buscar")) {
    if (type == "car" || dnLane->getType() == "car") {
      std::cerr << "Different type of lane cannot be connected." << std::endl;
      return;
    }
  }
  std::shared_ptr<Node> node;
  std::shared_ptr<NodeLane> tmpNodeLane;
  if (type == "car" || type == "bus") {
    if (getDNodeId() == nodeId && dnLane->getONodeId() == nodeId) {
      node = dnLane->getLink()->getNode(nodeId);
      tmpNodeLane =
          std::make_shared<NodeLane>(node, shared_from_this(), dnLane);
      dnLanes.emplace_back(tmpNodeLane);
      node->addNodeLane(tmpNodeLane);
    } else {
      return;
    }
  } else if (type == "ped" || type == "sidebelt") {
    if (getDNodeId() == nodeId &&
        (dnLane->getONodeId() == nodeId || dnLane->getDNodeId() == nodeId)) {
      node = dnLane->getLink()->getNode(nodeId);
      tmpNodeLane =
          std::make_shared<NodeLane>(node, shared_from_this(), dnLane);
      dnLanes.emplace_back(tmpNodeLane);
      node->addNodeLane(tmpNodeLane);
    } else if (
        getONodeId() == nodeId &&
        (dnLane->getONodeId() == nodeId ||
         dnLane->getDNodeId() ==
             nodeId)) {  // 車の進行方向とは逆の方向にdnLaneがあるような場合（upLaneからdnLaneを見たときに、中央線が左側に見える向きの場合）
      node = dnLane->getLink()->getNode(nodeId);
      tmpNodeLane =
          std::make_shared<NodeLane>(node, shared_from_this(), dnLane);
      upLanes.emplace_back(tmpNodeLane);
      node->addNodeLane(tmpNodeLane);
    } else {
      return;
    }
  }
  link.lock()->addDnLink(nodeId, dnLane->getLink());
}

bool Lane::isPassable(int oNodeId, int mode, bool bus) {
  if (mode == 3) {  // 歩行者
    if (type == "ped" || type == "sidebelt") {
      return true;
    } else {
      return false;
    }
  } else if (mode == 0) {  // 自動車
    if (bus) {
      if (type != "car" && type != "bus")
        return false;  // 車道かバス専用レーンしか通れない．
    } else {
      if (type != "car")
        return false;  // 車道でなければ通れない．かつバス専用レーンでない．
    }
    if (laneId > 0 && oNodeId == link.lock()->getONodeId()) {
      return true;
    } else if (laneId < 0 && oNodeId == link.lock()->getDNodeId()) {
      return true;
    }
  }
  return false;
}

bool Lane::isConnected(int nodeId, const std::shared_ptr<Link> &dnLink) {
  for (const auto &dnNodeLane : getDnLane(nodeId)) {
    if (dnNodeLane->getDnLink()->getId() == dnLink->getId()) {
      return true;
    }
  }
  return false;
}

bool Lane::isSaturated() {
  double space = 0.0;
  if (type == "car" || type == "bus") {
    space = 2.0;
  }
  double accumLength = std::accumulate(
      agentMap.begin(), agentMap.end(), 0.0,
      [=](double sum, const std::pair<int, std::shared_ptr<Agent>> &p) {
        return sum + p.second->getLength() + space;
      });
  return accumLength > link.lock()->getEffectiveLength();
}

void Lane::addAgent(const std::shared_ptr<Agent> &agent, double simTime) {
  agentMap.emplace(agent->getId(), agent);
  link.lock()->addAgent(agent, simTime);
}
void Lane::addAgent(const std::shared_ptr<Agent> &agent) {
  agentMap.emplace(agent->getId(), agent);
}
void Lane::removeAgent(int agentId, double simTime) {
  agentMap.erase(agentId);
  link.lock()->removeAgent(agentId, simTime);
}
void Lane::removeAgent(int agentId) { agentMap.erase(agentId); }

std::pair<double, double> Lane::getXYinLane(double x, double y) {
  double ox, oy, dx, dy;
  std::shared_ptr<Link> linkSpr = link.lock();
  std::tie(ox, oy) = linkSpr->getLaneXY(
      linkSpr->getONodeId(), centerOffset + y * (laneId / std::abs(laneId)));
  std::tie(dx, dy) = linkSpr->getLaneXY(
      linkSpr->getDNodeId(), centerOffset + y * (laneId / std::abs(laneId)));
  double length = linkSpr->getLength();
  if (length == 0) {
    return std::make_pair(0.0, 0.0);
  }
  return std::make_pair((ox * (length - x) + dx * x) / length,
                        (oy * (length - x) + dy * x) / length);
}
std::pair<double, double> Lane::getLonLatinLane(double x, double y) {
  double X, Y;
  std::tie(X, Y) = getXYinLane(x, y);
  if (X == 0 && Y == 0) {
    return std::make_pair(0.0, 0.0);
  }
  return Utility::xy2LonLat(X, Y);
}

NodeLane::NodeLane(const std::shared_ptr<Node> &_node,
                   const std::shared_ptr<Lane> &_upLane,
                   const std::shared_ptr<Lane> &_dnLane)
    : node(_node), dnLink(_dnLane->getLink()), dnLane(_dnLane) {
  int nodeId = _node->getId();

  double angle =
      Utility::angle(_upLane->getDirection(), _dnLane->getDirection());
  bool turn = (angle > 40. / 180. * M_PI && angle < 150. / 180. * M_PI);

  bool direct = true;  // 直接繋がる場合
  int upSign = 1;
  int dnSign = 1;
  if (nodeId == _upLane->getDNodeId()) {
    upSign = 1;
  } else {
    upSign = -1;
  }
  if (nodeId == _dnLane->getONodeId()) {
    dnSign = 1;
  } else {
    dnSign = -1;
  }
  if (upSign * dnSign < 0) {
    direct = false;
  }

  std::tie(ox, oy) =
      _upLane->getLink()->getLaneXY(nodeId, _upLane->getCenterOffset());
  std::tie(dx, dy) =
      _dnLane->getLink()->getLaneXY(nodeId, _dnLane->getCenterOffset());
  ix = 0.0;
  iy = 0.0;

  if (direct && !turn) {  // 直線
    r = 0.0;
    length = Utility::euclidean(ox, oy, dx, dy);
  } else if ((_upLane->getType() != "car" && _upLane->getType() != "bus") &&
             !turn) {
    r = 0.0;
    double dx1, dx2, dy1, dy2;
    std::tie(dx1, dy1) = _upLane->getLink()->getE();
    std::tie(dx2, dy2) = _dnLane->getLink()->getN();
    ix = (dx1 * dx2 * (dy - oy) - dx1 * dy2 * (dx - ox)) /
             (dy1 * dx2 - dx1 * dy2) +
         ox;
    iy = (ix - ox) * dy1 / dx1 + oy;
    length =
        Utility::euclidean(ox, oy, ix, iy) + Utility::euclidean(ix, iy, dx, dy);
  } else if ((_upLane->getType() != "car" && _upLane->getType() != "bus") &&
             direct) {
    r = 0.0;
    double dx1, dx2, dy1, dy2;
    std::tie(dx1, dy1) = _upLane->getLink()->getE();
    std::tie(dx2, dy2) = _dnLane->getLink()->getE();
    ix = (dx1 * dx2 * (dy - oy) - dx1 * dy2 * (dx - ox)) /
             (dy1 * dx2 - dx1 * dy2) +
         ox;
    iy = (ix - ox) * dy1 / dx1 + oy;
    length =
        Utility::euclidean(ox, oy, ix, iy) + Utility::euclidean(ix, iy, dx, dy);
  } else if ((_upLane->getType() != "car" && _upLane->getType() != "bus")) {
    r = 0.0;
    length = Utility::euclidean(ox, oy, dx, dy);
  } else {  // 車
    double dx1, dx2, dy1, dy2;
    std::tie(dx1, dy1) = _upLane->getLink()->getN();
    std::tie(dx2, dy2) = _dnLane->getLink()->getN();
    centerX = (dx1 * dx2 * (dy - oy) - dx1 * dy2 * (dx - ox)) /
                  (dy1 * dx2 - dx1 * dy2) +
              ox;
    centerY = (centerX - ox) * dy1 / dx1 + oy;
    r = Utility::euclidean(centerX, centerY, ox, oy);

    oTheta = Utility::direction(centerX, centerY, ox, oy);
    dTheta = Utility::direction(centerX, centerY, dx, dy);
    if (dTheta > oTheta + M_PI) {
      dTheta -= 2 * M_PI;
    } else if (dTheta < oTheta - M_PI) {
      dTheta += 2 * M_PI;
    }
    length = r * std::abs(dTheta - oTheta);
  }
}

std::vector<std::shared_ptr<NodeLane>> Lane::getDnLane(int nodeId) {
  if (type == "car" || type == "bus") {
    if (nodeId == getDNodeId()) {
      return dnLanes;
    } else {
      return std::vector<std::shared_ptr<NodeLane>>();
    }
  } else {
    if (nodeId == getDNodeId()) {
      return dnLanes;
    } else if (nodeId == getONodeId()) {
      return upLanes;
    } else {
      return std::vector<std::shared_ptr<NodeLane>>();
    }
  }
}

void NodeLane::addAgent(const std::shared_ptr<Agent> &agent) {
  agentMap.emplace(agent->getId(), agent);
}

double NodeLane::getUtility(int mode) {
  auto utilityCoefficients = dnLink.lock()->getUtilityCoefficients();
  double nodelaneUtil;
  if (mode == 3) {
    nodelaneUtil = utilityCoefficients.time * length / 0.5;
  } else {
    nodelaneUtil = utilityCoefficients.length * length;
  }
  return nodelaneUtil + dnLink.lock()->getUtility(mode);
}

int NodeLane::getNodeId() { return node.lock()->getId(); }

bool NodeLane::isSaturated() { return dnLane->isSaturated(); }

}  // namespace Hongo
