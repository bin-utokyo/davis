//
//  Link.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/07.
//

#include "Link.hpp"

#include "Agent.hpp"
#include "Lane.hpp"
#include "Node.hpp"
#include "Station.hpp"
#include "Utility.hpp"

namespace Hongo {

std::unordered_map<std::string, double> defaultUtilityCoefficients = {
  {"time", -1.0},
  {"angle", -0.50},
  {"length", -0.01},
  {"turn", -0.3}
};
std::unordered_map<int, std::unordered_map<std::string, double>>
  Link::utilityCoefficientsMap = {
    {0, defaultUtilityCoefficients},
    {3, defaultUtilityCoefficients},
    {5, defaultUtilityCoefficients},
    {6, defaultUtilityCoefficients}};  // key: mode, value: utilityCoefficients. 0: car, 3: pedestrian, 5: train, 6: bus

    
Link::Link(int _id, const std::shared_ptr<Node> &_oNode,
           const std::shared_ptr<Node> &_dNode, double _velocity)
    : id(_id), velocity(_velocity / 3.6), oNode(_oNode), dNode(_dNode) {
  length = Utility::hubeny(_oNode->getLon(), _oNode->getLat(), _dNode->getLon(),
                           _dNode->getLat());
  direction = Utility::direction(_oNode->getX(), _oNode->getY(), _dNode->getX(),
                                 _dNode->getY());
  e = std::make_pair(std::cos(direction), std::sin(direction));
  n = std::make_pair(std::sin(direction), -std::cos(direction));
}

void Link::setLanes(int leftCarNum, int leftPedNum, int leftBeltNum,
                    int rightCarNum, int rightPedNum, int rightBeltNum) {
  leftLanes.clear();
  rightLanes.clear();
  for (int i = 1; i <= leftCarNum; i++) {
    leftLanes.emplace_back(
        std::make_shared<Lane>(i, shared_from_this(), "car",
                               carLaneWidth * ((double)i - 1.0), carLaneWidth));
  }
  for (int i = 1; i <= rightCarNum; i++) {
    rightLanes.emplace_back(std::make_shared<Lane>(
        -i, shared_from_this(), "car", -carLaneWidth * ((double)i - 1.0),
        carLaneWidth));
  }
  for (int i = leftCarNum + 1; i <= leftCarNum + leftPedNum; i++) {
    leftLanes.emplace_back(std::make_shared<Lane>(
        i, shared_from_this(), "ped",
        carLaneWidth * leftCarNum +
            pedLaneWidth * ((double)i - leftCarNum - 1.0),
        pedLaneWidth));
  }
  for (int i = rightCarNum + 1; i <= rightCarNum + rightPedNum; i++) {
    rightLanes.emplace_back(std::make_shared<Lane>(
        -i, shared_from_this(), "ped",
        -carLaneWidth * rightCarNum -
            pedLaneWidth * ((double)i - rightCarNum - 1.0),
        pedLaneWidth));
  }
  for (int i = leftCarNum + 1; i <= leftCarNum + leftBeltNum; i++) {
    leftLanes.emplace_back(std::make_shared<Lane>(
        i, shared_from_this(), "sidebelt",
        carLaneWidth * leftCarNum +
            beltLaneWidth * ((double)i - leftCarNum - 1.0),
        beltLaneWidth));
  }
  for (int i = rightCarNum + 1; i <= rightCarNum + rightBeltNum; i++) {
    rightLanes.emplace_back(std::make_shared<Lane>(
        -i, shared_from_this(), "sidebelt",
        -carLaneWidth * rightCarNum -
            beltLaneWidth * ((double)i - rightCarNum - 1.0),
        beltLaneWidth));
  }
  width = carLaneWidth * (leftCarNum + rightCarNum) +
          pedLaneWidth * (leftPedNum + rightPedNum) +
          beltLaneWidth * (leftBeltNum + rightBeltNum);
}

std::shared_ptr<Lane> Link::getLane(int laneId) {
  std::shared_ptr<Lane> lane = nullptr;
  if (laneId > 0 && laneId <= (int)leftLanes.size()) {
    lane = leftLanes.at(laneId - 1);
  } else if (laneId < 0 && -laneId <= rightLanes.size()) {
    lane = rightLanes.at(-laneId - 1);
  }
  return lane;
}

std::vector<std::shared_ptr<Lane>> Link::getLanes() {
  std::vector<std::shared_ptr<Lane>> lanes;
  for (const auto &tmpLane : leftLanes) {
    lanes.emplace_back(tmpLane);
  }
  for (const auto &tmpLane : rightLanes) {
    lanes.emplace_back(tmpLane);
  }
  return lanes;
}

int Link::getONodeId() { return oNode->getId(); }
int Link::getDNodeId() { return dNode->getId(); }
int Link::getAnotherNodeId(int nodeId) {
  if (nodeId != oNode->getId()) {
    return oNode->getId();
  } else {
    return dNode->getId();
  }
}
std::shared_ptr<Node> Link::getNode(int nodeId) {
  if (nodeId == oNode->getId()) {
    return oNode;
  } else if (nodeId == dNode->getId()) {
    return dNode;
  } else {
    return nullptr;
  }
}

std::vector<std::shared_ptr<Link>> Link::getDnLinks(int nodeId) {
  // node によって接続されている他のリンク
  if (nodeId == dNode->getId()) {
    return dnLinks;
  } else {
    return upLinks;
  }
}

void Link::addDnLink(int nodeId, const std::shared_ptr<Link> &dnLink) {
  if (nodeId == dNode->getId()) {
    if (std::find(dnLinks.begin(), dnLinks.end(), dnLink) == dnLinks.end()) {
      dnLinks.emplace_back(dnLink);
    }
  } else if (nodeId == oNode->getId()) {
    if (std::find(upLinks.begin(), upLinks.end(), dnLink) == upLinks.end()) {
      upLinks.emplace_back(dnLink);
    }
  }
}

void Link::setDnLinks(int nodeId, std::vector<std::shared_ptr<Link>> _dnLinks) {
  if (nodeId == dNode->getId()) {
    dnLinks = _dnLinks;
  } else if (nodeId == oNode->getId()) {
    upLinks = _dnLinks;
  }
}

bool Link::isPassable(int oNodeId, int mode, bool bus) {
  if (mode == -1) {  // アクティビティリンク
    return oNode->getId() == oNodeId;
  }
  if (mode == 3) {
    const auto lanes = getLanes(0);
    bool passable = false;
    for (const auto &lane : lanes) {
      passable = passable || lane->isPassable(oNodeId, mode);
    }
    return passable;
  } else if (mode == 0) {  // 車両
    int sign;
    if (oNode->getId() == oNodeId) {
      sign = 1;
    } else {
      sign = -1;
    }
    const auto lanes = getLanes(sign);
    bool passable = false;
    for (const auto &lane : lanes) {
      passable = passable || lane->isPassable(oNodeId, mode, bus);
    }
    return passable;
  } else if (mode == 6) {  // バス利用のエージェント
    return oNodeId == oNode->getId();
  }
  return false;
}

bool Link::isConnected(int nodeId, const std::shared_ptr<Link> &dnLink,
                       int mode) {
  if (mode == -1) {
    return true;
  }
  if (mode == 3) {  // 歩行者
    for (const auto &lane : getLanes(0)) {
      for (const auto &dl : lane->getDnLane(nodeId)) {
        if (dl->getDnLink()->getId() == dnLink->getId()) {
          return true;
        }
      }
    }
  }
  if (mode == 0) {
    int sign = 1;
    if (dNode->getId() != nodeId) {
      sign = -1;
    }
    for (const auto &lane : getLanes(sign)) {
      for (const auto &dl : lane->getDnLane(nodeId)) {
        if (dl->getDnLink()->getId() == dnLink->getId()) {
          return true;
        }
      }
    }
  }
  return false;
}

double Link::getUtility(int mode) {
  if (mode == -1) {  // アクティビティリンクとして用いる場合
    return activityUtility;
  }

  if (mode == 6) {
    return travelTime;
  } else {
    // timePed and timeVeh are calculated dynamically
    // Calculate utility for dynamic properties
    double utility = 0.0;
    if (utilityCoefficientsMap.at(mode).find("time") != utilityCoefficientsMap.at(mode).end()) {
      utility += utilityCoefficientsMap.at(mode).at("time") * length /
                 std::max(0.1, getVelocity(mode)) / 100;
    }
    if (utilityCoefficientsMap.at(mode).find("velocity") !=
        utilityCoefficientsMap.at(mode).end()) {
      utility += utilityCoefficientsMap.at(mode).at("velocity") * getVelocity(mode) / 10;
    } 
    // Calculate utility for static properties
    for (const auto &pair : utilityCoefficientsMap.at(mode)) {
      if (pair.first == "time") {
        continue;
      }
      if (utilityCoefficientsMap.at(mode).find(pair.first) !=
          utilityCoefficientsMap.at(mode).end()) {
            if (properties.find(pair.first) != properties.end()) {
              double coeff = utilityCoefficientsMap.at(mode).at(pair.first);
              double val = properties.at(pair.first);
              utility += coeff * val;
            }
      } 
    }
    return utility;
  }
}

double Link::getConnectionUtility(int dnLinkId, int nodeId, int mode) {
  // リンク接続の部分での効用
  if (mode == -1) {  // アクティビティリンク
    return 0.0;
  }
  double utility = -std::numeric_limits<double>::infinity();
  std::shared_ptr<Link> dnLink;
  if (nodeId == dNode->getId()) {
    for (const auto &dLink : dnLinks) {
      if (dLink->getId() == dnLinkId) {
        dnLink = dLink;
        break;
      }
    }
  } else if (nodeId == oNode->getId()) {
    for (const auto &dLink : upLinks) {
      if (dLink->getId() == dnLinkId) {
        dnLink = dLink;
        break;
      }
    }
  }
  // angleベース（rad）の場合
  if (dnLink != nullptr) {
    double angle = Utility::angle(dnLink->getDirection(), direction);
    if (utilityCoefficientsMap.at(mode).find("angle") !=
        utilityCoefficientsMap.at(mode).end()) {
      utility = utilityCoefficientsMap.at(mode).at("angle") * angle;
    } else {
      utility = 0.0;
    }
  }
  // if (dnLink != nullptr){
  //     if (Link::turnJudge(direction, dnLink->getDirection())){
  //         utility = utilityCoefficients.turn;
  //     }else{
  //         utility = 0.0;
  //     }
  // }
  return utility;
}

std::vector<std::shared_ptr<Lane>> Link::getLanes(int sign) {
  std::vector<std::shared_ptr<Lane>> lanes;
  for (const auto &tmpLane : leftLanes) {
    if (tmpLane->getSign() == sign) {
      lanes.emplace_back(tmpLane);
    }
  }
  for (const auto &tmpLane : rightLanes) {
    if (tmpLane->getSign() == sign) {
      lanes.emplace_back(tmpLane);
    }
  }
  return lanes;
}

void Link::addAgent(const std::shared_ptr<Agent> &agent, double simTime) {
  agentInTime[agent->getId()] = simTime;
  agentMap[agent->getId()] = agent;
}
void Link::removeAgent(int agentId, double simTime) {
  std::shared_ptr<Agent> agent = agentMap.at(agentId);
  int mode = agent->getMode();
  if (mode == 0) {                        // 車
    if (agent->getLane()->getId() > 0) {  // 左側レーン
      carAccumTravelTime.first +=
          std::max(0.0, simTime - agentInTime.at(agentId));
      if (simTime - agentInTime.at(agentId) > 0) carTrafficVolume.first++;
    } else {  // 右側レーン
      carAccumTravelTime.second +=
          std::max(0.0, simTime - agentInTime.at(agentId));
      if (simTime - agentInTime.at(agentId) > 0) carTrafficVolume.second++;
    }
  } else if (mode == 3) {  // 歩行者
    accumTravelTime += std::max(0.0, simTime - agentInTime.at(agentId));
    if (simTime - agentInTime.at(agentId) > 0) trafficVolume++;
  }
  agentInTime.erase(agentId);
  agentMap.erase(agentId);
}

void Link::addStation(const std::shared_ptr<Station> &station) {
  stationMap[station->getId()] = station;
}
std::unordered_map<int, std::shared_ptr<Station>> Link::getStationMap() {
  return stationMap;
}

double Link::getOffset(int nodeId) {
  if (nodeId == oNode->getId()) {
    return oNode->getOffset(id);
  } else if (nodeId == dNode->getId()) {
    return dNode->getOffset(id);
  } else {
    return 0.0;
  }
}

double Link::getEffectiveLength() {
  return std::max(0.0, length - (oNode->getOffset(id) + dNode->getOffset(id)));
}

double Link::getVelocity(int mode) {
  if (mode == 0) {  // 車
    int carTraffic = carTrafficVolume.first + carTrafficVolume.second;
    if (carTraffic > 0) {
      return getEffectiveLength() /
             ((carAccumTravelTime.first + carAccumTravelTime.second) /
              carTraffic);
    }
  } else if (mode == 3) {  // 歩行者
    if (trafficVolume > 0) {
      return getEffectiveLength() / (accumTravelTime / trafficVolume);
    }
  }
  return velocity;
}

std::pair<double, double> Link::getCarVelocities() {
  double f = 0;
  double s = 0;
  if (carTrafficVolume.first > 0) {
    f = getEffectiveLength() / carAccumTravelTime.first *
        carTrafficVolume.first;
  }
  if (carTrafficVolume.second > 0) {
    s = getEffectiveLength() / carAccumTravelTime.second *
        carTrafficVolume.second;
  }
  return std::make_pair(f, s);
}

std::pair<double, double> Link::getLonLat(int nodeId) {
  double lon = 0.0;
  double lat = 0.0;
  if (nodeId == oNode->getId()) {
    lon = (oNode->getLon() * (length - getOffset(nodeId)) +
           dNode->getLon() * getOffset(nodeId)) /
          length;
    lat = (oNode->getLat() * (length - getOffset(nodeId)) +
           dNode->getLat() * getOffset(nodeId)) /
          length;
  } else if (nodeId == dNode->getId()) {
    lon = (dNode->getLon() * (length - getOffset(nodeId)) +
           oNode->getLon() * getOffset(nodeId)) /
          length;
    lat = (dNode->getLat() * (length - getOffset(nodeId)) +
           oNode->getLat() * getOffset(nodeId)) /
          length;
  }
  return std::make_pair(lon, lat);
}

std::pair<double, double> Link::getXY(int nodeId) {
  double x = 0.0;
  double y = 0.0;
  if (nodeId == oNode->getId()) {
    x = oNode->getX();
    y = oNode->getY();
  } else if (nodeId == dNode->getId()) {
    x = dNode->getX();
    y = dNode->getY();
  }
  return std::make_pair(x, y);
}

std::pair<double, double> Link::getLaneLonLat(int nodeId, double centerOffset) {
  double x, y;
  std::tie(x, y) = getLaneXY(nodeId, centerOffset);
  return Utility::xy2LonLat(x, y);
}

std::pair<double, double> Link::getLaneXY(int nodeId, double centerOffset) {
  double x, y;
  std::tie(x, y) = getXY(nodeId);
  x += n.first * centerOffset;
  y += n.second * centerOffset;
  if (nodeId == oNode->getId()) {
    x += e.first * oNode->getOffset(id);
    y += e.second * oNode->getOffset(id);
  } else if (nodeId == dNode->getId()) {
    x -= e.first * dNode->getOffset(id);
    y -= e.second * dNode->getOffset(id);
  } else {
    return std::make_pair(0.0, 0.0);
  }
  return std::make_pair(x, y);
}
}  // namespace Hongo
