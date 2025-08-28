//
//  Link.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/07.
//

#ifndef Link_hpp
#define Link_hpp

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Hongo {

class Node;
class Lane;
class NodeLane;
class Agent;
class Station;

struct LinkUtility {
  double timePed = -1.157 * (4.5 / 3.6);
  double timeVeh = -0.356 * (30 / 3.6);
  double anglePed = -0.506;
  double angleVeh = -0.640;
  double koutuPed = 0.024;
  double koutuVeh = -0.045;
  double syogyoPed = 0.289;
  double syogyoVeh = -0.131;
  double crossPed = 0.273;
  double crossVeh = 0.730;

  double time = -0.01;
  double length = -0.01;
  double turn = -0.3;
  // 魅力度（店舗数、口コミなど）
};


class Link : public std::enable_shared_from_this<Link> {
  const double carLaneWidth = 3.0;   // m
  const double pedLaneWidth = 2.0;   // m
  const double beltLaneWidth = 1.5;  // m
  int id;
  double length;                // m
  double velocity;              // m/s for car
  double travelTime = 0;        // for bus
  double width;                 // m 上下両方向合わせた道幅
  std::pair<double, double> e;  //(deltax,deltay) dx^2+dy^2=1m^2
  std::pair<double, double> n;  //(-deltay, deltax)
  double direction;             // 0~2*pi

  std::unordered_map<std::string, double> properties;

  LinkUtility utilityCoefficients;
  static std::unordered_map<int, std::unordered_map<std::string, double>> utilityCoefficientsMap;
  double activityUtility;

  std::vector<std::shared_ptr<Link>> dnLinks;
  std::vector<std::shared_ptr<Link>> upLinks;
  std::vector<std::shared_ptr<Lane>> leftLanes,
      rightLanes;  // 中央線側のレーンから順に
  std::shared_ptr<Node> oNode, dNode;

  std::vector<std::shared_ptr<Link>>
      priorityLinks;  // 通行の際に現リンクよりも優先される交差リンク

  std::unordered_map<int, double>
      agentInTime;  // key: agentID, value:
                    // agentがlinkに侵入した時間(SimulationTime)
  std::unordered_map<int, std::shared_ptr<Agent>>
      agentMap;  // key: agentID, value: agent

  std::unordered_map<int, std::shared_ptr<Station>>
      stationMap;  // key: stationID, value: station

  // 旅行時間計算用
  int trafficVolume = 0;         // 交通量（歩行者）
  double accumTravelTime = 0.0;  // 累積の旅行時間（歩行者）
  std::pair<int, int> carTrafficVolume = std::make_pair(0, 0);
  std::pair<double, double> carAccumTravelTime = std::make_pair(0.0, 0.0);

 public:
  Link(int _id, const std::shared_ptr<Node> &_oNode,
       const std::shared_ptr<Node> &_dNode, double _velocity);
  Link(int _id, const std::shared_ptr<Node> &_oNode,
       const std::shared_ptr<Node> &_dNode)
      : id(_id), oNode(_oNode), dNode(_dNode) {}

  void setLanes(int leftCarNum, int leftPedNum, int leftBeltNum,
                int rightCarNum, int rightPedNum, int rightBeltNum);
  std::shared_ptr<Lane> getLane(int laneId);
  std::vector<std::shared_ptr<Lane>> getLanes();

  int getONodeId();
  int getDNodeId();
  std::shared_ptr<Node> getNode(int nodeId);
  int getAnotherNodeId(int nodeId);
  std::vector<std::shared_ptr<Link>> getDnLinks(int nodeId);
  void addDnLink(int nodeId, const std::shared_ptr<Link> &dnLink);
  void setDnLinks(int nodeId, std::vector<std::shared_ptr<Link>> _dnLinks);
  bool isPassable(int oNodeId, int mode, bool bus = false);
  bool isConnected(int nodeId, const std::shared_ptr<Link> &dnLink, int mode);
  double getUtility(int mode);
  double getConnectionUtility(int dnLinkId, int nodeId,
                              int mode);  // リンク接続の部分での効用
  std::vector<std::shared_ptr<Lane>> getLanes(int sign);
  void addAgent(const std::shared_ptr<Agent> &agent, double simTime);
  void removeAgent(int agentId, double simTime);
  void addStation(const std::shared_ptr<Station> &station);
  std::unordered_map<int, std::shared_ptr<Station>> getStationMap();
  double getOffset(int nodeId);
  double getEffectiveLength();
  double getVelocity(int mode);
  std::pair<double, double>
  getCarVelocities();  // first: left_velocity(m/s), second: right_velocity(m/s)
  std::pair<double, double> getLonLat(int nodeId);
  std::pair<double, double> getXY(int nodeId);
  std::pair<double, double> getLaneLonLat(int nodeId, double centerOffset);
  std::pair<double, double> getLaneXY(int nodeId, double centerOffset);

  int getId() { return id; }
  double getWidth() { return width; }
  double getLength() { return length; }
  void setPriorityLinks(
      const std::vector<std::shared_ptr<Link>> &_priorityLinks) {
    priorityLinks = _priorityLinks;
  }
  void addPriorityLink(const std::shared_ptr<Link> &plink) {
    priorityLinks.emplace_back(plink);
  }
  std::vector<std::shared_ptr<Link>> getPriorityLinks() {
    return priorityLinks;
  }
  void setTravelTime(double _travelTime) { travelTime = _travelTime; }
  double getTravelTime() {
    // 制限速度で移動した時の旅行時間
    if (travelTime > 0) {
      return travelTime;
    } else {
      return length / velocity;
    }
  }
  std::pair<double, double> getN() { return n; }
  std::pair<double, double> getE() { return e; }
  double getSpeedLimit() { return velocity; }
  int getTrafficVolume() {
    // 歩行者の交通量
    return trafficVolume;
  }

  std::pair<int, int> getCarTrafficVolume() { return carTrafficVolume; }
  void clearVelocity() {
    accumTravelTime = 0;
    trafficVolume = 0;
    carAccumTravelTime = std::make_pair(0.0, 0.0);
    carTrafficVolume = std::make_pair(0, 0);
  }
  double getDirection() { return direction; }

  void setActUtility(double _activityUtility) {
    activityUtility = _activityUtility;
  }

  void setUtilityCoefficient(int mode, const std::string &propName,
                             double coeff) {
    utilityCoefficientsMap[mode][propName] = coeff;
  }
  
  const LinkUtility &getUtilityCoefficients() { return utilityCoefficients; }

  void setProperties(std::unordered_map<std::string, double> _properties) {
    properties = _properties;
  }

  std::unordered_map<std::string, double> getProperties() { return properties; }

  static bool turnJudge(double oDirection, double dDirection) {
    int angle = (int)((dDirection - oDirection) / M_PI * 180.0) % 360;
    if (angle < 40 || angle > 320) {
      return false;
    } else {
      return true;
    }
  }
};
}  // namespace Hongo

#endif /* Link_hpp */
