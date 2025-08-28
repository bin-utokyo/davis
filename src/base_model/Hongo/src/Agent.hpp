//
//  Agent.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/08.
//

#ifndef Agent_hpp
#define Agent_hpp

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace Hongo {

class Link;
class Lane;
class NodeLane;
class Node;
class Pedestrian;
class Vehicle;
class Bus;
class Station;
class RL;
struct Position {
  // Agentの位置(x,y)と速度(dx,dy),加速度(ddx,ddy)
  double x = 0.0;
  double y = 0.0;
  double dx = 0.0;
  double dy = 0.0;  // linkのoNodeからdNodeに向かう方向がx,
                    // 中央線から外側に向かう方向がy
  double ddx = 0.0;
  double ddy = 0.0;
  double endx, endy;  // agentが次のnodeLaneに移動する終点、道路の端

  static double distance(const Position &p1, const Position &p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
  }
  static double velocity(const Position &p) {
    return std::sqrt(p.dx * p.dx + p.dy * p.dy);
  }
  static double acceleration(const Position &p) {
    return std::sqrt(p.ddx * p.ddx + p.ddy * p.ddy);
  }
};
struct Attribute {
  // 個人属性
  int age;
  int sex;  // 男性: 0, 女性: 1
};
struct MigrationCoefficient {
  // 回遊モデルの係数
  // 各ノードにそれぞれの項目に対応する属性（店舗がいくつあるかなど）を与えて目的地選択に用いる。
  double time = 0.0;
  double spa = 1.44;
  double shop = 3.38;
  double famous = 2.74;  // 「名所」
  double art = 1.80;     //
};
class MigrationAttribute
    : public std::enable_shared_from_this<MigrationAttribute> {
  // ノードごとの属性
  int startTime, endTime;
  std::shared_ptr<Node> node;

  double spa;
  double shop;
  double famous;
  double art;

 public:
  MigrationAttribute(const std::shared_ptr<Node> &_node, int _startTime,
                     int _endTime, double _spa, double _shop, double _famous,
                     double _art)
      : startTime(_startTime),
        endTime(_endTime),
        node(_node),
        spa(_spa),
        shop(_shop),
        famous(_famous),
        art(_art) {}

  double getUtility(const std::shared_ptr<Node> &currentNode, int mode,
                    int currentTime, const MigrationCoefficient &coeff);
  double getUtility(int currentTime, const MigrationCoefficient &coeff);

  std::shared_ptr<Node> getNode() { return node; }
  double getTime(double distance, int mode) {
    double velocity;  // m/s
    if (mode == 3) {  // 歩行者
      velocity = 1.0;
    } else {
      velocity = 8.3;  // 時速30km
    }
    return distance / velocity;
  }
};
class Agent : public std::enable_shared_from_this<Agent> {
  const static bool ecursion;  // 回遊するか否か

  int id;
  int mode;

  int oNodeId;
  int dNodeId;
  int tmpDNodeId;  // 次による地点

  int startTime;
  int endTime;
  double simTime;

  double accumUtility = 0;       // 効用の和
  double accumTravelLength = 0;  // 移動距離

  int stayStep = 0;  // 滞在の状態のタイムステップ数
  double destinationDistance = std::numeric_limits<double>::infinity();

  std::shared_ptr<Pedestrian> pedestrian;
  std::shared_ptr<Vehicle> vehicle;
  std::shared_ptr<Bus> bus;

  std::shared_ptr<RL> upperRL;
  std::shared_ptr<RL> lowerRL;
  std::shared_ptr<RL> transRL;  // 公共交通を利用するエージェント

  std::unordered_map<int, int>
      actNodeIdMap;  // key: activityLinkID, value: nodeID
  int tmpActNodeId;
  std::shared_ptr<Link> actLink;

  std::shared_ptr<Link> link;
  std::shared_ptr<Lane> lane;
  std::shared_ptr<NodeLane> nodeLane;

  Position position;  // ローカル座標
  Position prevPosition;

  Attribute attribute;  // 属性
  MigrationCoefficient migrationCoefficient;
  std::unordered_map<int, std::shared_ptr<MigrationAttribute>> facilityMap;

  std::string purpose;  // 移動目的  commute, business, purchace, joy 目的地選択
  bool changeDestination = false;  // 目的地を変えた場合trueに変える。

  int direction;  // linkのoNodeからdNodeへ進んでいる場合: 1, そうでない場合: -1
  bool transportation = false;  // 公共交通の車両か否か

  bool bord = false;                  // 公共交通に乗車中か否か
  std::shared_ptr<Station> oStation;  // 現在地からの最寄駅

 public:
  const static int activityInterval =
      60 * 60;  // 1時間ごとに1アクティビティとする。

  Agent(int _id, int _startTime, int _endTime, std::string _purpose, int _mode,
        const std::shared_ptr<Node> &oNode, const std::shared_ptr<Node> &dNode,
        const std::unordered_map<int, std::shared_ptr<MigrationAttribute>>
            &_facilityMap);
  Agent(int _id, int _startTime, int _mode, const std::shared_ptr<Node> &oNode,
        const std::shared_ptr<Node> &dNode);  // 公共交通など用

  void initializeLowerRL(
      const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap);
  void initializeTransRL(
      const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap);
  void setLowerRL(double simTime);  // 次の目的地をセットする
  void chooseDestination();
  void chooseDestinationTransport();       // 公共交通
  void setDestinationTransport();          // 公共交通イグレス
  std::shared_ptr<NodeLane> planDnLane();  // dnNodeLane
  void setDnLane(const std::shared_ptr<NodeLane> &dnNodeLane);
  int getNextStation(int tmpStationId);  // 公共交通
  std::shared_ptr<Lane> planLaneChange();
  int chooseStayTime(int timestep, double simTime);

  void setPrevPosition();
  void updateVelocity(double timestep);
  void updateAcceleration(double timestep, double simTime);
  std::shared_ptr<Agent> moveAgent(double timestep, double simTime);
  std::shared_ptr<MigrationAttribute> getDestinationAttribute();
  bool isStop();
  bool isCarFollwing();
  bool isSignalStop();
  std::pair<double, double> getGlobalCoord();
  Position getGlobalPosition();
  bool isLastTrip();
  double getLength();

  int getId() { return id; }
  int getMode() { return mode; }
  std::string getPurpose() { return purpose; }
  int getStartTime() { return startTime; }

  void setSimTime(double _simTime) { simTime = _simTime; }
  double getSimTime() { return simTime; }
  double getDestinationDistance() { return destinationDistance; }
  bool destinationChanged() {
    if (changeDestination) {
      changeDestination = false;
      return true;
    }
    return false;
  }
  double getVelocity() { return Position::velocity(position); }
  double getAcceleration() { return Position::acceleration(position); }
  void setDirection(int _direction) { direction = _direction; }
  int getDirection() { return direction; }
  void setLink(std::shared_ptr<Link> _link) { link = _link; }
  std::shared_ptr<Link> getLink() { return link; }
  const MigrationCoefficient &getMigrationCoefficient() {
    return migrationCoefficient;
  }
  void setLane(std::shared_ptr<Lane> _lane) { lane = _lane; }
  std::shared_ptr<Lane> getLane() { return lane; }
  std::shared_ptr<NodeLane> getNodeLane() { return nodeLane; }
  bool inNodeLane() { return nodeLane != nullptr; }
  void offNodeLane() { nodeLane = nullptr; }
  Position &getPosition() { return position; }
  Position &getPrevPosition() { return prevPosition; }
  void setONodeId(int _oNodeId) { oNodeId = _oNodeId; }
  int getONodeId() { return oNodeId; }
  int getTmpDNodeId() { return tmpDNodeId; }
  void setNextDNodeId(int _tmpDNodeId) { tmpDNodeId = _tmpDNodeId; }
  std::pair<int, int> getOD() { return std::make_pair(oNodeId, tmpDNodeId); }
  int getDNodeId() { return dNodeId; }
  bool isStay() {
    if (stayStep > 0) {
      stayStep--;
      return true;
    } else {
      return false;
    }
  }
  bool isTransportation() { return transportation; }
  bool isOnBord() { return bord; }
  void offBord() { bord = false; }
  int getStayStep() { return stayStep; }
  void setBus(std::shared_ptr<Bus> _bus) { bus = _bus; }
  std::shared_ptr<Bus> getBus() { return bus; }

  void addTravelLength(double travelLength) {
    accumTravelLength += travelLength;
  }
  void addUtility(double utility) { accumUtility += utility; }
  double getTravelLength() { return accumTravelLength; }
  double getUtility() { return accumUtility; }

  void setLowerRL(const std::shared_ptr<RL> &_lowerRL) { lowerRL = _lowerRL; }
  void clearLane() {
    lane = nullptr;
    link = nullptr;
    nodeLane = nullptr;
    direction = 0;
  }
};
}  // namespace Hongo

#endif /* Agent_hpp */
