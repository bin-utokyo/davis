//
//  Lane.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/29.
//

#ifndef Lane_hpp
#define Lane_hpp

#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "Utility.hpp"

namespace Hongo {

class Link;
class NodeLane;
class Agent;
class Node;
class Lane : public std::enable_shared_from_this<Lane> {
  int laneId;           // 左：1,...,n、右：-1,...,-n
  double centerOffset;  // m 左:0以上、右:0以下
  double width;         // m
  double direction;     // 0~2pi
  std::string type;     // ped: 歩道、car: 車道, sidebelt: 路側帯, bus:
                        // バス専用レーン, rail: 鉄道

  std::vector<std::shared_ptr<NodeLane>> dnLanes;
  std::vector<std::shared_ptr<NodeLane>> upLanes;  // 歩道、路側帯のみ
  std::weak_ptr<Link> link;
  std::unordered_map<int, std::shared_ptr<Agent>> agentMap;

 public:
  Lane(int _laneId, const std::shared_ptr<Link> &_link, std::string _type,
       double _centerOffset, double _width);

  void addDnLane(int nodeId, const std::shared_ptr<Lane> &dnLane);
  std::vector<std::shared_ptr<NodeLane>> getDnLane(
      int nodeId);  // laneの下流側ノードのID
  void addAgent(const std::shared_ptr<Agent> &agent, double simTime);
  void addAgent(const std::shared_ptr<Agent>
                    &agent);  // 車線変更時（リンクにはagentを加えない）
  void removeAgent(int agentId, double simTime);
  void removeAgent(int agentId);  // 車線変更時（リンクからはagentを除かない）
  int getONodeId();
  int getDNodeId();
  bool isPassable(int oNodeId, int mode, bool bus = false);
  bool isConnected(int nodeId, const std::shared_ptr<Link> &dnLink);
  bool isSaturated();
  std::pair<double, double> getXYinLane(double x, double y);
  std::pair<double, double> getLonLatinLane(double x, double y);

  int getId() { return laneId; }
  std::string getType() { return type; }
  void setType(std::string _type) { type = _type; }
  double getDirection() { return direction; }
  double getCenterOffset() { return centerOffset; }
  std::shared_ptr<Link> getLink() { return link.lock(); }
  std::vector<std::shared_ptr<NodeLane>> getDnLane() { return dnLanes; }
  std::unordered_map<int, std::shared_ptr<Agent>> getAgentMap() {
    return agentMap;
  }
  double getWidth() { return width; }
  int getSign() {  // 進行可能方向（linkのoNode->dNodeが正）
    if (type == "car" || type == "bus") {
      return laneId / std::abs(laneId);
    } else {
      return 0;  // 両方向進行可
    }
  }
};

class NodeLane : public std::enable_shared_from_this<NodeLane> {
  std::weak_ptr<Node> node;
  double length;  // m
  double centerX, centerY, r, oTheta,
      dTheta;  // 弧の中心座標、弧の半径(m)
               // r=0は直線、始点の角度(0~2pi)、終点の角度(-pi~3pi,
               // dTheta>oTheta: 右折、dTheta<oTheta: 左折)
  double ox, oy, dx, dy, ix,
      iy;  // r=0の時, 平面直交座標系での起終点、経由点の座標値

  std::vector<std::weak_ptr<NodeLane>> crossLanes;
  std::unordered_map<int, std::shared_ptr<Agent>> agentMap;

  std::weak_ptr<Link> dnLink;
  std::shared_ptr<Lane> dnLane;

 public:
  NodeLane(const std::shared_ptr<Node> &_node,
           const std::shared_ptr<Lane> &_upLane,
           const std::shared_ptr<Lane> &_dnLane);

  void addAgent(const std::shared_ptr<Agent> &agent);
  double getUtility(int mode);
  int getNodeId();
  bool isSaturated();

  static bool crossing(const std::shared_ptr<NodeLane> &nodeLane1,
                       const std::shared_ptr<NodeLane> &nodeLane2) {
    double ox1, oy1, dx1, dy1, ox2, oy2, dx2, dy2;
    if (nodeLane1->getR() == 0) {
      std::tie(ox1, oy1) = nodeLane1->getOXY();
      std::tie(dx1, dy1) = nodeLane1->getDXY();
    } else {
      double cx, cy, oTheta, dTheta, r;
      std::tie(cx, cy) = nodeLane1->getCenterXY();
      std::tie(oTheta, dTheta) = nodeLane1->getTheta();
      r = nodeLane1->getR();
      ox1 = cx + r * std::cos(oTheta);
      oy1 = cy + r * std::sin(oTheta);
      dx1 = cx + r * std::cos(dTheta);
      dy1 = cy + r * std::sin(dTheta);
    }
    if (nodeLane2->getR() == 0) {
      std::tie(ox2, oy2) = nodeLane2->getOXY();
      std::tie(dx2, dy2) = nodeLane2->getDXY();
    } else {
      double cx, cy, oTheta, dTheta, r;
      std::tie(cx, cy) = nodeLane2->getCenterXY();
      std::tie(oTheta, dTheta) = nodeLane2->getTheta();
      r = nodeLane2->getR();
      ox2 = cx + r * std::cos(oTheta);
      oy2 = cy + r * std::sin(oTheta);
      dx2 = cx + r * std::cos(dTheta);
      dy2 = cy + r * std::sin(dTheta);
    }
    double dir1 = Utility::direction(ox1, oy1, ox2, oy2);
    double dir2 = Utility::direction(dx1, dy1, dx2, dy2);
    return std::cos(Utility::angle(dir1, dir2)) < 0;
  }

  std::shared_ptr<Node> getNode() { return node.lock(); }
  std::shared_ptr<Link> getDnLink() { return dnLink.lock(); }
  std::shared_ptr<Lane> getDnLane() { return dnLane; }
  double getLength() { return length; }
  double getR() { return r; }
  std::pair<double, double> getOXY() { return std::make_pair(ox, oy); }
  std::pair<double, double> getDXY() { return std::make_pair(dx, dy); }
  std::pair<double, double> getIXY() { return std::make_pair(ix, iy); }
  std::pair<double, double> getCenterXY() {
    return std::make_pair(centerX, centerY);
  }
  std::pair<double, double> getTheta() {
    return std::make_pair(oTheta, dTheta);
  }
  std::unordered_map<int, std::shared_ptr<Agent>> getAgentMap() {
    return agentMap;
  }
  void removeAgent(int agentId) { agentMap.erase(agentId); }
};

class Plaza : public std::enable_shared_from_this<Plaza> {
  int id;
};
}  // namespace Hongo

#endif /* Lane_hpp */
