//
//  Node.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/07.
//

#ifndef Node_hpp
#define Node_hpp

#include <stdio.h>

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Hongo {

class NodeLane;
class Link;
class Signal;
class Node : public std::enable_shared_from_this<Node> {
  int id;
  double lon, lat, x, y;
  std::unordered_map<int, double>
      offset;  // key: linkID, value: 交差点内に入っている部分の長さ(m)
  std::map<double, std::shared_ptr<Link>>
      linkDirection;  // key: nodeから出る方向, value: link

  std::vector<std::shared_ptr<NodeLane>> nodeLanes;
  std::shared_ptr<Signal> signal;

 public:
  Node(int _id, double _lon, double _lat);
  Node(int _id, double _lon, double _lat, double _x, double _y);

  void addLink(const std::shared_ptr<Link> &link);
  void setOffset();

  int getId() { return id; }
  void addNodeLane(const std::shared_ptr<NodeLane> &_nodeLane) {
    nodeLanes.emplace_back(_nodeLane);
  }
  const std::vector<std::shared_ptr<NodeLane>> &getNodeLanes() {
    return nodeLanes;
  }
  void setSignal(std::shared_ptr<Signal> _signal) { signal = _signal; }
  const std::shared_ptr<Signal> &getSignal() { return signal; }
  double getLon() { return lon; }
  double getLat() { return lat; }
  double getX() { return x; }
  double getY() { return y; }
  double getOffset(int linkId) {
    if (offset.find(linkId) != offset.end()) {
      return offset.at(linkId);
    } else {
      return 0.0;
    }
  }
  const std::map<double, std::shared_ptr<Link>> &getLinkDirection() {
    return linkDirection;
  }
};
}  // namespace Hongo

#endif /* Node_hpp */
