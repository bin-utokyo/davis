//
//  ChoiceSet.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/18.
//

#ifndef ChoiceSet_hpp
#define ChoiceSet_hpp

#include <stdio.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Hongo {

class Node;
class Link;
class NodeLane;
class Lane;
class Dijkstra;
class Physarum;
class ChoiceSet : public std::enable_shared_from_this<ChoiceSet> {
  int k = 3;                      // 探索経路数
  std::string type = "dijkstra";  // dijkstra, physarum

  std::unordered_map<int, std::shared_ptr<Link>> linkMap;
  std::shared_ptr<Dijkstra> dijkstra;
  std::shared_ptr<Physarum> physarum;

 public:
  ChoiceSet(const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap,
            std::string _type);  // type  dijkstra, physarum
  std::unordered_set<int> generateChoiceSet_dijkstra(int oNodeId, int dNodeId,
                                                     int mode,
                                                     bool bus = false);
  std::unordered_set<int> generateChoiceSet_physarum(int oNodeId, int dNodeId,
                                                     int mode);

  std::unordered_set<int> generateChoiceSet_dijkstra(
      const std::shared_ptr<Lane> &tmpLane, int dNodeId, int mode,
      bool bus = false);
  std::unordered_set<int> generateChoiceSet_dijkstra(
      int oNodeId, const std::shared_ptr<Lane> &nextLane, int mode,
      bool bus = false);
  std::unordered_set<int> generateChoiceSet_dijkstra(
      const std::shared_ptr<Lane> &tmpLane,
      const std::shared_ptr<Lane> &nextLane, int mode, bool bus = false);

  std::unordered_map<int, std::shared_ptr<Link>> generateChoiceSet(
      int oNodeId, int dNodeId, int mode, bool bus = false);

  std::unordered_map<int, std::shared_ptr<Link>> generateChoiceSet(
      const std::shared_ptr<Lane> &tmpLane, int dNodeId, int mode,
      bool bus = false);  // 公共交通など
  std::unordered_map<int, std::shared_ptr<Link>> generateChoiceSet(
      int oNodeId, const std::shared_ptr<Lane> &nextLane, int mode,
      bool bus = false);  // 公共交通など
  std::unordered_map<int, std::shared_ptr<Link>> generateChoiceSet(
      const std::shared_ptr<Lane> &tmpLane,
      const std::shared_ptr<Lane> &nextLane, int mode,
      bool bus = false);  // 公共交通など

  std::pair<double, std::vector<std::shared_ptr<Link>>> getShortestPath(
      int oNodeId, int dNodeId, int mode, bool bus = false);

  void resetDijkstra();
  std::shared_ptr<Dijkstra> getDijkstra() { return dijkstra; }
  void setK(int _k) { k = _k; }
};
}  // namespace Hongo

#endif /* ChoiceSet_hpp */
