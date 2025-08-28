//
//  Dijkstra.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/21.
//

#ifndef Dijkstra_hpp
#define Dijkstra_hpp

#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Hongo {

class Link;
class Station;
struct PathNode {
  int nodeId;
  double cost;
  std::shared_ptr<Link> prevLink;
};
static inline bool operator>(const PathNode &a, const PathNode &b) {
  return a.cost > b.cost;
}
class Dijkstra : public std::enable_shared_from_this<Dijkstra> {
  std::unordered_map<int, std::vector<std::shared_ptr<Link>>>
      odLinkMap;  // key: odNodeID, value: 繋がっているリンクのlinkID
 public:
  Dijkstra() {}

  void setLinkMap(
      const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap);
  std::pair<double, std::vector<std::shared_ptr<Link>>> getShortestPath(
      int oNodeId, int dNodeId, int mode,
      const std::unordered_set<int> &cutLinks,
      bool bus = false) const;  // costType  0: distance, 1: duration
  std::vector<std::shared_ptr<Station>> getNearStations(int oNodeId);

  bool linkJudge(std::shared_ptr<Link> nextLink, std::shared_ptr<Link> tmpLink,
                 int nodeId, int mode, const std::unordered_set<int> &cutLinks,
                 bool bus = false) const;
  double getCost(std::shared_ptr<Link> link,
                 int costType) const;  // costType  0: distance, 1: duration
};
}  // namespace Hongo

#endif /* Dijkstra_hpp */
