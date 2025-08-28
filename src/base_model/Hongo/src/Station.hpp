//
//  Station.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/19.
//

#ifndef Station_hpp
#define Station_hpp

#include <stdio.h>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Hongo {

class Agent;
class Lane;
class Station : public std::enable_shared_from_this<Station> {
  int id;
  std::unordered_set<int> modeSet;  // 5: train, 6:bus
  std::shared_ptr<Lane> lane;
  double x;  // oNodeからの距離

  std::unordered_map<int, std::shared_ptr<Agent>> agentMap;

 public:
  Station(int _id, const std::shared_ptr<Lane> &_lane, double _x,
          std::unordered_set<int> _modeSet)
      : id(_id), modeSet(_modeSet), lane(_lane), x(_x) {}

  void addAgent(const std::shared_ptr<Agent> &agent);
  std::unordered_map<int, std::shared_ptr<Agent>> &getAgentMap() {
    return agentMap;
  }

  int getId() { return id; }
  std::shared_ptr<Lane> getLane() { return lane; }
  double getX() { return x; }
  void setModeSet(std::unordered_set<int> _modeSet) { modeSet = _modeSet; }
  std::unordered_set<int> getModeSet() { return modeSet; }
};
}  // namespace Hongo

#endif /* Station_hpp */
