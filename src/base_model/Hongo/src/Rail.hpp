//
//  Rail.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/19.
//

#ifndef Rail_hpp
#define Rail_hpp

#include <stdio.h>

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Hongo {

class Agent;
class Station;
class Link;
class RL;
class Rail : public std::enable_shared_from_this<Rail> {
  std::weak_ptr<Agent> agent;
  std::shared_ptr<RL> lowerRL;  // agentと同じもの

  int routeId;
  int nextStationNum = -1;
  std::map<double, std::shared_ptr<Station>>
      table;                     // key: 出発時刻, value: station
  std::vector<double> depTimes;  // 出発時刻、昇順

  std::unordered_map<int, std::shared_ptr<Agent>> agentMap;

  int stopStep = 0;  // 停止するタイムステップ数
 public:
  Rail(const std::shared_ptr<Agent> &_agent, int _routeId);
  void insertTimetable(double depTime, const std::shared_ptr<Station> &station);

  void updateAcceleration(double timestep);
  void updateVelocity(double timestep);
  std::shared_ptr<Agent> moveRail(int timestep, double simTime);

  void initializeLowerRL(
      const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap);
  void setLowerRL(int simTime);
  bool isSignalStop();

  int getRouteId() { return routeId; }

  bool isStop() {
    if (stopStep > 0) {
      stopStep--;
      return true;
    }
    return false;
  }

  std::shared_ptr<Station> getNextStation() {
    if (nextStationNum < (int)depTimes.size()) {
      return table.at(depTimes.at(nextStationNum));
    } else {
      return nullptr;
    }
  }
};
}  // namespace Hongo

#endif /* Rail_hpp */
