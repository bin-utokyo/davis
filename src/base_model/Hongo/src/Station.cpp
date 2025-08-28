//
//  Station.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/19.
//

#include "Station.hpp"

#include "Agent.hpp"

namespace Hongo {

void Station::addAgent(const std::shared_ptr<Agent> &agent) {
  agentMap[agent->getId()] = agent;
}
}  // namespace Hongo
