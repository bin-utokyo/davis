//
//  Rail.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/19.
//

#include "Rail.hpp"

#include "Agent.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "RL.hpp"
#include "Station.hpp"
#include "Vehicle.hpp"

namespace Hongo {

Rail::Rail(const std::shared_ptr<Agent> &_agent, int _routeId)
    : agent(_agent), routeId(_routeId) {}

void Rail::insertTimetable(double depTime,
                           const std::shared_ptr<Station> &station) {
  table.emplace(depTime, station);
  depTimes.emplace_back(depTime);
  std::sort(depTimes.begin(), depTimes.end());
}

void Rail::initializeLowerRL(
    const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap) {
  int oNodeId, tmpDNodeId;
  std::shared_ptr<Agent> agentSpr = agent.lock();
  std::tie(oNodeId, tmpDNodeId) = agentSpr->getOD();
  lowerRL = std::make_shared<RL>(_linkMap, 0);
  lowerRL->setRail(true);
  lowerRL->setBeta(1.0);
  lowerRL->setK(1);
  lowerRL->setMu(0.01);
  agentSpr->setLowerRL(lowerRL);
  vehicle = std::make_shared<Vehicle>(agentSpr);
  vehicle->setWeight(1.5);
}
void Rail::setLowerRL(int simTime) {
  nextStationNum++;

  std::shared_ptr<Agent> agentSpr = agent.lock();
  int oNodeId, tmpDNodeId;
  std::tie(oNodeId, tmpDNodeId) = agentSpr->getOD();
  agentSpr->setONodeId(tmpDNodeId);
  if (nextStationNum < depTimes.size()) {
    std::shared_ptr<Station> nextStation =
        table.at(depTimes.at(nextStationNum));
    agentSpr->setNextDNodeId(nextStation->getLane()->getDNodeId());
    if (agentSpr->getLane() == nullptr) {
      lowerRL->setDLane(tmpDNodeId, nextStation->getLane());
    } else {
      lowerRL->setDLane(agentSpr->getLane(), nextStation->getLane());
    }
  } else {
    agentSpr->setNextDNodeId(agentSpr->getDNodeId());
    if (agentSpr->getLane() == nullptr) {
      lowerRL->setDNodeId(tmpDNodeId, agentSpr->getDNodeId());
    } else {
      lowerRL->setDNodeId(agentSpr->getLane(), agentSpr->getDNodeId());
    }
  }

  if (agentSpr->getLane() == nullptr) {
    std::shared_ptr<Link> link = lowerRL->chooseFirstLink(tmpDNodeId);
    if (link == nullptr) {
      agentSpr->clearLane();
    } else {
      agentSpr->setLink(link);
      std::shared_ptr<Lane> lane = lowerRL->chooseFirstLane(link, tmpDNodeId);
      if (lane != nullptr) {
        agentSpr->setLane(lane);
        lane->addAgent(agentSpr, simTime);
        Position &position = agentSpr->getPosition();
        if (link->getDNodeId() == oNodeId) {
          position.x = link->getLength() - link->getOffset(oNodeId);
          position.endx = link->getOffset(link->getAnotherNodeId(oNodeId));
          agentSpr->setDirection(-1);
        } else {
          position.x = link->getOffset(oNodeId);
          position.endx = link->getLength() -
                          link->getOffset(link->getAnotherNodeId(oNodeId));
          agentSpr->setDirection(1);
        }
        position.endy = lane->getWidth();
      }
    }
  }
}

void Rail::updateAcceleration(double timestep) {
  if (stopStep > 0) {
    return;
  }
  vehicle->updateAcceleration(timestep);
}

void Rail::updateVelocity(double timestep) {
  if (stopStep > 0) {
    return;
  }
  vehicle->updateVelocity(timestep);
}

std::shared_ptr<Agent> Rail::moveRail(int timestep, double simTime) {
  std::shared_ptr<Agent> finishAgent = nullptr;
  std::shared_ptr<Agent> agentSpr = agent.lock();
  if (stopStep > 0) {
    return nullptr;
  }
  const Position &position = agentSpr->getPosition();
  finishAgent = vehicle->moveVehicle(timestep, simTime);
  for (const auto &pair : agentMap) {  // 乗客agentの位置を更新
    pair.second->setLink(agentSpr->getLink());
    pair.second->setLane(agentSpr->getLane());
    Position &aPos = pair.second->getPosition();
    aPos.x = position.x;
    aPos.y = position.y;
    aPos.dx = position.dx;
    aPos.dy = position.dy;
  }
  std::shared_ptr<Station> nextStation = getNextStation();
  if (finishAgent == nullptr && nextStation != nullptr) {
    vehicle->setStationDistance(-1);
    if (!agentSpr->inNodeLane() &&
        (agentSpr->getLink()->getId() ==
         nextStation->getLane()->getLink()->getId())) {
      double stationDist = std::max(
          1.0, (std::min(nextStation->getX(),
                         nextStation->getLane()->getLink()->getLength()) -
                agentSpr->getPosition().x) *
                   nextStation->getLane()->getSign());
      if (stationDist < 5) {  // 停車判定
        stopStep = std::max(
            30, (int)((depTimes.at(nextStationNum) - simTime) / timestep));

        std::shared_ptr<Lane> nonCarLane = nextStation->getLane();
        int left = nonCarLane->getId() / std::abs(nonCarLane->getId());
        std::string tmpLaneType = nonCarLane->getType();
        while (tmpLaneType == "car" || tmpLaneType == "bus") {
          nonCarLane =
              nonCarLane->getLink()->getLane(nonCarLane->getId() + left);
          if (nonCarLane == nullptr) {
            break;
          }
          tmpLaneType = nonCarLane->getType();
        }
        // 行き先の違うagentをおろす
        int tmpStationId = nextStation->getId();
        int nextStationId = -1;
        int agentNextStationId = -1;
        if (nextStationNum + 1 < (int)depTimes.size()) {
          nextStationId = table.at(depTimes.at(nextStationNum + 1))->getId();
        }
        std::vector<int> eraseId;
        for (const auto &pair : agentMap) {
          agentNextStationId = pair.second->getNextStation(tmpStationId);
          if (nextStationId == -1 || agentNextStationId != nextStationId) {
            if (agentNextStationId != -1) {  // 別のバスに乗車
              nextStation->addAgent(pair.second);
            } else {  // イグレスに移行
              Position &aPos = pair.second->getPosition();
              pair.second->setLane(nonCarLane);
              pair.second->setLink(nonCarLane->getLink());
              aPos.x = nextStation->getX();
              aPos.y = 0;
              aPos.dx = 0;
              aPos.dy = 0.1;
              pair.second->offBord();
              pair.second->setDestinationTransport();
            }
            eraseId.emplace_back(pair.first);
          }
        }
        for (const auto &i : eraseId) {
          agentMap.erase(i);
        }
        // 行き先が同じagentをのせる
        auto &stationAgentMap = nextStation->getAgentMap();
        eraseId.clear();
        for (const auto &pair : stationAgentMap) {
          if (nextStationId != -1 &&
              pair.second->getNextStation(tmpStationId) ==
                  nextStationId) {  // 行き先が同じ場合
            agentMap[pair.first] = pair.second;
            eraseId.emplace_back(pair.first);
          }
        }
        for (const auto &i : eraseId) {
          stationAgentMap.erase(i);
        }

        setLowerRL(simTime + stopStep * timestep);
      } else if (stationDist < 30) {
        vehicle->setStationDistance(stationDist);
      }
    }
  }
  if (finishAgent != nullptr && agentSpr->getLane() == nullptr) {  // 走行終了
    return agentSpr;
  } else if (finishAgent != nullptr) {
    setLowerRL(simTime);
  }
  return nullptr;
}

bool Rail::isSignalStop() {
  if (vehicle != nullptr) {
    return vehicle->isSignalStop();
  } else {
    return false;
  }
}

}  // namespace Hongo
