//
//  Vehicle.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/18.
//

#include "Vehicle.hpp"

#include "Agent.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "Node.hpp"
#include "Signal.hpp"
#include "Utility.hpp"

namespace Hongo {

double VehicleBehaviorModel::getLaneChangeAcceptance(double leadSpacing,
                                                     double leadRelativeSpeed,
                                                     double lagSpacing,
                                                     double lagRelativeSpeed) {
  // 前or後ろに車両がいない場合は，relativeSpeed=0，relativeSpeed=前方or後続車両速度-速度
  double leadCreteriaGap =
      laneChangeLead.at(0) +
      laneChangeLead.at(1) * std::max(0.0, leadRelativeSpeed) +
      laneChangeLead.at(2) * std::min(0.0, leadRelativeSpeed);
  double lagCreteriaGap = laneChangeLag.at(0) +
                          laneChangeLag.at(1) * std::max(0.0, lagRelativeSpeed);

  double leadProb = Utility::stNormCDF(
      (std::log(std::max(0.0, leadSpacing)) - leadCreteriaGap) /
      laneChangeLead.at(4));
  double lagProb = Utility::stNormCDF(
      (std::log(std::max(0.0, lagSpacing)) - lagCreteriaGap) /
      laneChangeLag.at(3));

  return leadProb * lagProb;
}

double VehicleBehaviorModel::getCarFollowingProb(double headway) {
  if (headway < 30.0) {
    return 1 - Utility::stNormCDF((headway - CFJudge.at(0)) /
                                  std::exp(CFJudge.at(1)));
  } else {
    return 0.0;
  }
}

Vehicle::Vehicle(const std::shared_ptr<Agent> &_agent) : agent(_agent) {
  Position &position = _agent->getPosition();
  position.y = 1.5;
}

void Vehicle::updateAcceleration(double timestep) {
  getObstacle();

  std::shared_ptr<Agent> agentSpr = agent.lock();
  Position &position = agentSpr->getPosition();
  const Position &prevPosition = agentSpr->getPrevPosition();
  double velocity = std::abs(position.dx + position.ddx * timestep);
  int sign = agentSpr->getLane()->getSign();
  if (agentSpr->inNodeLane()) {
    sign = 1;
  }
  // 各obstacleについて加速度を計算し，最小のものを採用する．
  position.ddx = 100 * sign;
  if (vehicleObstacle.nearestAgent.second != nullptr) {  // 歩行者がいる場合
    double spacing = vehicleObstacle.nearestAgent.first - 5.0;
    // double relativeSpeed = std::max((spacing - 3.0) / 10.0, 0.0) -
    // std::abs(position.dx);
    double relativeSpeed = -std::abs(position.dx);

    position.ddx =
        std::min(vbModel.getAcceleration(velocity, spacing, relativeSpeed),
                 position.ddx * sign) *
        sign;
  }
  if (vehicleObstacle.stationDistance > 0) {  // バス停の近くの場合
    double spacing = vehicleObstacle.stationDistance + 2;
    // double relativeSpeed = std::max((spacing - 3.0) / 10.0, 0.0) -
    // std::abs(position.dx);
    double relativeSpeed = -std::abs(position.dx);

    position.ddx =
        std::min(vbModel.getAcceleration(velocity, spacing, relativeSpeed),
                 position.ddx * sign) *
        sign;

    if (spacing < 0.5 *
                      (0.5 * (position.ddx + prevPosition.ddx) * timestep +
                       2.0 * prevPosition.dx) *
                      timestep * sign) {  // バス停を通り過ぎてしまう場合
      position.ddx =
          2.0 * (2.0 * spacing * sign / timestep - 2.0 * prevPosition.dx) /
              timestep -
          prevPosition.ddx;
    } else if ((0.5 * (position.ddx + prevPosition.ddx) * timestep +
                prevPosition.dx) *
                       sign <
                   1.0 &&
               spacing > 5.0) {  // バス停との距離を大きく開けて停止している場合
      position.ddx = 1.0;
    }
  }
  if (vehicleObstacle.signal.second != nullptr) {  // 信号停止の場合
    double spacing = vehicleObstacle.signal.first;
    // double relativeSpeed = std::max((spacing - 3.0) / 10.0, 0.0) -
    // std::abs(position.dx);
    double relativeSpeed = -std::abs(position.dx);

    position.ddx =
        std::min(vbModel.getAcceleration(velocity, spacing, relativeSpeed),
                 position.ddx * sign) *
        sign;

    if (spacing - 5.0 <
        0.5 *
            (0.5 * (position.ddx + prevPosition.ddx) * timestep +
             2.0 * prevPosition.dx) *
            timestep * sign) {  // 信号を通り過ぎてしまう場合
      position.ddx = 2.0 *
                         (2.0 * (spacing - 5.0) * sign / timestep -
                          2.0 * prevPosition.dx) /
                         timestep -
                     prevPosition.ddx;
    } else if ((0.5 * (position.ddx + prevPosition.ddx) * timestep +
                prevPosition.dx) *
                       sign <
                   1.0 &&
               spacing > 7.0) {  // 信号との距離を大きく開けて停止している場合
      position.ddx = 0.5;
    }
  }
  if (carFollowingJudge()) {  // car-following acceleration
    int signLead = sign;
    if (vehicleObstacle.leadVehicle.second->inNodeLane()) {
      signLead = 1;
    }

    const Position &aPos =
        vehicleObstacle.leadVehicle.second->getPrevPosition();
    // double relativeSpeed = (aPos.dx - position.dx) * sign;
    double relativeSpeed = std::abs(aPos.dx) - std::abs(position.dx);
    double spacing = std::max(0.1, vehicleObstacle.leadVehicle.first - length);
    double density;
    if (nextSideLane == nullptr) {
      std::shared_ptr<Lane> agentLane = agentSpr->getLane();
      density = (double)agentLane->getAgentMap().size() /
                agentLane->getLink()->getLength() * 1000.0;
    } else {  // 車線変更する場合
      density = (double)nextSideLane->getAgentMap().size() /
                nextSideLane->getLink()->getLength() * 1000.0;
    }
    position.ddx = std::min(vbModel.getAcceleration(velocity, spacing, density,
                                                    relativeSpeed),
                            position.ddx * sign) *
                   sign;

    // if (spacing + 0.5 * (aPos.ddx * timestep + 2.0 * aPos.dx) * timestep*
    // signLead - 5.0 < 0.5 * (0.5 * (position.ddx + prevPosition.ddx) *
    // timestep + 2.0 * prevPosition.dx) * timestep *
    // sign){//前の車を追い越してしまう時
    if (spacing - 2.0 <
        0.5 *
            (0.5 * (position.ddx + prevPosition.ddx) * timestep +
             2.0 * prevPosition.dx) *
            timestep * sign) {  // 前の車を追い越してしまう時
      double spacing_mov =
          spacing +
          0.5 * (aPos.ddx * timestep + 2.0 * aPos.dx) * timestep * signLead -
          2.0;
      position.ddx = 2.0 *
                         (2.0 * (spacing - 2.0) * sign / timestep -
                          2.0 * prevPosition.dx) /
                         timestep -
                     prevPosition.ddx;
    } else if ((0.5 * (position.ddx + prevPosition.ddx) * timestep +
                prevPosition.dx) *
                       sign <
                   1.0 &&
               spacing >
                   5.0) {  // 前方車両との距離を大きく開けて停止している場合
      position.ddx = 1.0;
    }

  } else {  // free-flow acceleration
    double vd = agentSpr->getLink()->getSpeedLimit();
    if (agentSpr->inNodeLane() && agentSpr->getNodeLane()->getR() != 0) {
      vd = std::min(
          vd, std::sqrt(0.5 *
                        agentSpr->getNodeLane()->getR()));  // 加速度が0.5G以下
    }
    position.ddx = std::min(vbModel.getFreeFlowAcceleration(velocity, vd),
                            position.ddx * sign) *
                   sign;
  }
  position.ddx += 0.05 * Utility::getNormalDist();  // 動きの不確かさ
  position.ddx /= weight;

  if (position.ddx * sign > maxAcceleration) {
    position.ddx = maxAcceleration * sign;
  } else if (position.ddx * sign < minimumAcceleration) {
    position.ddx = minimumAcceleration * sign;
  }
}

void Vehicle::updateVelocity(double timestep) {
  // 速度は前タイムステップの加速度に基づいて更新する．
  std::shared_ptr<Agent> agentSPtr = agent.lock();
  Position &position = agentSPtr->getPosition();
  Position &prevPosition = agentSPtr->getPrevPosition();
  double prevdx = position.dx;
  position.dx =
      position.dx + 0.5 * (prevPosition.ddx + position.ddx) * timestep;
  if (agentSPtr->getLane()->getSign() == 1 || agentSPtr->inNodeLane())
    position.dx = std::max(0.0, position.dx);  // 逆方向に進まないため
  else
    position.dx = std::min(0.0, position.dx);

  double maxVelocity = agentSPtr->getLink()->getSpeedLimit() + 3.0;
  if (agent.lock()->inNodeLane() && agentSPtr->getNodeLane()->getR() != 0) {
    maxVelocity = std::sqrt(0.5 * agentSPtr->getNodeLane()->getR());  // 0.5G
  }
  if (std::abs(position.dx) >
      maxVelocity) {  // カーブで速度が大きくなりすぎるのを防ぐ。
    position.dx = maxVelocity * position.dx / std::abs(position.dx);
  }
  prevPosition.ddx = position.dx - prevdx;  // 実際の加速度に直す
}

std::shared_ptr<Agent> Vehicle::moveVehicle(double timestep, double simTime) {
  // 一つ前のタイムステップの速度を使って位置を更新する．
  std::shared_ptr<Agent> finishVehicle = nullptr;
  std::shared_ptr<Agent> agentSPtr = agent.lock();
  Position &position = agentSPtr->getPosition();
  const Position &prevPosition = agentSPtr->getPrevPosition();
  if (nextSideLane != nullptr) {
    agentSPtr->getLane()->removeAgent(agentSPtr->getId());
    agentSPtr->setLane(nextSideLane);
    nextSideLane->addAgent(agentSPtr);
    nextSideLane = nullptr;
  }
  double dx = 0.5 * (prevPosition.dx + position.dx) * timestep;
  // lane->nodeLane->laneで移動する場合が一番複雑で、lane->nodeLane,nodeLane->lane,nodeLane
  // or laneのみ,lane->終了の全6種類を考える
  if (!agentSPtr->inNodeLane()) {
    if (dx * agentSPtr->getDirection() <
        (position.endx - position.x) * agentSPtr->getDirection()) {  // laneのみ
      position.x += dx;
      return finishVehicle;
    } else {  // lane->nodeLane, lane->nodeLane->lane,lane->終了
      std::shared_ptr<NodeLane> dnNodeLane;
      auto prevLink = agentSPtr->getLink();
      dnNodeLane =
          agentSPtr
              ->planDnLane();  // 次のdnLaneの取得,終了しない場合はdnNodeLane=nullptrになる
      if (dnNodeLane != nullptr && dnNodeLane->isSaturated()) {
        // 下流リンクに進めないため，link内で停止
        return finishVehicle;
      }

      agentSPtr->getLane()->removeAgent(agentSPtr->getId(), simTime);
      agentSPtr->setDnLane(dnNodeLane);
      dx = std::abs(dx - (position.endx -
                          position.x));  // nodeLane内は一方通行のため正値をとる

      agentSPtr->addTravelLength(prevLink->getLength());
      double connectionUtility = 0;
      if (agentSPtr->getLane() != nullptr) {
        connectionUtility = prevLink->getConnectionUtility(
            agentSPtr->getLink()->getId(),
            agentSPtr->getNodeLane()->getNodeId(), 0);
      }
      agentSPtr->addUtility(prevLink->getUtility(0) + connectionUtility);

      if (agentSPtr->getLane() != nullptr) {
        agentSPtr->getNodeLane()->addAgent(agentSPtr);

        position.x = 0.0;
        position.dx = std::abs(position.dx);
        position.endx = agentSPtr->getNodeLane()->getLength();
      } else {  // lane->終了
        position.dx = 0.0;
        return agent.lock();
      }
    }
  }
  if (agentSPtr->inNodeLane()) {
    if (dx < position.endx - position.x) {  // nodeLane内にとどまる場合
      position.x += dx;
      dx = 0.0;
    } else {
      agentSPtr->getNodeLane()->removeAgent(agentSPtr->getId());
      agentSPtr->getLane()->addAgent(agentSPtr, simTime);

      dx -= agentSPtr->getNodeLane()->getLength() - position.x;

      // positionのx,y,endx,endyの設定
      auto nextLink = agentSPtr->getLink();
      int nodeId = agentSPtr->getNodeLane()->getNodeId();
      int nextNodeId = nextLink->getAnotherNodeId(nodeId);
      if (agentSPtr->getDirection() == 1) {  // oNode->dNodeの方向に進む場合
        position.x = nextLink->getOffset(nodeId);
        position.endx = nextLink->getLength() - nextLink->getOffset(nextNodeId);
        position.endx = std::max(
            0.0,
            std::min(
                nextLink->getLength(),
                position.endx));  // endxがリンクの長さの範囲に入るようにする
      } else {
        position.x = nextLink->getLength() - nextLink->getOffset(nodeId);
        position.endx = nextLink->getOffset(nextNodeId);
        position.endx = std::max(
            0.0,
            std::min(
                nextLink->getLength(),
                position.endx));  // endxがリンクの長さの範囲に入るようにする
      }

      // dxの符号の設定
      dx = std::abs(dx) * agentSPtr->getDirection();
      position.dx = std::abs(position.dx) * agentSPtr->getDirection();
      position.ddx = std::abs(position.ddx) * agentSPtr->getDirection();

      agentSPtr->offNodeLane();
    }
  }
  if (std::abs(dx) > 0) {  // lane->nodeLane->lane,nodeLane->lane
    int direction = agentSPtr->getDirection();
    position.x =
        std::min((position.x + dx) * direction, position.endx * direction) *
        direction;  // xの範囲を超えないため
  }
  return finishVehicle;
}

bool Vehicle::laneChangeJudge() {
  // nodeLane内にいるときは呼ばれない。
  bool accept = true;

  Position &position = agent.lock()->getPosition();

  double leadSpace = std::max(0.0, vehicleObstacle.leadVehicle.first);
  double lagSpace = std::max(0.0, vehicleObstacle.lagVehicle.first);

  int sign = agent.lock()->getLane()->getSign();
  double leadRelativeSpeed = 0;
  double lagRelativeSpeed = 0;
  if (vehicleObstacle.leadVehicle.second != nullptr) {
    Position &aPos = vehicleObstacle.leadVehicle.second->getPosition();
    // leadRelativeSpeed = (aPos.dx - position.dx) * sign;
    leadRelativeSpeed = std::abs(aPos.dx) - std::abs(position.dx);
  }
  if (vehicleObstacle.lagVehicle.second != nullptr) {
    Position &aPos = vehicleObstacle.lagVehicle.second->getPosition();
    // lagRelativeSpeed = (aPos.dx - position.dx) * sign;
    lagRelativeSpeed = std::abs(aPos.dx) - std::abs(position.dx);
  }

  double acceptProb = vbModel.getLaneChangeAcceptance(
      leadSpace, leadRelativeSpeed, lagSpace, lagRelativeSpeed);

  double rnd = Utility::getRandRange01();
  if (rnd > acceptProb) {
    accept = false;
  }
  return accept;
}

bool Vehicle::carFollowingJudge() {
  carFollowing = false;
  if (vehicleObstacle.leadVehicle.second != nullptr) {
    double headway = vehicleObstacle.leadVehicle.first - length;

    double cfProb = vbModel.getCarFollowingProb(headway);

    double rnd = Utility::getRandRange01();
    if (rnd < cfProb) {
      carFollowing = true;
    }
  }
  return carFollowing;
}

void Vehicle::getObstacle() {
  // 探索範囲は前方100mまで（lane走行中は同一link内とその先のnodeLaneのみ、nodeLane走行中はnodeLane->laneまで）
  // 後方の探索範囲は50mまで（同一link内のみ）
  vehicleObstacle.leadVehicle = std::make_pair(viewLimit, nullptr);
  vehicleObstacle.lagVehicle = std::make_pair(viewLimit / 2.0, nullptr);
  vehicleObstacle.nearestAgent = std::make_pair(viewLimit, nullptr);
  vehicleObstacle.signal = std::make_pair(viewLimit, nullptr);

  std::shared_ptr<Agent> agentSpr = agent.lock();
  const Position &position = agentSpr->getPosition();
  std::shared_ptr<Lane> nextLane;
  int sign = agentSpr->getLane()->getSign();
  if (agentSpr->inNodeLane()) {  // 前方車両,交差点内agentを探索
    Position aPos;
    double dist;
    for (const auto &pair : agentSpr->getNodeLane()->getAgentMap()) {
      aPos = pair.second->getPosition();
      dist = aPos.x - position.x;
      if (dist > 0 && dist < vehicleObstacle.leadVehicle.first) {
        vehicleObstacle.leadVehicle = std::make_pair(dist, pair.second);
      }
    }
    if (vehicleObstacle.leadVehicle.second ==
        nullptr) {  // nodelaneの先のlaneを探索
      for (const auto &pair : agentSpr->getLane()->getAgentMap()) {
        aPos = pair.second->getPosition();
        dist = agentSpr->getLink()->getLength() - std::abs(aPos.endx - aPos.x) +
               (position.endx - position.x);
        if (dist > 0 && dist < vehicleObstacle.leadVehicle.first) {
          vehicleObstacle.leadVehicle = std::make_pair(dist, pair.second);
        }
      }
    }
    std::shared_ptr<NodeLane> nodeLane = agentSpr->getNodeLane();
    if (nodeLane->getR() != 0) {  // 曲がる場合，歩行者との相互作用を考慮
      Position globalPos = agentSpr->getGlobalPosition();
      for (const auto &nl : nodeLane->getNode()->getNodeLanes()) {
        if (nl == nodeLane) {
          continue;
        }
        if (NodeLane::crossing(nodeLane, nl)) {
          for (const auto &pair : nl->getAgentMap()) {
            aPos = pair.second->getGlobalPosition();
            double deltadx = aPos.dx - globalPos.dx;
            double deltady = aPos.dy - globalPos.dy;
            if ((deltadx == 0.0 && aPos.x == globalPos.x) ||
                (deltady == 0.0 &&
                 aPos.y == globalPos.y)) {  // 速度差0の場合は接触しない
              continue;
            }
            double timex = std::numeric_limits<double>::infinity();
            double timey = std::numeric_limits<double>::infinity();
            if (deltadx != 0.0) {
              timex = (aPos.x - globalPos.x) / deltadx;
            }
            if (deltady != 0.0) {
              timey = (aPos.y - globalPos.y) / deltady;
            }
            if (timex < 0 || timey < 0) {
              continue;
            }
            if (std::abs(timex - timey) <
                Position::distance(aPos, globalPos) * 0.1) {
              if (Position::distance(aPos, globalPos) <
                  vehicleObstacle.nearestAgent.first) {
                vehicleObstacle.nearestAgent = std::make_pair(
                    Position::distance(aPos, globalPos), pair.second);
              }
            }
          }
        }
      }
    }
    const auto &priorityLinks = nodeLane->getDnLink()->getPriorityLinks();
    if (priorityLinks.size() >
        0) {  // 進行先のリンクに優先リンクがある場合，優先リンク内のエージェントの位置を考慮
      Position globalPos = agentSpr->getGlobalPosition();
      for (const auto &plink : priorityLinks) {
        for (const auto &plane : plink->getLanes()) {
          for (const auto &pair : plane->getAgentMap()) {
            aPos = pair.second->getGlobalPosition();
            double deltadx = aPos.dx - globalPos.dx;
            double deltady = aPos.dy - globalPos.dy;
            if ((deltadx == 0.0 && aPos.x == globalPos.x) ||
                (deltady == 0.0 &&
                 aPos.y == globalPos.y)) {  // 速度差0の場合は接触しない
              continue;
            }
            double timex = std::numeric_limits<double>::infinity();
            double timey = std::numeric_limits<double>::infinity();
            if (deltadx != 0.0) {
              timex = (aPos.x - globalPos.x) / deltadx;
            }
            if (deltady != 0.0) {
              timey = (aPos.y - globalPos.y) / deltady;
            }
            if (timex < 0 || timey < 0) {
              continue;
            }
            if (std::abs(timex - timey) <
                Position::distance(aPos, globalPos) * 0.1) {
              if (Position::distance(aPos, globalPos) <
                  vehicleObstacle.nearestAgent.first) {
                vehicleObstacle.nearestAgent = std::make_pair(
                    Position::distance(aPos, globalPos), pair.second);
              }
            }
          }
        }
      }
    }
    return;
  }
  const auto &priorityLinks = agentSpr->getLink()->getPriorityLinks();
  if (priorityLinks.size() >
      0) {  // 進行中のリンクに優先リンクがある場合，優先リンク内のエージェントの位置を考慮
    Position globalPos = agentSpr->getGlobalPosition();
    for (const auto &plink : priorityLinks) {
      for (const auto &plane : plink->getLanes()) {
        for (const auto &pair : plane->getAgentMap()) {
          const auto &aPos = pair.second->getGlobalPosition();
          double deltadx = aPos.dx - globalPos.dx;
          double deltady = aPos.dy - globalPos.dy;
          if ((deltadx == 0.0 && aPos.x == globalPos.x) ||
              (deltady == 0.0 &&
               aPos.y == globalPos.y)) {  // 速度差0の場合は接触しない
            continue;
          }
          double timex = std::numeric_limits<double>::infinity();
          double timey = std::numeric_limits<double>::infinity();
          if (deltadx != 0.0) {
            timex = (aPos.x - globalPos.x) / deltadx;
          }
          if (deltady != 0.0) {
            timey = (aPos.y - globalPos.y) / deltady;
          }
          if (timex < 0 || timey < 0) {
            continue;
          }
          if (std::abs(timex - timey) <
              Position::distance(aPos, globalPos) * 0.1) {
            if (Position::distance(aPos, globalPos) <
                vehicleObstacle.nearestAgent.first) {
              vehicleObstacle.nearestAgent = std::make_pair(
                  Position::distance(aPos, globalPos), pair.second);
            }
          }
        }
      }
    }
  }

  nextLane = agentSpr->planLaneChange();
  Position aPos;
  double dist;
  for (const auto &pair : nextLane->getAgentMap()) {
    aPos = pair.second->getPosition();
    dist = (aPos.x - position.x) * sign;
    if (dist > 0 && dist < vehicleObstacle.leadVehicle.first) {
      vehicleObstacle.leadVehicle = std::make_pair(dist, pair.second);
    } else if (dist < 0 && std::abs(dist) < vehicleObstacle.lagVehicle.first) {
      vehicleObstacle.lagVehicle = std::make_pair(std::abs(dist), pair.second);
    }
  }
  if (nextLane->getId() != agentSpr->getLane()->getId()) {  // 車線変更する場合
    // 車線変更可否
    if (!laneChangeJudge() &&
        std::abs(position.endx - position.x) >
            20.0) {  // 交差点が近い場合は強制的に車線変更。このifの中は車線変更しない場合
      nextLane = agentSpr->getLane();
      vehicleObstacle.leadVehicle = std::make_pair(viewLimit, nullptr);
      vehicleObstacle.lagVehicle = std::make_pair(viewLimit / 2.0, nullptr);
      for (const auto &pair : agentSpr->getLane()->getAgentMap()) {
        aPos = pair.second->getPosition();
        dist = (aPos.x - position.x) * sign;
        if (dist > 0 && dist < vehicleObstacle.leadVehicle.first) {
          vehicleObstacle.leadVehicle = std::make_pair(dist, pair.second);
        } else if (dist < 0 &&
                   std::abs(dist) < vehicleObstacle.lagVehicle.first) {
          vehicleObstacle.lagVehicle =
              std::make_pair(std::abs(dist), pair.second);
        }
      }
    } else {  // 車線変更する場合
      nextSideLane = nextLane;
    }
  }
  // 下流nodeLane
  if (vehicleObstacle.leadVehicle.second != nullptr) {
    for (const auto &dnLane : nextLane->getDnLane()) {
      for (const auto &pair : dnLane->getAgentMap()) {
        aPos = pair.second->getPosition();
        dist = std::abs(position.endx - position.x) + aPos.x;
        if (dist < vehicleObstacle.leadVehicle.first) {
          vehicleObstacle.leadVehicle = std::make_pair(dist, pair.second);
        }
      }
    }
  }
  // 前方信号を探索
  std::shared_ptr<Node> nextNode =
      agentSpr->getLink()->getNode(agentSpr->getLane()->getDNodeId());
  std::shared_ptr<Signal> signal = nextNode->getSignal();
  if (signal != nullptr) {
    if (!signal->isPassible(nextLane, 0, agentSpr->getSimTime())) {
      double space =
          std::max(0.1, std::abs(position.endx - position.x) -
                            nextNode->getOffset(agentSpr->getLink()->getId()));
      if (space < viewLimit)
        vehicleObstacle.signal = std::make_pair(space, signal);
    }
  }
}

double Vehicle::getHeadway() {
  // 最も近い位置にある障害物までの距離を返す．
  double headway = viewLimit;
  if (vehicleObstacle.leadVehicle.second != nullptr)
    headway = std::min(headway, vehicleObstacle.leadVehicle.first);
  if (vehicleObstacle.lagVehicle.second != nullptr)
    headway = std::min(headway, vehicleObstacle.lagVehicle.first);
  if (vehicleObstacle.nearestAgent.second != nullptr)
    headway = std::min(headway, vehicleObstacle.nearestAgent.first);
  if (vehicleObstacle.signal.second != nullptr)
    headway = std::min(headway, vehicleObstacle.signal.first);
  return headway;
}

}  // namespace Hongo
