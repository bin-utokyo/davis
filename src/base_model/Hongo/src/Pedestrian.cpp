//
//  Pedestrian.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/29.
//

#include "Pedestrian.hpp"

#include "Agent.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "Node.hpp"
#include "Signal.hpp"
#include "Utility.hpp"

namespace Hongo {

Pedestrian::Pedestrian(const std::shared_ptr<Agent> &_agent) : agent(_agent) {
  double velocityRate = 1.0 + 0.1 * Utility::getNormalDist();
  maxVelocity *= velocityRate;
  maxAcceleration *= velocityRate;
}

void Pedestrian::updateAcceleration(double timestep) {
  getObstacle();  // agentの周りの障害物の情報を取得
  Position &position =
      agent.lock()
          ->getPosition();  // 参照を渡しているので、positionを変更するとagentのpositionも変更される。

  double fx = 0.0;
  double fy = 0.0;

  std::string laneType = agent.lock()->getLane()->getType();
  if (position.dy > 0.0) {
    if (laneType == "car" || laneType == "bus") {
      fy = -position.dy / obstacle.outsideDist * forceCoefficient.openSpace;
    } else {
      fy = -position.dy / obstacle.outsideDist * forceCoefficient.wall;
    }
  } else if (position.dy < 0.0) {
    if (laneType == "car" || laneType == "bus") {
      fy = -position.dy / obstacle.insideDist * forceCoefficient.openSpace;
    } else {
      if (laneType == "ped") {
        fy = -position.dy / obstacle.insideDist * forceCoefficient.wall;
      } else {
        fy = -position.dy / obstacle.insideDist * forceCoefficient.openSpace;
      }
    }
  }
  // 信号は，行き先に依存するため，movePedestrianで考慮する．
  // if (obstacle.signal.second != nullptr && obstacle.signal.first >
  // DBL_EPSILON){//信号停止
  //     fx -= position.dx / obstacle.signal.first * forceCoefficient.openSpace;
  // }

  double tmpVelocity = Position::velocity(position);
  for (const auto &a : obstacle.agents) {
    Position aPos = a->getPosition();
    double deltadx = aPos.dx - position.dx;
    double deltady = aPos.dy - position.dy;
    if ((deltadx == 0.0 && aPos.x == position.x) ||
        (deltady == 0.0 &&
         aPos.y == position.y)) {  // 速度差0の場合は接触しない
      continue;
    }
    double timex = std::numeric_limits<double>::infinity();
    double timey = std::numeric_limits<double>::infinity();
    if (deltadx != 0.0) {
      timex = (aPos.x - position.x) / deltadx;
    }
    if (deltady != 0.0) {
      timey = (aPos.y - position.y) / deltady;
    }
    if (timex < 0 || timey < 0) {
      continue;
    }
    if (std::abs(timex - timey) <
        Position::distance(aPos, position) * 0.1) {  // とりあえずの接触判定
      int sign = 1;                                  // どちらに避けるか
      if (Utility::getRandRange01() < 0.5) {
        sign = -1;
      }
      double collisionTime = std::max(0.0, (timex + timey) / 2);
      if (collisionTime > 0.0) {
        double fabs =
            forceCoefficient.people / collisionTime;  // 受ける力の絶対値
        double stopRate = std::exp(
            -collisionTime);  // 横に避ける方向と停止する方向に分配する。差し迫って衝突しそうな場合ほど減速しやすい。

        fx -= position.dx / tmpVelocity * fabs * stopRate;  // 停止方向
        fy -= position.dy / tmpVelocity * fabs * stopRate;

        fx += -position.dy / tmpVelocity * fabs *
              std::sqrt(1 - stopRate * stopRate) * sign;  // 横方向
        fy += position.dx / tmpVelocity * fabs *
              std::sqrt(1 - stopRate * stopRate) * sign;
      }
    }
  }
  if (obstacle.signal.second == nullptr ||
      obstacle.signal.first > 3) {  // 信号停止中は前に進まない。
    if (agent.lock()->inNodeLane()) {  // nodeLane通過中は常に正方向に進む
      fx += (maxVelocity - tmpVelocity);
    } else {
      int dir = 1;
      if (std::abs(position.dx) > 0.0) {
        dir = position.dx / std::abs(position.dx);
      }
      fx += agent.lock()->getDirection() * maxVelocity - dir * tmpVelocity;
    }
  }

  fx += 0.05 * Utility::getNormalDist();  // 動きの不確かさ
  fy += 0.05 * Utility::getNormalDist();

  double fabs = fx * fx + fy * fy;
  if (fabs > maxAcceleration) {  // 加速度が大きくなりすぎるのを防ぐ。
    fx = fx / fabs * maxAcceleration;
    fy = fy / fabs * maxAcceleration;
  }
  position.ddx = fx;
  position.ddy = fy;
}

void Pedestrian::updateVelocity(double timestep) {
  Position &position = agent.lock()->getPosition();
  Position &prevPosition = agent.lock()->getPrevPosition();
  position.dx =
      position.dx + (prevPosition.ddx + position.ddx) * timestep * 0.5;
  position.dy =
      position.dy + (prevPosition.ddy + position.ddy) * timestep * 0.5;
  double tmpVelocity = Position::velocity(position);

  double velocityRate =
      std::min(300.0, agent.lock()->getDestinationDistance() + 50.0) / 300.0;
  if (tmpVelocity >
      maxVelocity * velocityRate) {  // 速度が大きくなりすぎるのを防ぐ。
    position.dx = position.dx / tmpVelocity * maxVelocity * velocityRate;
    position.dy = position.dy / tmpVelocity * maxVelocity * velocityRate;
  }
  // 実際の加速度に直す．
  prevPosition.ddx = position.dx - prevPosition.dx;
  prevPosition.ddy = position.dy - prevPosition.dy;
}

std::shared_ptr<Agent> Pedestrian::movePedestrian(double timestep,
                                                  double simTime) {
  std::shared_ptr<Agent> finishPed = nullptr;
  auto agentSPtr = agent.lock();
  Position &position = agentSPtr->getPosition();
  const Position &prevPosition = agentSPtr->getPrevPosition();
  double dx = 0.5 * (prevPosition.dx + position.dx) * timestep;
  double dy = 0.5 * (prevPosition.dy + position.dy) * timestep;
  // lane->nodeLane->laneで移動する場合が一番複雑で、lane->nodeLane,nodeLane->lane,nodeLane
  // or laneのみ,lane->終了の全6種類を考える
  if (std::abs(position.x) > 1000) {
    std::cout << std::endl;
  }
  if (!agentSPtr->inNodeLane()) {
    if (std::max(0.0, dx * agentSPtr->getDirection()) <
        (position.endx - position.x) * agentSPtr->getDirection()) {  // laneのみ
      position.x += dx;
      position.y = std::max(
          0.0,
          std::min(position.endy, position.y + dy));  // laneの外に出ないため
      return finishPed;
    } else {  // lane->nodeLane, lane->nodeLane->lane,lane->終了
      auto prevLane = agentSPtr->getLane();
      auto prevLink = agentSPtr->getLink();
      int prevDirection = agentSPtr->getDirection();

      std::shared_ptr<NodeLane> dnNodeLane;
      dnNodeLane =
          agentSPtr
              ->planDnLane();  // 次のdnLaneの取得,終了しない場合はdnNodeLane=nullptrになる
      if (dnNodeLane != nullptr && dnNodeLane->isSaturated()) {
        // 下流リンクに進めないため，link内で停止
        return finishPed;
      }
      agentSPtr->setDnLane(dnNodeLane);

      auto tmpLane = agentSPtr->getLane();
      if (obstacle.signal.second != nullptr) {
        if (tmpLane != nullptr) {
          if (!obstacle.signal.second->isPassible(
                  prevLane, 3, simTime,
                  tmpLane->getLink()
                      ->getId())) {  // 信号により進行先のリンクへいけなかった場合
            position.x = position.endx;
            position.dx = 0;
            position.ddx = 0;
            agentSPtr->setLink(prevLane->getLink());
            agentSPtr->setLane(prevLane);
            agentSPtr->offNodeLane();
            agentSPtr->setDirection(prevDirection);
            return finishPed;
          }
        }
      }
      prevLane->removeAgent(agentSPtr->getId(), simTime);

      double rate = (position.endx - position.x) / dx;
      dx = std::abs(dx - (position.endx -
                          position.x));  // nodeLane内は一方通行のため正値をとる
      dy = dy * (1 - rate);

      agentSPtr->addTravelLength(prevLink->getLength());
      double connectionUtility = 0;
      if (agentSPtr->getLane() != nullptr) {
        connectionUtility = prevLink->getConnectionUtility(
            agentSPtr->getLink()->getId(),
            agentSPtr->getNodeLane()->getNodeId(), 3);
      }
      agentSPtr->addUtility(prevLink->getUtility(3) + connectionUtility);

      if (agentSPtr->getLane() != nullptr) {
        agentSPtr->getNodeLane()->addAgent(agentSPtr);

        position.x = 0.0;
        position.y = std::max(
            0.0, std::min(position.endy, position.y + dy * rate / (1 - rate)));
        position.dx = std::abs(position.dx);
        position.endx = agentSPtr->getNodeLane()->getLength();
      } else {  // lane->終了
        position.dx = 0.0;
        position.dy = 0.0;
        return agent.lock();
      }
    }
  }
  if (agentSPtr->inNodeLane()) {
    if (dx < position.endx - position.x) {  // nodeLane内にとどまる場合
      position.x += dx;
      position.y = std::max(0.0, std::min(position.endy, position.y + dy));
      dx = 0.0;
      dy = 0.0;
    } else {
      agentSPtr->getNodeLane()->removeAgent(agentSPtr->getId());
      agentSPtr->getLane()->addAgent(agentSPtr, simTime);

      double rate = (agentSPtr->getNodeLane()->getLength() - position.x) / dx;
      dx -= agentSPtr->getNodeLane()->getLength() - position.x;
      dy = dy * (1 - rate);

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
      position.endy = agentSPtr->getLane()->getWidth();
      position.y = std::max(
          0.0, std::min(position.endy, position.y + dy * rate / (1 - rate)));

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
    position.y = std::max(0.0, std::min(position.endy, position.y + dy));
  }
  return finishPed;
}

void Pedestrian::getObstacle() {
  std::shared_ptr<Agent> agentSPtr = agent.lock();
  Position position = agentSPtr->getPosition();

  obstacle.insideDist = std::max(0.1, position.y);
  obstacle.outsideDist = std::max(0.1, position.endy - position.y);

  obstacle.agents.clear();
  if (!agentSPtr->inNodeLane()) {
    for (const auto &pair : agentSPtr->getLane()->getAgentMap()) {
      auto aPos = pair.second->getPosition();
      if ((aPos.x - position.x) * position.dx +
              (aPos.y - position.y) * position.dy >
          0) {  // aが視野の中にいる場合
        if (Position::distance(aPos, position) < viewLimit) {
          obstacle.agents.emplace_back(pair.second);
        }
      }
    }
  }
  std::shared_ptr<Signal> signal = nullptr;
  if (!agentSPtr->inNodeLane()) {
    int direction = agentSPtr->getDirection();
    int nodeId;
    if (direction > 0) {
      nodeId = agentSPtr->getLink()->getDNodeId();
    } else {
      nodeId = agentSPtr->getLink()->getONodeId();
    }
    signal = agentSPtr->getLink()->getNode(nodeId)->getSignal();
    if (signal != nullptr) {
      if (signal->isPassible(agentSPtr->getLane(), 3,
                             agentSPtr->getSimTime())) {
        signal = nullptr;
      }
    }
    obstacle.signal =
        std::make_pair(std::abs(position.endx - position.x), signal);
  } else {  // nodeLaneに入っている場合
    std::shared_ptr<NodeLane> nodeLane = agentSPtr->getNodeLane();
    double dist = 0.0;
    if (nodeLane->getR() == 0) {
      signal =
          agentSPtr->getLink()->getNode(nodeLane->getNodeId())->getSignal();
      if (signal != nullptr) {
        double ix, iy, ox, oy;
        std::tie(ox, oy) = nodeLane->getOXY();
        std::tie(ix, iy) = nodeLane->getIXY();
        double length1 = Utility::euclidean(ox, oy, ix, iy);
        if ((ix == 0 && iy == 0 && position.x < 1.0) ||
            (position.x > length1 &&
             position.x < length1 + 1.0)) {  // 交差点で曲がる場合
          if (signal->isPassible(agentSPtr->getLane(), 3,
                                 agentSPtr->getSimTime())) {
            signal = nullptr;
          } else {
            dist = forceCoefficient
                       .openSpace;  // この場合、速度が0になるような加速度になる
          }
        } else {
          signal = nullptr;
        }
      }
    }
    obstacle.signal = std::make_pair(dist, signal);
  }
}
}  // namespace Hongo
