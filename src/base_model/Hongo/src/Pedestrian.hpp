//
//  Pedestrian.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/29.
//

#ifndef Pedestrian_hpp
#define Pedestrian_hpp

#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Hongo {

class Lane;
class Agent;
class Signal;
struct Obstacle {
  double insideDist;  // レーンの中央線側の端までの距離
  double outsideDist;
  std::vector<std::shared_ptr<Agent>> agents;
  std::pair<double, std::shared_ptr<Signal>> signal;  // 距離、singal
};
struct ForceCoefficient {
  //(衝突までの時間)^-1に比例する力を受けるとする
  double wall = 0.03;
  double openSpace = 0.01;
  double people = 0.1;
};
class Pedestrian : public std::enable_shared_from_this<Pedestrian> {
  const double viewLimit = 10.0;  // m
  double maxVelocity = 1.7;       // m/s
  double maxAcceleration = 0.8;   // m/s^2

  std::weak_ptr<Agent> agent;
  Obstacle obstacle;

  ForceCoefficient forceCoefficient;

 public:
  Pedestrian(const std::shared_ptr<Agent> &_agent);

  void updateAcceleration(double timestep);  // 加速度を更新
  void updateVelocity(double timestep);      // 速度を更新。
  std::shared_ptr<Agent> movePedestrian(
      double timestep,
      double
          simTime);  // 行先が決まった状態でposition(位置)を更新する。（実際には行き先はまだ決まっていないが、Agent側のメソッドで逐次的に決めていく。）1タイムステップでのlaneの移動は一回まで
  void getObstacle();

  bool isSignalStop() { return obstacle.signal.second != nullptr; }
};
}  // namespace Hongo

#endif /* Pedestrian_hpp */
