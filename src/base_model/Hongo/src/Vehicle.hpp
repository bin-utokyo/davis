//
//  Vehicle.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/18.
//

#ifndef Vehicle_hpp
#define Vehicle_hpp

#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Hongo {

class Signal;
class Agent;
class Lane;
struct VehicleObstacle {
  std::pair<double, std::shared_ptr<Agent>> leadVehicle, lagVehicle,
      nearestAgent;                                   // 距離, agent
  std::pair<double, std::shared_ptr<Signal>> signal;  // 距離, signal
  double stationDistance = -1;                        // 公共交通 30以下
};
class VehicleBehaviorModel {
  std::vector<double> laneChangeLead = {1.127, -2.178, -0.153, 0.789, 1.217};
  std::vector<double> laneChangeLag = {0.968, 0.491, 0.107, 0.622};
  // std::vector<double> CFAcceleration = {0.0355, 0.291, -0.166, 0.55, 0.52};
  std::vector<double> CFAcceleration = {1.5, 0.3, -0.7, 0, 1};
  // std::vector<double> CFAcceleration = {0.6, 0, 0, 0, 1};
  std::vector<double> CFDeceleration = {-0.86, -0.565, 0.143, 0.834};
  // std::vector<double> FFAcceleration = {0.0881, 17.636, -0.105};
  std::vector<double> FFAcceleration = {0.3};
  // std::vector<double> CFJudge = {2.579, -0.799};
  std::vector<double> CFJudge = {10.0, -0.799};

 public:
  double getLaneChangeAcceptance(double leadSpacing, double leadRelativeSpeed,
                                 double lagSpacing, double lagRelativeSpeed);

  double getCarFollowingProb(double headway);

  double getAcceleration(double velocity, double spacing,
                         double relativeSpeed) {
    // velocity: 現在の車両速度
    // spacing: 一つ前ステップの車両間隔
    // relativeSpeed: 一つ前ステップの車両相対速度
    double density = 1.0 / spacing * 1000.0;
    return getAcceleration(velocity, spacing, density, relativeSpeed);
  }
  double getAcceleration(double velocity, double spacing, double density,
                         double relativeSpeed) {
    if (relativeSpeed > 0) {
      return CFAcceleration.at(0) *
             std::pow(std::max(0.1, velocity), CFAcceleration.at(1)) *
             std::pow(spacing, CFAcceleration.at(2)) *
             std::pow(density, CFAcceleration.at(3)) *
             std::pow(relativeSpeed, CFAcceleration.at(4));
    } else if (relativeSpeed < 0) {
      // return CFDeceleration.at(0) * std::pow(spacing, CFDeceleration.at(1)) *
      // std::pow(density, CFDeceleration.at(2)) * std::pow(-relativeSpeed,
      // CFDeceleration.at(3));
      return -CFAcceleration.at(0) *
             std::pow(std::max(0.1, velocity), CFAcceleration.at(1)) *
             std::pow(spacing, CFAcceleration.at(2)) *
             std::pow(density, CFAcceleration.at(3)) *
             std::pow(-relativeSpeed,
                      CFAcceleration.at(4));  // 減速と加速で共通の式を使用
    } else {
      return 0.0;
    }
  }
  double getFreeFlowAcceleration(double velocity, double speedLimit) {
    if (velocity > 0) {
      return FFAcceleration.at(0) * (speedLimit - velocity);  // 車両重量未考慮
    } else {
      return FFAcceleration.at(0) * speedLimit;  // 車両重量未考慮
    }
  }
};

class Vehicle : public std::enable_shared_from_this<Vehicle> {
  const double viewLimit = 100.0;     // m
  double maxAcceleration = 6.0;       // m/s^2
  double minimumAcceleration = -6.0;  // m/s^2

  double length = 2.0;  // m
  double weight = 1.0;  // 乗用車：1, バス:1.5

  bool carFollowing = false;  // 車両追従になっているか

  std::weak_ptr<Agent> agent;
  VehicleObstacle vehicleObstacle;
  VehicleBehaviorModel vbModel;

  std::shared_ptr<Lane> nextSideLane;

 public:
  Vehicle(const std::shared_ptr<Agent> &_agent);

  void updateAcceleration(double timestep);
  void updateVelocity(double timestep);
  std::shared_ptr<Agent> moveVehicle(double timestep, double simTime);

  bool laneChangeJudge();
  bool carFollowingJudge();

  void getObstacle();
  double getHeadway();

  bool isSignalStop() { return vehicleObstacle.signal.second != nullptr; }

  void setStationDistance(double dist) {
    vehicleObstacle.stationDistance = dist;
  }
  double getLength() { return length; }
  double getWeight() { return weight; }
  void setWeight(double _weight) { weight = _weight; }
  bool isCarFollowing() { return carFollowing; }
};
}  // namespace Hongo

#endif /* Vehicle_hpp */
