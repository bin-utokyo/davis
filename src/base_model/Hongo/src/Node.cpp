//
//  Node.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/07.
//

#include "Node.hpp"

#include "Link.hpp"
#include "Utility.hpp"

namespace Hongo {

Node::Node(int _id, double _lon, double _lat) : id(_id), lon(_lon), lat(_lat) {
  std::tie(x, y) = Utility::lonLat2xy(lon, lat);
}
Node::Node(int _id, double _lon, double _lat, double _x, double _y)
    : id(_id), lon(_lon), lat(_lat), x(_x), y(_y) {}

void Node::addLink(const std::shared_ptr<Link> &link) {
  if (id == link->getONodeId()) {
    linkDirection.emplace(link->getDirection(), link);
  } else if (id == link->getDNodeId()) {
    linkDirection.emplace(Utility::oppositeDirection(link->getDirection()),
                          link);
  }
  offset[link->getId()] = 0.0;
}

void Node::setOffset() {
  if (linkDirection.size() <= 1) {
    return;
  }
  std::vector<double> keyVec;
  for (const auto &pair : linkDirection) {
    keyVec.emplace_back(pair.first);
  }
  double d1 = keyVec.at(0);
  double d2 = keyVec.at(1);
  double d3 = keyVec.at(keyVec.size() - 1);
  double theta1 = std::min(d2 - d1, 2 * M_PI - (d2 - d1));  // 0~pi
  double theta2 = std::min(d3 - d1, 2 * M_PI - (d3 - d1));  // 0~pi
  double offset1 =
      linkDirection.at(d1)->getWidth() / 2. * std::abs(std::tan(theta1));
  double offset2 =
      linkDirection.at(d1)->getWidth() / 2. * std::abs(std::tan(theta2));

  if (theta1 > 30. / 180. * M_PI &&
      theta1 < 150. / 180. * M_PI) {  // sin(theta1)が十分大きいとき
    offset1 = (linkDirection.at(d1)->getWidth() * std::cos(theta1) +
               linkDirection.at(d2)->getWidth()) /
              2 / std::sin(theta1);
  }
  if (theta2 > 30. / 180. * M_PI && theta2 < 150. / 180. * M_PI) {
    offset2 = (linkDirection.at(d1)->getWidth() * std::cos(theta2) +
               linkDirection.at(d3)->getWidth()) /
              2 / std::sin(theta2);
  }
  offset[linkDirection.at(d1)->getId()] = std::max(
      {0.0, offset1, offset2});  // directionが一番小さいリンクのoffsetを埋める
  for (std::size_t i = 1; i < keyVec.size() - 1; i++) {
    d1 = keyVec.at(i);
    d2 = keyVec.at(i + 1);
    d3 = keyVec.at(i - 1);

    theta1 = std::min(d2 - d1, 2 * M_PI - (d2 - d1));
    theta2 = std::min(d1 - d3, 2 * M_PI - (d1 - d3));
    offset1 =
        linkDirection.at(d1)->getWidth() / 2. * std::abs(std::tan(theta1));
    offset2 =
        linkDirection.at(d1)->getWidth() / 2. * std::abs(std::tan(theta2));

    if (theta1 > 30. / 180. * M_PI && theta1 < 150. / 180. * M_PI) {
      offset1 = (linkDirection.at(d1)->getWidth() * std::cos(theta1) +
                 linkDirection.at(d2)->getWidth()) /
                2 / std::sin(theta1);
    }
    if (theta2 > 30. / 180. * M_PI && theta2 < 150. / 180. * M_PI) {
      offset2 = (linkDirection.at(d1)->getWidth() * std::cos(theta2) +
                 linkDirection.at(d3)->getWidth()) /
                2 / std::sin(theta2);
    }
    offset[linkDirection.at(d1)->getId()] = std::max({0.0, offset1, offset2});
  }
  d1 = keyVec.at(keyVec.size() - 1);
  d2 = keyVec.at(0);
  d3 = keyVec.at(keyVec.size() - 2);

  theta1 = std::min(d1 - d2, 2 * M_PI - (d1 - d2));
  theta2 = std::min(d1 - d3, 2 * M_PI - (d1 - d3));
  offset1 = linkDirection.at(d1)->getWidth() / 2. * std::abs(std::tan(theta1));
  offset2 = linkDirection.at(d1)->getWidth() / 2. * std::abs(std::tan(theta2));

  if (theta1 > 30. / 180. * M_PI && theta1 < 150. / 180. * M_PI) {
    offset1 = (linkDirection.at(d1)->getWidth() * std::cos(theta1) +
               linkDirection.at(d2)->getWidth()) /
              2 / std::sin(theta1);
  }
  if (theta2 > 30. / 180. * M_PI && theta2 < 150. / 180. * M_PI) {
    offset2 = (linkDirection.at(d1)->getWidth() * std::cos(theta2) +
               linkDirection.at(d3)->getWidth()) /
              2 / std::sin(theta2);
  }
  offset[linkDirection.at(d1)->getId()] = std::max(
      {0.0, offset1, offset2});  // directionが一番大きいリンクのoffsetを埋める
}
}  // namespace Hongo
