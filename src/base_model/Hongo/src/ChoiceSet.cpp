//
//  ChoiceSet.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/18.
//

#include "ChoiceSet.hpp"

#include "Dijkstra.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "Node.hpp"
#include "Physarum.hpp"

namespace Hongo {

ChoiceSet::ChoiceSet(
    const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap,
    std::string _type) {
  linkMap = _linkMap;
  type = _type;
  if (type == "dijkstra") {
    dijkstra = std::make_shared<Dijkstra>();
    dijkstra->setLinkMap(_linkMap);
  } else if (type == "physarum") {
    physarum = std::make_shared<Physarum>();
    physarum->setLinkMap(_linkMap);
  }
}

std::unordered_map<int, std::shared_ptr<Link>> ChoiceSet::generateChoiceSet(
    int oNodeId, int dNodeId, int mode, bool bus) {
  std::unordered_set<int> linkIdSet;
  if (type == "dijkstra") {
    linkIdSet = generateChoiceSet_dijkstra(oNodeId, dNodeId, mode, bus);
  } else if (type == "physarum") {
    linkIdSet = generateChoiceSet_physarum(oNodeId, dNodeId, mode);
  } else {
    return linkMap;
  }

  std::unordered_map<int, std::shared_ptr<Link>> resultLinkMap;
  for (const auto &linkId : linkIdSet) {
    resultLinkMap.emplace(linkId, linkMap.at(linkId));
  }
  return resultLinkMap;
}

std::unordered_map<int, std::shared_ptr<Link>> ChoiceSet::generateChoiceSet(
    const std::shared_ptr<Lane> &tmpLane, int dNodeId, int mode, bool bus) {
  // コスト最小経路のlinkを取得
  std::unordered_set<int> linkIdSet =
      generateChoiceSet_dijkstra(tmpLane, dNodeId, mode, bus);

  std::unordered_map<int, std::shared_ptr<Link>> resultLinkMap;
  for (const auto &linkId : linkIdSet) {
    resultLinkMap.emplace(linkId, linkMap.at(linkId));
  }
  return resultLinkMap;
}
std::unordered_map<int, std::shared_ptr<Link>> ChoiceSet::generateChoiceSet(
    int oNodeId, const std::shared_ptr<Lane> &nextLane, int mode, bool bus) {
  // コスト最小経路のlinkを取得
  std::unordered_set<int> linkIdSet =
      generateChoiceSet_dijkstra(oNodeId, nextLane, mode, bus);

  std::unordered_map<int, std::shared_ptr<Link>> resultLinkMap;
  for (const auto &linkId : linkIdSet) {
    resultLinkMap.emplace(linkId, linkMap.at(linkId));
  }
  return resultLinkMap;
}

std::unordered_map<int, std::shared_ptr<Link>> ChoiceSet::generateChoiceSet(
    const std::shared_ptr<Lane> &tmpLane, const std::shared_ptr<Lane> &nextLane,
    int mode, bool bus) {
  // コスト最小経路のlinkを取得
  std::unordered_set<int> linkIdSet =
      generateChoiceSet_dijkstra(tmpLane, nextLane, mode, bus);

  std::unordered_map<int, std::shared_ptr<Link>> resultLinkMap;
  for (const auto &linkId : linkIdSet) {
    resultLinkMap.emplace(linkId, linkMap.at(linkId));
  }
  return resultLinkMap;
}

std::unordered_set<int> ChoiceSet::generateChoiceSet_dijkstra(int oNodeId,
                                                              int dNodeId,
                                                              int mode,
                                                              bool bus) {
  std::unordered_set<int> linkSet;
  std::unordered_set<int> cutLinks;
  int prevCutLinkId = -1;
  for (int i = 0; i < k; i++) {
    std::pair<double, std::vector<std::shared_ptr<Link>>> path =
        dijkstra->getShortestPath(oNodeId, dNodeId, mode, cutLinks, bus);
    if (path.second.size() == 0) {
      if (prevCutLinkId >= 0) {
        cutLinks.erase(prevCutLinkId);
        prevCutLinkId = -1;
      }
      continue;
    }
    for (const auto &link : path.second) {
      linkSet.emplace(link->getId());
    }
    prevCutLinkId = path.second[std::min((int)path.second.size() - 1,
                                         (int)path.second.size() * (i + 1) / k)]
                        ->getId();
    cutLinks.emplace(prevCutLinkId);
  }
  return linkSet;
}

std::unordered_set<int> ChoiceSet::generateChoiceSet_physarum(int oNodeId,
                                                              int dNodeId,
                                                              int mode) {
  std::unordered_set<int> linkSet;
  std::vector<std::shared_ptr<Link>> linkVec =
      physarum->getPathLinks(oNodeId, dNodeId, mode, k);
  for (const auto &link : linkVec) {
    linkSet.emplace(link->getId());
  }
  return linkSet;
}

std::unordered_set<int> ChoiceSet::generateChoiceSet_dijkstra(
    const std::shared_ptr<Lane> &tmpLane, int dNodeId, int mode, bool bus) {
  std::unordered_set<int> linkSet;
  std::unordered_set<int> cutLinks = {};
  if (tmpLane->getType() == "car" || tmpLane->getType() == "bus") {
    cutLinks = {tmpLane->getLink()->getId()};
  }
  int oNodeId = tmpLane->getDNodeId();
  std::pair<double, std::vector<std::shared_ptr<Link>>> path =
      dijkstra->getShortestPath(oNodeId, dNodeId, mode, cutLinks, bus);
  if (path.second.size() == 0) {
    return linkSet;
  }
  for (const auto &link : path.second) {
    linkSet.emplace(link->getId());
  }
  linkSet.emplace(tmpLane->getLink()->getId());
  return linkSet;
}
std::unordered_set<int> ChoiceSet::generateChoiceSet_dijkstra(
    int oNodeId, const std::shared_ptr<Lane> &nextLane, int mode, bool bus) {
  // stationLaneは一方通行を想定する
  std::unordered_set<int> linkSet;
  std::unordered_set<int> cutLinks = {};
  if (nextLane->getType() == "car" || nextLane->getType() == "bus") {
    cutLinks = {nextLane->getLink()->getId()};
  }
  int dNodeId = nextLane->getONodeId();
  std::pair<double, std::vector<std::shared_ptr<Link>>> path =
      dijkstra->getShortestPath(oNodeId, dNodeId, mode, cutLinks, bus);
  if (path.second.size() == 0) {
    linkSet.emplace(nextLane->getLink()->getId());
    return linkSet;
  }
  for (const auto &link : path.second) {
    linkSet.emplace(link->getId());
  }
  linkSet.emplace(nextLane->getLink()->getId());
  return linkSet;
}

std::unordered_set<int> ChoiceSet::generateChoiceSet_dijkstra(
    const std::shared_ptr<Lane> &tmpLane, const std::shared_ptr<Lane> &nextLane,
    int mode, bool bus) {
  // tmpLane, stationLaneは一方通行を想定する
  std::unordered_set<int> linkSet;
  std::unordered_set<int> cutLinks = {};
  if (tmpLane->getType() == "car" || tmpLane->getType() == "bus") {
    cutLinks = {nextLane->getLink()->getId(), tmpLane->getLink()->getId()};
  }
  int oNodeId = tmpLane->getDNodeId();
  int dNodeId = nextLane->getONodeId();
  std::pair<double, std::vector<std::shared_ptr<Link>>> path =
      dijkstra->getShortestPath(oNodeId, dNodeId, mode, cutLinks, bus);
  if (path.second.size() == 0) {
    return linkSet;
  }
  for (const auto &link : path.second) {
    linkSet.emplace(link->getId());
  }
  linkSet.emplace(tmpLane->getLink()->getId());
  linkSet.emplace(nextLane->getLink()->getId());
  return linkSet;
}

std::pair<double, std::vector<std::shared_ptr<Link>>>
ChoiceSet::getShortestPath(int oNodeId, int dNodeId, int mode, bool bus) {
  std::unordered_set<int> cutLinks;
  return dijkstra->getShortestPath(oNodeId, dNodeId, mode, cutLinks, bus);
}

void ChoiceSet::resetDijkstra() {
  linkMap.clear();
  dijkstra = nullptr;
}
}  // namespace Hongo
