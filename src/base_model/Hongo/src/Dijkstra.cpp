//
//  Dijkstra.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/21.
//

#include "Dijkstra.hpp"

#include "Link.hpp"
#include "Station.hpp"

namespace Hongo {

void Dijkstra::setLinkMap(
    const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap) {
  for (const auto &pair : _linkMap) {
    odLinkMap[pair.second->getONodeId()].emplace_back(pair.second);
    odLinkMap[pair.second->getDNodeId()].emplace_back(pair.second);
  }
}

std::pair<double, std::vector<std::shared_ptr<Link>>> Dijkstra::getShortestPath(
    int oNodeId, int dNodeId, int mode, const std::unordered_set<int> &cutLinks,
    bool bus) const {
  std::vector<std::shared_ptr<Link>> path;
  int costType;
  if (mode == 3) {
    costType = 0;
  } else if (mode == -1) {
    costType = 2;
  } else if (mode == 6) {
    costType = 3;
  } else {
    costType = 1;
  }

  if (odLinkMap.find(dNodeId) == odLinkMap.end()) {
    return std::make_pair(0.0, path);
  }

  std::priority_queue<PathNode, std::vector<PathNode>, std::greater<PathNode>>
      minHeap;  //(cost, Link)
  std::unordered_map<int, PathNode>
      determinedNode;  // key: nodeID, value: (id, cost, prevLink)

  if (odLinkMap.find(oNodeId) != odLinkMap.end()) {
    for (const auto &link : odLinkMap.at(oNodeId)) {
      if (link->isPassable(oNodeId, mode, bus) &&
          cutLinks.find(link->getId()) == cutLinks.end()) {
        PathNode pNode = {link->getAnotherNodeId(oNodeId),
                          getCost(link, costType), link};
        minHeap.push(pNode);
      }
    }
  }
  while (!minHeap.empty()) {
    auto pNode = minHeap.top();
    minHeap.pop();

    int nodeId = pNode.nodeId;
    if (determinedNode.find(nodeId) != determinedNode.end()) {
      continue;
    }

    determinedNode[nodeId] = pNode;  // コストが決定されるノード
    for (const auto &link : odLinkMap.at(nodeId)) {
      if (linkJudge(link, pNode.prevLink, nodeId, mode, cutLinks, bus) &&
          determinedNode.find(link->getAnotherNodeId(nodeId)) ==
              determinedNode.end()) {
        PathNode nextPNode = {link->getAnotherNodeId(nodeId),
                              pNode.cost + getCost(link, costType), link};
        minHeap.push(nextPNode);
      }
    }

    if (nodeId == dNodeId) {
      int tmpNodeId = dNodeId;
      while (tmpNodeId != oNodeId) {
        path.emplace_back(determinedNode.at(tmpNodeId).prevLink);
        tmpNodeId =
            determinedNode.at(tmpNodeId).prevLink->getAnotherNodeId(tmpNodeId);
      }
      std::reverse(path.begin(), path.end());
      break;
    }
  }
  if (determinedNode.find(dNodeId) == determinedNode.end()) {
    return std::make_pair(0.0, path);
  } else {
    return std::make_pair(determinedNode.at(dNodeId).cost, path);
  }
}

std::vector<std::shared_ptr<Station>> Dijkstra::getNearStations(int oNodeId) {
  double searchDist = 500;  // m
  int costType = 0;         // distance
  std::unordered_set<int> cutLinks;

  std::vector<std::shared_ptr<Station>> stationVec;

  std::priority_queue<PathNode, std::vector<PathNode>, std::greater<PathNode>>
      minHeap;  //(cost, Link)
  std::unordered_map<int, PathNode>
      determinedNode;  // key: nodeID, value: (id, cost, prevLink)

  if (odLinkMap.find(oNodeId) != odLinkMap.end()) {
    for (const auto &link : odLinkMap.at(oNodeId)) {
      if (link->isPassable(oNodeId, 3)) {
        PathNode pNode = {link->getAnotherNodeId(oNodeId),
                          getCost(link, costType), link};
        minHeap.push(pNode);
        for (const auto &pair : link->getStationMap()) {
          stationVec.emplace_back(pair.second);
        }
      }
    }
  }
  while (!minHeap.empty()) {
    auto pNode = minHeap.top();
    minHeap.pop();

    int nodeId = pNode.nodeId;
    if (determinedNode.find(nodeId) != determinedNode.end()) {
      continue;
    }

    determinedNode[nodeId] = pNode;  // コストが決定されるノード
    for (const auto &link : pNode.prevLink->getDnLinks(nodeId)) {
      if (linkJudge(link, pNode.prevLink, nodeId, 3, cutLinks) &&
          determinedNode.find(link->getAnotherNodeId(nodeId)) ==
              determinedNode.end()) {
        PathNode nextPNode = {link->getAnotherNodeId(nodeId),
                              pNode.cost + getCost(link, costType), link};
        for (const auto &pair : link->getStationMap()) {
          stationVec.emplace_back(pair.second);
        }
        if (nextPNode.cost > searchDist) {
          continue;
        }
        minHeap.push(nextPNode);
      }
    }
  }
  return stationVec;
}

bool Dijkstra::linkJudge(std::shared_ptr<Link> nextLink,
                         std::shared_ptr<Link> tmpLink, int nodeId, int mode,
                         const std::unordered_set<int> &cutLinks,
                         bool bus) const {
  bool passible = nextLink->isPassable(nodeId, mode, bus);
  bool notcut = cutLinks.find(nextLink->getId()) == cutLinks.end();
  bool connected = tmpLink->isConnected(nodeId, nextLink, mode);
  return passible && notcut && connected;
}

double Dijkstra::getCost(std::shared_ptr<Link> link, int costType) const {
  if (costType == 0) {  // distance
    return link->getLength();
  } else if (costType == 1) {  // duration
    return link->getLength() / std::max(0.1, link->getVelocity(0));
  } else if (costType == 2) {  // activity link
    return std::exp(link->getUtility(-1));
  } else if (costType == 3) {
    return link->getTravelTime();
  } else {
    return 1.0;
  }
}
}  // namespace Hongo
