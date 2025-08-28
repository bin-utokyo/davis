//
//  Physarum.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/25.
//

#include "Physarum.hpp"

#include "Link.hpp"
#include "SparseMat.hpp"

namespace Hongo {

std::shared_ptr<Link> PhysarumLink::getLink() { return link; }
double PhysarumLink::getPot(int nodeId) {
  if (link->getONodeId() == nodeId) {
    return oPot;
  } else if (link->getDNodeId() == nodeId) {
    return dPot;
  } else {
    return 0.0;
  }
}
void PhysarumLink::setPot(int nodeId, double pot) {
  if (link->getONodeId() == nodeId) {
    oPot = pot;
  } else if (link->getDNodeId() == nodeId) {
    dPot = pot;
  }
}
void Physarum::setLinkMap(
    const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap) {
  for (const auto &pair : _linkMap) {
    odLinkMap[pair.second->getONodeId()].emplace_back(pair.second);
    odLinkMap[pair.second->getDNodeId()].emplace_back(pair.second);
  }
}

std::vector<std::shared_ptr<Link>> Physarum::getPathLinks(int oNodeId,
                                                          int dNodeId, int mode,
                                                          int pathNum) const {
  std::unordered_map<int, double>
      potMap;  // key: nodeID, value: potential(dNodeのポテンシャルは0)
  std::unordered_map<int, int> nodeIndexMap;  // key: nodeID, value: index
  std::vector<int> nodeIdVec;                 // indexのnodeID
  std::unordered_map<int, std::vector<std::shared_ptr<PhysarumLink>>>
      physarumLinkMap;  // key: nodeID, value: vector(PhysarumLink)
  std::unordered_map<int, std::shared_ptr<PhysarumLink>>
      tmpPLinkMap;  // key: linkID, value: PhysarumLink

  int costType;
  if (mode == 3) {
    costType = 0;
  } else if (mode == -1) {
    costType = 2;
  } else {
    costType = 1;
  }

  int index = 0;
  for (const auto &pair : odLinkMap) {
    bool usedNode = false;

    for (const auto &link : pair.second) {
      if (link->isPassable(pair.first, mode) ||
          link->isPassable(link->getAnotherNodeId(pair.first), mode)) {
        std::shared_ptr<PhysarumLink> pLink;
        if (tmpPLinkMap.find(link->getId()) == tmpPLinkMap.end()) {
          tmpPLinkMap[link->getId()] = std::make_shared<PhysarumLink>(link);
          pLink = tmpPLinkMap.at(link->getId());
        } else {
          pLink = tmpPLinkMap.at(link->getId());
        }
        physarumLinkMap[pair.first].emplace_back(pLink);
        usedNode = true;
      }
    }
    if (pair.first == dNodeId) {
      continue;
    }
    if (usedNode) {
      potMap[pair.first] = 0;
      nodeIndexMap[pair.first] = index;
      nodeIdVec.emplace_back(pair.first);
      index++;
    }
  }
  double maxD = 0.0;
  int tmpPathNum = countPathNum(oNodeId, dNodeId, maxD, mode, physarumLinkMap);
  while (tmpPathNum > pathNum) {
    SparseMat<double> A(nodeIndexMap.size(), nodeIndexMap.size());
    std::vector<double> b(nodeIndexMap.size(), 0.0);

    for (const auto &pair : physarumLinkMap) {
      int oIndex = -1;
      if (nodeIndexMap.find(pair.first) != nodeIndexMap.end()) {
        oIndex = nodeIndexMap.at(pair.first);
      } else {
      }
      for (const auto &pLink : pair.second) {
        int nextNodeId = pLink->getLink()->getAnotherNodeId(pair.first);
        if (nodeIndexMap.find(nextNodeId) == nodeIndexMap.end()) {
          continue;
        }
        int dIndex = nodeIndexMap.at(nextNodeId);
        if (oIndex != -1) {
          A.add_at(dIndex, oIndex,
                   pLink->getD() / getCost(pLink->getLink(), costType));
        }
        A.add_at(dIndex, dIndex,
                 -pLink->getD() / getCost(pLink->getLink(), costType));
      }
    }
    b.at(nodeIndexMap.at(oNodeId)) = -q0;

    std::vector<double> p = SparseMat<double>::linsolve(A, b);
    if (std::isinf(p[0])) {
      break;
    }
    maxD = 0.0;
    int tmpONodeId, tmpDNodeId;
    double oPot, dPot;
    for (const auto &pair : tmpPLinkMap) {
      oPot = 0.0;
      dPot = 0.0;
      tmpONodeId = pair.second->getLink()->getONodeId();
      tmpDNodeId = pair.second->getLink()->getDNodeId();
      if (nodeIndexMap.find(tmpONodeId) != nodeIndexMap.end()) {
        oPot = p.at(nodeIndexMap.at(tmpONodeId));
      }
      if (nodeIndexMap.find(tmpDNodeId) != nodeIndexMap.end()) {
        dPot = p.at(nodeIndexMap.at(tmpDNodeId));
      }
      if (oPot - dPot == 0) {
        pair.second->setPot(tmpONodeId, oPot);
        pair.second->setPot(tmpDNodeId, dPot);
        pair.second->setQ(0);
        pair.second->setD(0);
      } else {
        pair.second->setPot(tmpONodeId, oPot);
        pair.second->setPot(tmpDNodeId, dPot);
        pair.second->setQ(std::abs(pair.second->getD() /
                                   getCost(pair.second->getLink(), costType) *
                                   (oPot - dPot)));
        pair.second->setD(pair.second->getQ());
      }
      maxD = std::max(maxD, pair.second->getD());
    }
    tmpPathNum = countPathNum(oNodeId, dNodeId, maxD, mode, physarumLinkMap);
  }
  std::vector<std::shared_ptr<Link>> path;
  if (tmpPathNum > 0) {
    for (const auto &pair : tmpPLinkMap) {
      if (pair.second->getD() >= maxD * thresh) {
        path.emplace_back(pair.second->getLink());
      }
    }
  }
  return path;
}

bool Physarum::linkJudge(std::shared_ptr<Link> nextLink,
                         std::shared_ptr<Link> tmpLink, int nodeId, int mode,
                         const std::unordered_set<int> &cutLinks) const {
  bool passible = nextLink->isPassable(nodeId, mode);
  bool notcut = cutLinks.find(nextLink->getId()) == cutLinks.end();
  return passible && notcut;
}

int Physarum::countPathNum(
    int oNodeId, int dNodeId, double maxD, int mode,
    const std::unordered_map<int, std::vector<std::shared_ptr<PhysarumLink>>>
        &physarumLinkMap) const {
  if (physarumLinkMap.find(oNodeId) == physarumLinkMap.end()) {
    return 0;
  }
  std::shared_ptr<PathCount> root = std::make_shared<PathCount>(oNodeId);
  std::unordered_map<int, std::shared_ptr<PathCount>>
      openNode;                           // 一度通ったノード
  std::unordered_set<int> prevFrontNode;  // 一度frontNodeとして使ったノード
  openNode.emplace(oNodeId, root);
  std::vector<std::shared_ptr<PathCount>> fronts = root->getFront();
  while (fronts.size() > 0 && root->getPathNum() < root->getMaxCount()) {
    for (const auto &front : fronts) {
      prevFrontNode.insert(front->getNodeId());
      for (const auto &pLink : physarumLinkMap.at(front->getNodeId())) {
        if (!pLink->getLink()->isPassable(front->getNodeId(), mode)) {
          continue;
        }
        int nextNodeId = pLink->getLink()->getAnotherNodeId(front->getNodeId());
        if (pLink->getPot(nextNodeId) >
            pLink->getPot(
                front
                    ->getNodeId())) {  // 逆行するような方向ではchildを追加しない
          continue;
        }
        if (pLink->getD() < maxD * thresh) {  // 使わないリンクの場合
          continue;
        }
        if (openNode.find(nextNodeId) == openNode.end()) {
          auto nextNode = front->addChild(nextNodeId, dNodeId);
          openNode[nextNodeId] = nextNode;
        } else {
          front->addChild(openNode.at(nextNodeId), dNodeId);
        }
      }
    }
    fronts = root->getFront();
    auto itr =
        std::remove_if(fronts.begin(), fronts.end(), [&](const auto &front) {
          return prevFrontNode.find(front->getNodeId()) !=
                     prevFrontNode.end() ||
                 front->getNodeId() == dNodeId;
        });
    fronts.erase(itr, fronts.end());
  }
  return root->getPathNum();
}

double Physarum::getCost(std::shared_ptr<Link> link, int costType) const {
  if (costType == 0) {  // distance
    return link->getLength();
  } else if (costType == 1) {  // duration
    return link->getLength() / std::max(0.1, link->getVelocity(0));
  } else if (costType == 2) {  // activity link
    return std::exp(link->getUtility(-1));
  } else {
    return 1.0;
  }
}
}  // namespace Hongo
