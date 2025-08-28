//
//  RL.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/28.
//

#include "RL.hpp"

#include "ChoiceSet.hpp"
#include "Dijkstra.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "SparseMat.hpp"
#include "Station.hpp"
#include "Utility.hpp"

namespace Hongo {

RL::RL(std::unordered_map<int, std::shared_ptr<Link>> _linkMap, int _mode)
    : mode(_mode) {
  if (mode == -1) {
    choiceSet = std::make_shared<ChoiceSet>(_linkMap, "");
  } else {
    choiceSet = std::make_shared<ChoiceSet>(_linkMap, "dijkstra");
  }
  // choiceSet = std::make_shared<ChoiceSet>(_linkMap, "physarum");
}

double RL::getLaneUtility(const std::shared_ptr<Lane> &lane, int nodeId) {
  // laneのnode側に繋がっているリンクの効用のexpのlogsum
  double maxUtility = -INFINITY;
  std::unordered_map<int, double> dnLinkMap;  // key: linkID, value: 効用
  for (const auto &nodeLane : lane->getDnLane(nodeId)) {
    const auto &dnLink = nodeLane->getDnLink();
    if (dnLinkMap.find(dnLink->getId()) == dnLinkMap.end()) {
      dnLinkMap[dnLink->getId()] =
          lane->getLink()->getConnectionUtility(dnLink->getId(), nodeId, mode) +
          dnLink->getUtility(mode) + beta * getVValue(dnLink, nodeId);
      maxUtility = std::max(maxUtility, dnLinkMap.at(dnLink->getId()));
    }
  }
  double v = std::accumulate(
      dnLinkMap.begin(), dnLinkMap.end(), 0, [&](double acc, const auto pair) {
        double diff = (std::isinf(pair.second) && std::isinf(maxUtility))
                          ? -INFINITY
                          : pair.second - maxUtility;
        return acc + std::exp(diff / mu);
      });
  return maxUtility + log(v);
}

std::shared_ptr<Link> RL::chooseFirstLink(int nodeId) {
  // nodeIdは上流側のノード
  std::shared_ptr<Link> nextLink = nullptr;
  if (odLinkMap.find(nodeId) == odLinkMap.end()) {
    return nextLink;
  }

  std::unordered_map<int, double>
      utilities;  // key:下流リンクのlinkID, value:exp(sum(V))
  double usum = 0.0;
  double maxUtility = -INFINITY;
  for (const auto &l : odLinkMap.at(nodeId)) {
    if (l->isPassable(nodeId, mode, bus) &&
        l->getONodeId() != l->getDNodeId()) {
      utilities[l->getId()] = l->getUtility(mode) + beta * getVValue(l, nodeId);
      maxUtility = std::max(maxUtility, utilities.at(l->getId()));
    }
  }
  for (auto &pair : utilities) {  // expのオーバーフロー防止
    pair.second = (std::isinf(pair.second) && std::isinf(maxUtility))
                      ? -INFINITY
                      : pair.second - maxUtility;
    usum += std::exp(pair.second / mu);
  }
  if (!utilities.empty()) {
    // 乱数を用いて選択
    double rnd = Utility::getRandRange01();
    if (abs(usum) > 0) {
      double bound = 0;
      for (const auto &u : utilities) {
        bound += std::exp(u.second / mu) / usum;
        if (rnd <= bound) {
          nextLink = linkMap.at(u.first);
          break;
        }
      }
    } else {
      nextLink = linkMap.at(utilities.begin()->first);
    }
  } else {
    std::cerr << "Utility is empty." << std::endl;
  }
  path.emplace_back(nextLink);
  return nextLink;
}

std::shared_ptr<Lane> RL::chooseFirstLane(const std::shared_ptr<Link> &nextLink,
                                          int nodeId) {
  std::vector<std::shared_ptr<Lane>> lanes;
  if (mode == 3) {  // 歩行者
    lanes = nextLink->getLanes(0);
  } else if (nodeId == nextLink->getONodeId()) {
    lanes = nextLink->getLanes(1);
  } else if (nodeId == nextLink->getDNodeId()) {
    lanes = nextLink->getLanes(-1);
  }

  std::shared_ptr<Lane> nextLane = nullptr;
  std::unordered_map<int, std::shared_ptr<Lane>> laneMap;
  std::unordered_map<int, double>
      utilities;  // key:車線変更レーンのlaneID, value:exp(sum(V))
  double usum = 0.0;
  double maxUtility = -INFINITY;

  for (const auto &lane : lanes) {
    if (lane->isPassable(nodeId, mode, bus)) {
      laneMap.emplace(lane->getId(), lane);
      utilities[lane->getId()] =
          lane->getLink()->getUtility(mode) +
          beta *
              getLaneUtility(lane, lane->getLink()->getAnotherNodeId(nodeId));
      maxUtility = std::max(maxUtility, utilities.at(lane->getId()));
    }
  }
  for (auto &pair : utilities) {  // expのオーバーフロー防止
    pair.second = (std::isinf(pair.second) && std::isinf(maxUtility))
                      ? -INFINITY
                      : pair.second - maxUtility;
    usum += std::exp(pair.second / mu);
  }
  if (!utilities.empty()) {
    // 乱数を用いて選択
    double rnd = Utility::getRandRange01();
    if (abs(usum) > 0) {
      double bound = 0;
      for (const auto &u : utilities) {
        bound += std::exp(u.second / mu) / usum;
        if (rnd <= bound) {
          nextLane = laneMap.at(u.first);
          break;
        }
      }
    } else {
      nextLane = laneMap.at(utilities.begin()->first);
    }
  } else {
    std::cerr << "Utility is empty." << std::endl;
  }
  return nextLane;
}
std::shared_ptr<Link> RL::chooseDnLink(const std::shared_ptr<Link> &link,
                                       const std::shared_ptr<Lane> &lane,
                                       int nodeId) {
  std::shared_ptr<Link> nextLink = nullptr;
  // 終点に到着した場合
  if (dLane == nullptr && nodeId == dNodeId) {
    return nextLink;
  } else if (dLane != nullptr) {
    if (dLane->getLink()->getId() == link->getId()) {
      return nextLink;
    }
  }
  // 下流リンクを探索
  std::unordered_map<int, double>
      utilities;  // key:下流リンクのlinkID, value:v(a|k)+V(a)
  double usum = 0.0;
  double maxUtility = -INFINITY;

  for (const auto &a : link->getDnLinks(nodeId)) {
    if (linkMap.find(a->getId()) != linkMap.end() &&
        lane->isConnected(nodeId, a)) {
      utilities[a->getId()] =
          link->getConnectionUtility(a->getId(), nodeId, mode) +
          a->getUtility(mode) + beta * getVValue(a, nodeId);
      maxUtility = std::max(maxUtility, utilities.at(a->getId()));
    }
  }
  for (auto &pair : utilities) {  // expのオーバーフロー防止
    pair.second = (std::isinf(pair.second) && std::isinf(maxUtility))
                      ? -INFINITY
                      : pair.second - maxUtility;
    usum += std::exp(pair.second / mu);
  }

  if (!utilities.empty()) {
    // 乱数を用いて選択
    double rnd = Utility::getRandRange01();
    if (abs(usum) > 0) {
      double bound = 0;
      for (const auto &u : utilities) {
        bound += std::exp(u.second / mu) / usum;
        if (rnd <= bound) {
          nextLink = linkMap.at(u.first);
          break;
        }
      }
    } else {
      nextLink = linkMap.at(utilities.begin()->first);
    }
  } else {
    std::cerr << "Utility is empty." << std::endl;
  }
  path.emplace_back(nextLink);
  return nextLink;
}
std::shared_ptr<NodeLane> RL::chooseDnLane(
    const std::shared_ptr<Lane> &lane, const std::shared_ptr<Link> &nextLink,
    int nodeId) {
  std::shared_ptr<NodeLane> nextLane = nullptr;
  std::unordered_map<int, double>
      utilities;  // key:前方のレーンのlaneID, value:exp(sum(V))
  double usum = 0.0;
  double maxUtility = -INFINITY;
  std::unordered_map<int, std::shared_ptr<NodeLane>> nextLaneMap;

  int num = 0;
  for (const auto &dnNodeLane : lane->getDnLane(nodeId)) {
    if (dLane != nullptr) {
      // 終点に到着するレーンが指定されていて，そのレーンが下流にあった場合
      // もしない場合は，別のレーンに入ったのち，chooseSideLaneで移動
      if (dnNodeLane->getDnLane() == dLane) {
        nextLaneMap.clear();
        utilities.clear();  // 他のレーンの情報を削除
        nextLaneMap[0] = dnNodeLane;
        utilities[0] = 0.0;
        maxUtility = 0.0;
        break;
      }
    }
    if (dnNodeLane->getDnLink()->getId() == nextLink->getId()) {
      nextLaneMap[num] = dnNodeLane;
      utilities[num] = dnNodeLane->getUtility(mode) +
                       dnNodeLane->getDnLink()->getUtility(mode) +
                       beta * getVValue(dnNodeLane->getDnLink(), nodeId);
      maxUtility = std::max(maxUtility, utilities.at(num));
      num++;
    }
  }
  for (auto &pair : utilities) {  // expのオーバーフロー防止
    pair.second = (std::isinf(pair.second) && std::isinf(maxUtility))
                      ? -INFINITY
                      : pair.second - maxUtility;
    usum += std::exp(pair.second / mu);
  }

  if (!utilities.empty()) {
    // 乱数を用いて選択
    double rnd = Utility::getRandRange01();
    if (abs(usum) > 0) {
      double bound = 0;
      for (const auto &u : utilities) {
        bound += std::exp(u.second / mu) / usum;
        if (rnd <= bound) {
          nextLane = nextLaneMap.at(u.first);
          break;
        }
      }
    } else {
      nextLane = nextLaneMap.at(utilities.begin()->first);
    }
  } else {
    std::cerr << "Utility is empty." << std::endl;
  }
  return nextLane;
}

std::shared_ptr<Lane> RL::chooseSideLane(const std::shared_ptr<Lane> &lane,
                                         int nodeId, double position) {
  std::shared_ptr<Lane> nextLane = nullptr;
  if (dLane != nullptr) {
    if (lane->getLink()->getId() ==
        dLane->getLink()
            ->getId()) {  // 終点に到着するレーンが指定されていた場合はそのレーンに車線変更（or車線維持）
      nextLane = dLane;
      return nextLane;
    }
  }
  std::unordered_map<int, double>
      utilities;  // key:車線変更レーンのlaneID, value:exp(sum(V))
  double usum = 0.0;
  double maxUtility = -INFINITY;
  double laneChangeUtil =
      -1.0;  // 車線変更の効用（十分小さければ必要でない時以外は車線変更しない）
  std::unordered_map<int, std::shared_ptr<Lane>> sideLaneMap;
  for (const auto &sideLane : lane->getLink()->getLanes(lane->getSign())) {
    sideLaneMap[sideLane->getId()] = sideLane;
    if (sideLane->getId() == lane->getId()) {
      utilities[sideLane->getId()] =
          lane->getLink()->getUtility(mode) * (1 - position) +
          beta * getLaneUtility(sideLane, nodeId);
    } else {
      utilities[sideLane->getId()] =
          laneChangeUtil + lane->getLink()->getUtility(mode) * (1 - position) +
          beta * getLaneUtility(sideLane, nodeId);
    }
    maxUtility = std::max(maxUtility, utilities.at(sideLane->getId()));
  }
  for (auto &pair : utilities) {  // expのオーバーフロー防止
    pair.second = (std::isinf(pair.second) && std::isinf(maxUtility))
                      ? -INFINITY
                      : pair.second - maxUtility;
    usum += std::exp(pair.second / mu);
  }
  if (!utilities.empty()) {
    // 乱数を用いて選択
    double rnd = Utility::getRandRange01();
    if (abs(usum) > 0) {
      double bound = 0;
      for (const auto &u : utilities) {
        bound += std::exp(u.second / mu) / usum;
        if (rnd <= bound) {
          nextLane = sideLaneMap.at(u.first);
          break;
        }
      }
    } else {
      nextLane = sideLaneMap.at(utilities.begin()->first);
      if (nodeId != dNodeId) {
        std::cout << lane->getLink()->getId() << std::endl;
      }
    }
  } else {
    std::cerr << "Utility is empty." << std::endl;
  }
  return nextLane;
}

void RL::setDNodeId(int _oNodeId, int _dnodeId) {
  oNodeId = _oNodeId;
  dNodeId = _dnodeId;
  odLinkMap.clear();
  dLane = nullptr;

  linkMap = choiceSet->generateChoiceSet(oNodeId, dNodeId, mode, bus);

  for (const auto &pair : linkMap) {
    odLinkMap[pair.second->getONodeId()].emplace_back(pair.second);
    odLinkMap[pair.second->getDNodeId()].emplace_back(pair.second);
  }

  // calcV_bellman();
  calcV_dn(oNodeId);
}
void RL::setDNodeId(const std::shared_ptr<Lane> &_oLane, int _dnodeId) {
  oNodeId = _oLane->getONodeId();
  dNodeId = _dnodeId;
  odLinkMap.clear();
  dLane = nullptr;

  linkMap = choiceSet->generateChoiceSet(_oLane, dNodeId, mode, bus);

  for (const auto &pair : linkMap) {
    odLinkMap[pair.second->getONodeId()].emplace_back(pair.second);
    odLinkMap[pair.second->getDNodeId()].emplace_back(pair.second);
  }

  // calcV_bellman();
  calcV_dn(oNodeId);
}
void RL::setDLane(int _oNodeId, const std::shared_ptr<Lane> &_dLane) {
  oNodeId = _oNodeId;
  dNodeId = _dLane->getDNodeId();
  dLane = _dLane;
  odLinkMap.clear();

  linkMap = choiceSet->generateChoiceSet(oNodeId, dLane, mode, bus);

  for (const auto &pair : linkMap) {
    odLinkMap[pair.second->getONodeId()].emplace_back(pair.second);
    odLinkMap[pair.second->getDNodeId()].emplace_back(pair.second);
  }

  // calcV_bellman();
  calcV_dn(oNodeId);
}
void RL::setDLane(const std::shared_ptr<Lane> &_oLane,
                  const std::shared_ptr<Lane> &_dLane) {
  oNodeId = _oLane->getONodeId();
  dNodeId = _dLane->getDNodeId();
  dLane = _dLane;
  odLinkMap.clear();

  linkMap = choiceSet->generateChoiceSet(_oLane, dLane, mode);

  for (const auto &pair : linkMap) {
    odLinkMap[pair.second->getONodeId()].emplace_back(pair.second);
    odLinkMap[pair.second->getDNodeId()].emplace_back(pair.second);
  }

  // calcV_bellman();
  calcV_dn(oNodeId);
}

std::vector<std::shared_ptr<Station>> RL::getNearStations(int oNodeId) {
  return choiceSet->getDijkstra()->getNearStations(oNodeId);
}
std::pair<double, std::vector<std::shared_ptr<Link>>> RL::getShortestPath(
    int oNodeId, int dNodeId) {
  return choiceSet->getShortestPath(oNodeId, dNodeId, mode);
}

void RL::setPath(std::vector<std::shared_ptr<Link>> _path) {
  path = _path;
  pathMap.clear();
  for (size_t i = 0; i < path.size(); i++) {
    pathMap[path.at(i)->getId()] = i;
  }
  if (path.size() > 0) {
    oNodeId = path.at(0)->getONodeId();
    dNodeId = path.at(path.size() - 1)->getDNodeId();
  }
}

std::shared_ptr<Link> RL::getNextLink(int linkId) {
  if (pathMap.find(linkId) != pathMap.end()) {
    int index = pathMap.at(linkId);
    if ((size_t)index + 1 < path.size()) {
      return path.at(index + 1);
    }
  }
  return nullptr;
}
int RL::getNextNodeId(int tmpNodeId) {
  for (const auto &l : path) {
    if (l->isPassable(tmpNodeId, mode, bus)) {
      return l->getAnotherNodeId(tmpNodeId);
    }
  }
  return -1;
}

// 下流リンクの効用計算→次のリンクを選択 の場合
void RL::calcV_dn(int oNodeId) {
  V.clear();
  for (const auto &l : linkMap) {
    V[l.first] = std::make_pair(-INFINITY, -INFINITY);
  }
  std::unordered_map<int, std::pair<bool, bool>>
      active;  // 既に効用計算を行ったか（サイクリックな経路で計算が終わらなくなるのを防ぐため）
  for (const auto &l : linkMap) {
    active[l.first] = std::make_pair(false, false);
  }
  // 価値関数の更新
  if (odLinkMap.find(oNodeId) != odLinkMap.end()) {
    for (const auto &link : odLinkMap.at(oNodeId)) {
      if (linkMap.find(link->getId()) == linkMap.end()) {
        continue;
      }
      double dL = 100;
      int i = 0;
      while (dL > 0.001) {
        // Vehicle::laneより下流側のレーンの効用を更新
        dL = 0.0;
        recursive_dn(link, oNodeId, active, dL);

        if (std::isinf(getVValue(link, oNodeId))) {
          std::cout << "No path to reach the destination." << std::endl;
        }

        for (auto &a : active) {
          a.second = std::make_pair(false, false);
        }
        if (++i > 1000) {
          std::cout << "Too much iteration for V value calculation."
                    << std::endl;
          break;
        }
      }
    }
  }
}

void RL::recursive_dn(const std::shared_ptr<Link> &link, int nodeId,
                      std::unordered_map<int, std::pair<bool, bool>> &active,
                      double &dL) {
  /*
   calcZ内で使う再帰関数
   nodeIdはlinkの上流側のノード
   V(k)=log(sum_a(exp(v(a|k)+beta*V(a))))
   */
  if (dLane == nullptr &&
      link->getAnotherNodeId(nodeId) == dNodeId) {  // 終点に接続するリンク
    setVValue(link, nodeId, 0.0);
    return;
  } else if (dLane !=
             nullptr) {  // 目的地に到着するためのレーンが指定されている場合
    if (dLane->getLink()->getId() == link->getId() &&
        link->getAnotherNodeId(nodeId) == dNodeId) {
      setVValue(link, nodeId, 0.0);
      return;
    } else if (dLane->getLink()->getId() == link->getId() &&
               nodeId == dNodeId) {  // dlaneを逆走
      setVValue(link, nodeId, -INFINITY);
      return;
    }
  }
  if (nodeId == link->getONodeId()) {  // ONodeからDNodeに向かっている場合
    active.at(link->getId()).first = true;
  } else {
    active.at(link->getId()).second = true;
  }
  int anotherNodeId = link->getAnotherNodeId(nodeId);
  auto dnLinks = link->getDnLinks(anotherNodeId);  // 下流リンク
  std::vector<double> tmpVs;  // v(a|k)+beta*V(a) for a in A(k)
  for (const auto &a : dnLinks) {
    // vehicleのリンクマップにaが含まれる場合
    if (V.find(a->getId()) != V.end()) {
      if ((a->getONodeId() == anotherNodeId && !active[a->getId()].first) ||
          (a->getDNodeId() == anotherNodeId &&
           !active[a->getId()].second)) {  // 初めて見た場合
        // Zの値を更新する。
        recursive_dn(a, link->getAnotherNodeId(nodeId), active, dL);
      }
      tmpVs.emplace_back(
          link->getConnectionUtility(a->getId(), anotherNodeId, mode) +
          a->getUtility(mode) + beta * getVValue(a, anotherNodeId));
    }
  }
  if (tmpVs.empty())
    return;  // リンクの逆走などにより，下流リンクがlinkMapの中に見つからなかった場合
  double maxV = *std::max_element(tmpVs.begin(),
                                  tmpVs.end());  // 下流リンクの効用関数の最大値

  double v =
      std::accumulate(tmpVs.begin(), tmpVs.end(), 0, [&](double acc, double i) {
        double diff = (std::isinf(i) && std::isinf(maxV))
                          ? -INFINITY
                          : i - maxV;  // INFINITY-INFINITYがnanのため
        return acc + std::exp(diff / mu);  // expのオーバーフロー防止のため
      });
  v = std::log(v) + maxV;
  dL += std::abs(getVValue(link, nodeId) - v);
  setVValue(link, nodeId, v);
}

// ベルマン方程式解法
void RL::calcV_bellman() {
  /*
   Bellman eq. z=M(z^β)+b ただし、z^βはzの各要素をβ乗
   z(k)=exp(V(k)) V(k)は状態kにいて目的地まで行く際の期待効用
   M(k,a)=exp(V(a|k))
   V(a|k)は状態kから状態aに移る際の効用、リンクkの終点が目的地の場合は0 b(k)=0
   or 1 リンクkの終点が目的地なら1 状態kはリンクkの終点にいる状態
   */
  std::unordered_map<int, std::pair<double, double>>
      Z;  // key: linkID, value: exp(V(k) (first:oNodeからdNodeへ行く場合,
          // second:逆）
  std::vector<double> b(
      linkMap.size() * 2,
      0.0);  // l=linkMap.size()とした場合、b(index)がlinkの順方向、b(index+l)がlinkの逆方向に対応
  SparseMat<double> M(linkMap.size() * 2, linkMap.size() * 2);
  std::unordered_map<int, int> indexM;  // key: linkID, value: index
  std::vector<int> linkIndex;           // linkID
  int index = 0;
  for (const auto &l : linkMap) {
    Z[l.first] = std::make_pair(0.0, 0.0);
    // ベクトル演算で使うインデックスとlinkIDの対応をとる。
    indexM[l.first] = index;
    linkIndex.emplace_back(l.first);

    if (l.second->getDNodeId() == dNodeId) {  // 順方向終点がdNodeの場合
      b[index] = 1.0;
    } else if (l.second->getONodeId() == dNodeId) {  // 逆方向終点がdNodeの場合
      b[index + linkMap.size()] = 1.0;
    }
    index++;
  }
  for (const auto &l : linkMap) {  // Mの値の更新
    if (l.second->getDNodeId() != dNodeId) {
      for (const auto &dnLink :
           l.second->getDnLinks(l.second->getDNodeId())) {  // lの順方向
        if (indexM.find(dnLink->getId()) != indexM.end()) {
          if (dnLink->getONodeId() ==
              l.second->getDNodeId()) {  // dnLinkも順方向
            M.insert_at(
                indexM.at(l.first), indexM.at(dnLink->getId()),
                std::exp(dnLink->getUtility(mode) +
                         l.second->getConnectionUtility(
                             dnLink->getId(), l.second->getDNodeId(), mode)));
          } else if (dnLink->getDNodeId() ==
                     l.second->getDNodeId()) {  // dnLinkは逆方向
            M.insert_at(
                indexM.at(l.first), indexM.at(dnLink->getId()) + linkMap.size(),
                std::exp(dnLink->getUtility(mode) +
                         l.second->getConnectionUtility(
                             dnLink->getId(), l.second->getDNodeId(), mode)));
          }
        }
      }
    }
    if (l.second->getONodeId() != dNodeId) {
      for (const auto &dnLink :
           l.second->getDnLinks(l.second->getONodeId())) {  // lの逆方向
        if (indexM.find(dnLink->getId()) != indexM.end()) {
          if (dnLink->getONodeId() ==
              l.second->getONodeId()) {  // dnLinkは順方向
            M.insert_at(
                indexM.at(l.first) + linkMap.size(), indexM.at(dnLink->getId()),
                std::exp(dnLink->getUtility(mode) +
                         l.second->getConnectionUtility(
                             dnLink->getId(), l.second->getONodeId(), mode)));
          } else if (dnLink->getDNodeId() ==
                     l.second->getONodeId()) {  // dnLinkも逆方向
            M.insert_at(
                indexM.at(l.first) + linkMap.size(),
                indexM.at(dnLink->getId()) + linkMap.size(),
                std::exp(dnLink->getUtility(mode) +
                         l.second->getConnectionUtility(
                             dnLink->getId(), l.second->getONodeId(), mode)));
          }
        }
      }
    }
  }
  // ベルマン方程式の求解、Zの値代入
  double dL = 1;
  int cnt = 0;
  while (dL > 0.001) {  // 非線形なので、繰り返し計算
    std::unordered_map<int, std::pair<double, double>> prevZ = Z;
    // Az=yの形にして解く。
    // 現在のZの周りでテーラー展開。z+Δz=M(z^β+βz^(β-1)Δz)+b をΔzについて解く。
    //(I-Mβz^(β-1))Δz=M(z^β)+b-z
    SparseMat<double> A = SparseMat<double>::eye(linkMap.size() * 2);
    SparseMat<double> ZMat(linkMap.size() * 2, linkMap.size() * 2);
    SparseMat<double> ZVec(linkMap.size() * 2, 1);
    std::vector<double> y(linkMap.size() * 2);
    for (const auto &l : linkMap) {
      ZMat.insert_at(indexM.at(l.first), indexM.at(l.first),
                     std::exp(getVValue(l.second, l.second->getONodeId())));
      ZMat.insert_at(indexM.at(l.first) + linkMap.size(),
                     indexM.at(l.first) + linkMap.size(),
                     std::exp(getVValue(l.second, l.second->getDNodeId())));
      ZVec.insert_at(indexM.at(l.first), 0,
                     std::exp(getVValue(l.second, l.second->getONodeId())));
      ZVec.insert_at(indexM.at(l.first) + linkMap.size(), 0,
                     std::exp(getVValue(l.second, l.second->getDNodeId())));
    }
    A = A - beta * M * ZMat.power(beta - 1.0);

    SparseMat<double> Mzb = M * ZVec.power(beta);
    for (std::size_t i = 0; i < linkMap.size() * 2; i++) {
      y.at(i) = Mzb.at(i, 0) + b.at(i) - ZVec.at(i, 0);
    }
    std::vector<double> dz = SparseMat<double>::linsolve(A, y);
    dL = std::accumulate(
        dz.begin(), dz.end(), 0.0,
        [](double sum, double deltaz) { return sum + std::abs(deltaz); });
    for (const auto &l : linkMap) {
      double tmpZ1 =
          std::exp(getVValue(l.second, l.second->getONodeId())) +
          dz.at(indexM.at(
              l.first));  // リンクl.seocndのONodeからDNodeへ行く方向の価値関数
      double tmpZ2 =
          std::exp(getVValue(l.second, l.second->getDNodeId())) +
          dz.at(
              indexM.at(l.first) +
              linkMap
                  .size());  // リンクl.seocndのDNodeからONodeへ行く方向の価値関数
      setVValue(l.second, l.second->getONodeId(), std::log(tmpZ1));
      setVValue(l.second, l.second->getDNodeId(), std::log(tmpZ2));
    }
    if (++cnt > 100) {
      std::cout << "too many repetition." << std::endl;
      break;
    }
  }
  /*
  for (int i = 0; i<linkIdM.size(); i++){
      double rightValue = b.at(i);
      for (int j = 0; j<linkIdM.size(); j++){
          rightValue += M[i][j] * std::pow(Z.at(linkIdM.at(j)), beta);
      }
      std::cout<<Z.at(linkIdM.at(i))<<","<<rightValue<<std::endl;
  }
   */
}

double RL::getVValue(const std::shared_ptr<Link> &link, int nodeId) {
  // nodeIdは上流側のノード
  if (V.find(link->getId()) == V.end()) {
    return -INFINITY;
  } else if (nodeId == link->getONodeId()) {
    return V.at(link->getId()).first;
  } else if (nodeId == link->getDNodeId()) {
    return V.at(link->getId()).second;
  } else {
    return -INFINITY;
  }
}

void RL::setVValue(const std::shared_ptr<Link> &link, int nodeId,
                   double value) {
  double first, second;
  // 値の初期化
  if (V.find(link->getId()) != V.end()) {
    std::tie(first, second) = V.at(link->getId());
  } else {
    first = -INFINITY;
    second = -INFINITY;
  }
  // 新しい値をセット
  if (nodeId == link->getONodeId()) {
    first = value;
  } else if (nodeId == link->getDNodeId()) {
    second = value;
  }
  V[link->getId()] = std::make_pair(first, second);
}

void RL::resetDijkstra() { choiceSet->resetDijkstra(); }

void RL::trimDnLinks() {
  for (const auto &pair : linkMap) {
    std::vector<std::shared_ptr<Link>> dnLinks =
        pair.second->getDnLinks(pair.second->getONodeId());
    auto itr =
        std::remove_if(dnLinks.begin(), dnLinks.end(), [&](const auto &dl) {
          return linkMap.find(dl->getId()) == linkMap.end();
        });
    dnLinks.erase(itr, dnLinks.end());
    pair.second->setDnLinks(pair.second->getONodeId(), dnLinks);

    dnLinks = pair.second->getDnLinks(pair.second->getDNodeId());
    itr = std::remove_if(dnLinks.begin(), dnLinks.end(), [&](const auto &dl) {
      return linkMap.find(dl->getId()) == linkMap.end();
    });
    dnLinks.erase(itr, dnLinks.end());
    pair.second->setDnLinks(pair.second->getDNodeId(), dnLinks);
  }
}

void RL::setK(int _k) { choiceSet->setK(_k); }

}  // namespace Hongo
