//
//  Physarum.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/25.
//

#ifndef Physarum_hpp
#define Physarum_hpp

#include <stdio.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Hongo {

class Link;
class PhysarumLink : public std::enable_shared_from_this<PhysarumLink> {
  std::shared_ptr<Link> link;

  double d = 1.0;  // コンダクティビティ
  double q = 0.0;  // 流量

  double oPot = 0.0;  // linkのoNodeのポテンシャル
  double dPot = 0.0;  // linkのdNodeのポテンシャル
 public:
  PhysarumLink(const std::shared_ptr<Link> &_link) : link(_link) {}

  double getD() { return d; }
  void setD(double _d) { d = _d; }
  double getQ() { return q; }
  void setQ(double _q) { q = _q; }

  std::shared_ptr<Link> getLink();
  double getPot(int nodeId);
  void setPot(int nodeId, double pot);
};
class PathCount : public std::enable_shared_from_this<PathCount> {
  // 経路数カウント用
  static const int maxCount = 50;
  int nodeId;
  int pathNum = 1;
  std::vector<std::weak_ptr<PathCount>> parents;  // 上流側の分岐
  std::vector<std::shared_ptr<PathCount>> children;  // 下流側のノードまたは末端
  std::vector<std::shared_ptr<PathCount>>
      nonFrontChildren;  // 他のparentにすでに登録されているノード
 public:
  PathCount(int _nodeId) : nodeId(_nodeId) {}
  PathCount(int _nodeId, std::shared_ptr<PathCount> parent) : nodeId(_nodeId) {
    parents.emplace_back(parent);
  }

  int getPathNum() { return pathNum; }
  void updatePathNum(int dNodeId) {
    if (nodeId == dNodeId) {
      pathNum = 1;
    } else {
      pathNum = 0;
      for (const auto &dcnt : children) {
        if (pathNum > maxCount) {
          break;
        }
        pathNum += dcnt->getPathNum();
      }
      for (const auto &dcnt : nonFrontChildren) {
        if (pathNum > maxCount) {
          break;
        }
        pathNum += dcnt->getPathNum();
      }
    }
    for (auto &parent : parents) {
      if (!parent.expired()) {
        parent.lock()->updatePathNum(dNodeId);
      }
    }
  }
  std::shared_ptr<PathCount> addChild(int _nodeId, int dNodeId) {
    // 末端の追加
    std::shared_ptr<PathCount> child =
        std::make_shared<PathCount>(_nodeId, shared_from_this());
    children.emplace_back(child);
    updatePathNum(dNodeId);
    return child;
  }
  void addChild(const std::shared_ptr<PathCount> &child, int dNodeId) {
    // 下流側のノードを追加
    child->addParent(weak_from_this());
    nonFrontChildren.emplace_back(child);
    updatePathNum(dNodeId);
  }
  bool haveChild(const std::shared_ptr<PathCount> &child) {
    if (children.size() == 0 && nonFrontChildren.size() == 0) {
      return false;
    } else {
      for (const auto &tmpChild : children) {
        if (tmpChild->getNodeId() == child->getNodeId()) {
          return true;
        }
      }
      for (const auto &tmpChild : nonFrontChildren) {
        if (tmpChild->getNodeId() == child->getNodeId()) {
          return true;
        }
      }
      for (const auto &tmpChild : children) {
        if (tmpChild->haveChild(child)) {
          return true;
        }
      }
      for (const auto &tmpChild : nonFrontChildren) {
        if (tmpChild->haveChild(child)) {
          return true;
        }
      }
      return false;
    }
  }
  void addParent(const std::weak_ptr<PathCount> &parent) {
    if (!haveChild(parent.lock())) {
      parents.emplace_back(parent);
    }
  }
  std::vector<std::shared_ptr<PathCount>> getFront() {
    std::vector<std::shared_ptr<PathCount>> front;
    if (children.size() == 0 &&
        nonFrontChildren.size() == 0) {  // 自分が末端ノード
      front.emplace_back(shared_from_this());
    } else {
      for (const auto &child : children) {
        for (const auto &grandchild : child->getFront()) {
          front.emplace_back(grandchild);
        }
      }
    }
    return front;
  }

  const int getMaxCount() { return maxCount; }
  int getNodeId() { return nodeId; }
};
class Physarum {
  double q0 = 100;
  double thresh = 0.5;  // 探索範囲から外すリンクのコンダクティビティの閾値
  std::unordered_map<int, std::vector<std::shared_ptr<Link>>>
      odLinkMap;  // key: odNodeID, value: 繋がっているリンクのlinkID
 public:
  Physarum() {}

  void setLinkMap(
      const std::unordered_map<int, std::shared_ptr<Link>> &_linkMap);
  std::vector<std::shared_ptr<Link>> getPathLinks(
      int oNodeId, int dNodeId, int mode,
      int pathNum) const;  // costType  0: distance, 1: duration
  int countPathNum(
      int oNodeId, int dNodeId, double maxD, int mode,
      const std::unordered_map<int, std::vector<std::shared_ptr<PhysarumLink>>>
          &physarumLinkMap) const;

  bool linkJudge(std::shared_ptr<Link> nextLink, std::shared_ptr<Link> tmpLink,
                 int nodeId, int mode,
                 const std::unordered_set<int> &cutLinks) const;
  double getCost(std::shared_ptr<Link> link,
                 int costType) const;  // costType  0: distance, 1: duration
};
}  // namespace Hongo

#endif /* Physarum_hpp */
