//
//  SparseMat.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/25.
//

#ifndef SparseMat_hpp
#define SparseMat_hpp

#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define EPSILON DBL_EPSILON

namespace Hongo {

struct hash_pair {
  // unordered_mapでstd::pairをkeyとして使うため
  template <typename T1, typename T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const noexcept {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);

    std::size_t seed = 0;
    seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
template <typename T>
class SparseMat {
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, T>>
      valueMap;  // key: row  value: (key: col  value: value)
  std::size_t rowNum = 0;
  std::size_t colNum = 0;

 public:
  SparseMat() {}
  SparseMat(std::size_t _rowNum, std::size_t _colNum)
      : rowNum(_rowNum), colNum(_colNum) {}
  SparseMat(const std::vector<std::vector<T>> &values) {
    rowNum = values.size();
    for (std::size_t i = 0; i < rowNum; i++) {
      const std::vector<T> &tmpVec = values.at(i);
      for (std::size_t j = 0; j < tmpVec.size(); j++) {
        if (std::abs(tmpVec.at(j)) > EPSILON) {
          valueMap[i][j] = tmpVec.at(j);
        }
      }
      colNum = std::max(colNum, tmpVec.size());
    }
  }
  SparseMat(const SparseMat &sm)
      : valueMap(sm.getValueMap()), rowNum(sm.getRow()), colNum(sm.getCol()) {}

  static SparseMat<T> eye(std::size_t s) {
    SparseMat<T> result(s, s);
    for (std::size_t i = 0; i < s; i++) {
      result.insert_at(i, i, 1);
    }
    return result;
  }

  std::size_t getRow() const { return rowNum; }
  std::size_t getCol() const { return colNum; }

  const std::unordered_map<std::size_t, std::unordered_map<std::size_t, T>> &
  getValueMap() const {
    return valueMap;
  }

  void clear() {
    valueMap.clear();
    rowNum = 0;
    colNum = 0;
  }

  T at(std::size_t row, std::size_t col) const {
    const auto itr = valueMap.find(row);
    if (itr != valueMap.end()) {
      const auto itr2 = itr->second.find(col);
      if (itr2 != itr->second.end()) {
        return itr2->second;
      }
    }
    return 0.0;
  }
  void insert_at(std::size_t row, std::size_t col, const T &value) {
    if (std::abs(value) > EPSILON) {
      valueMap[row][col] = value;
    } else {
      auto itr = valueMap.find(row);
      if (itr != valueMap.end()) {
        const auto itr2 = itr->second.find(col);
        if (itr2 != itr->second.end()) {
          itr->second.erase(itr2);
        }
      }
    }
    rowNum = std::max(row + 1, rowNum);
    colNum = std::max(col + 1, colNum);
  }
  void add_at(std::size_t row, std::size_t col, const T &value) {
    T v = at(row, col) + value;
    insert_at(row, col, v);
  }
  void swap_row(std::size_t row1, std::size_t row2) {
    if (row1 == row2) {
      return;
    }
    auto itr1 = valueMap.find(row1);
    auto itr2 = valueMap.find(row2);
    if (itr1 != valueMap.end()) {
      if (itr2 != valueMap.end()) {
        itr1->second.swap(itr2->second);
      } else {
        valueMap[row2] = itr1->second;
        valueMap.erase(itr1);
      }
    } else if (itr2 != valueMap.end()) {
      valueMap[row1] = itr2->second;
      valueMap.erase(itr2);
    }
  }

  void rowAddition(std::size_t row1, std::size_t row2, T m) {
    // 行1に行2のm倍を足す
    auto itr2 = valueMap.find(row2);
    if (itr2 == valueMap.end()) {
      return;
    }
    T v;
    for (const auto &pair : itr2->second) {
      v = pair.second * m + at(row1, pair.first);
      insert_at(row1, pair.first, v);
    }
  }

  SparseMat<T> add(const SparseMat<T> &A) const {
    SparseMat<T> result(A);
    for (const auto &pair : valueMap) {
      for (const auto &pair2 : pair.second) {
        result.add_at(pair.first, pair2.first, pair2.second);
      }
    }
    return result;
  }

  SparseMat<T> minus() const {
    SparseMat<T> result(rowNum, colNum);
    for (const auto &pair : valueMap) {
      for (const auto &pair2 : pair.second) {
        result.insert_at(pair.first, pair2.first, -pair2.second);
      }
    }
    return result;
  }

  SparseMat<T> minus(const SparseMat<T> &A) const {  // self - A
    return add(A.minus());
  }

  SparseMat<T> multiplyRight(const SparseMat<T> &A) const {  // self * A
    if (colNum != A.getRow()) {
      std::cerr << "Not matching matrix size." << std::endl;
      return SparseMat<T>();
    } else {
      SparseMat<T> result(rowNum, A.getCol());
      const auto &valueMapA = A.getValueMap();
      for (const auto &pair : valueMap) {
        for (const auto &pair2 : pair.second) {
          const auto itr = valueMapA.find(pair2.first);
          if (itr != valueMapA.end()) {
            for (const auto &pair3 : itr->second) {
              result.add_at(pair.first, pair3.first,
                            pair2.second * pair3.second);
            }
          }
        }
      }
      return result;
    }
  }

  std::vector<T> multiplyVecRight(const std::vector<T> &A) const {  // self * A
    if (colNum != A.size()) {
      std::cerr << "Not matching matrix size." << std::endl;
      return std::vector<T>();
    } else {
      T v;
      std::vector<T> result(rowNum, 0.0);
      for (const auto &pair : valueMap) {
        v = 0.0;
        for (const auto &pair2 : pair.second) {
          v += pair2.second * A.at(pair2.first);
        }
        result.at(pair.first) = v;
      }
      return result;
    }
  }

  SparseMat<T> multiplySchalar(const T alpha) const {
    SparseMat<T> result(rowNum, colNum);
    for (const auto &pair : valueMap) {
      for (const auto &pair2 : pair.second) {
        result.insert_at(pair.first, pair2.first, pair2.second * alpha);
      }
    }
    return result;
  }

  SparseMat<T> power(const T alpha) const {
    SparseMat<T> result(rowNum, colNum);
    for (const auto &pair : valueMap) {
      for (const auto &pair2 : pair.second) {
        result.insert_at(pair.first, pair2.first,
                         std::pow(pair2.second, alpha));
      }
    }
    return result;
  }

  SparseMat<T> transpose() const {
    SparseMat<T> result(colNum, rowNum);
    for (const auto &pair : valueMap) {
      for (const auto &pair2 : pair.second) {
        result.insert_at(pair2.first, pair.first, pair2.second);
      }
    }
    return result;
  }

  static std::vector<T> linsolve(const SparseMat<T> &Amat,
                                 const std::vector<T> &b) {
    SparseMat<T> A(Amat);
    std::vector<T> bcopy(b);

    if (A.getCol() != A.getRow()) {
      std::cerr << "Matrix A is not square." << std::endl;
      return std::vector<T>(b.size(), std::numeric_limits<T>::infinity());
    }
    if (A.getCol() != b.size()) {
      std::cerr << "Sizes of A and b are not matching." << std::endl;
      return std::vector<T>(A.getCol(), std::numeric_limits<T>::infinity());
    }
    if (A.getValueMap().size() < A.getRow()) {
      std::cerr << " Matrix A is singular." << std::endl;
      return std::vector<T>(b.size(), std::numeric_limits<T>::infinity());
    }

    std::size_t maxRow;
    T tmpMax, m, btmp;
    for (std::size_t i = 0; i < A.getCol(); i++) {
      maxRow = i;
      tmpMax = std::abs(A.at(i, i));
      for (std::size_t j = i + 1; j < A.getRow(); j++) {
        if (std::abs(A.at(j, i)) > tmpMax) {
          maxRow = j;
          tmpMax = std::abs(A.at(j, i));
        }
      }
      if (tmpMax < EPSILON) {
        std::cerr << " Matrix A is singular." << std::endl;
        return std::vector<T>(b.size(), std::numeric_limits<T>::infinity());
      }
      A.swap_row(i, maxRow);
      btmp = bcopy.at(i);
      bcopy.at(i) = bcopy.at(maxRow);
      bcopy.at(maxRow) = btmp;

      tmpMax = A.at(i, i);
      for (std::size_t j = 0; j < A.getRow(); j++) {
        if (i == j) {
          continue;
        }
        m = -A.at(j, i) / tmpMax;
        A.rowAddition(j, i, m);
        bcopy.at(j) += bcopy.at(i) * m;
      }
    }
    std::vector<T> result(A.getCol());
    for (std::size_t i = 0; i < A.getCol(); i++) {
      result.at(i) = bcopy.at(i) / A.at(i, i);
    }
    return result;
  }

  SparseMat<T> &operator=(const SparseMat<T> &A) {
    rowNum = A.getRow();
    colNum = A.getCol();
    valueMap = A.getValueMap();
    return *this;
  }
};
template <typename T>
SparseMat<T> operator+(const SparseMat<T> &A, const SparseMat<T> &B) {
  return A.add(B);
}
template <typename T>
SparseMat<T> operator-(const SparseMat<T> &A, const SparseMat<T> &B) {
  return A.minus(B);
}
template <typename T>
SparseMat<T> operator-(const SparseMat<T> &A) {
  return A.minus();
}
template <typename T>
SparseMat<T> operator*(const SparseMat<T> &A, const SparseMat<T> &B) {
  return A.multiplyRight(B);
}
template <typename T>
SparseMat<T> operator*(const SparseMat<T> &A, const std::vector<T> &B) {
  return A.multiplyVecRight(B);
}
template <typename T>
SparseMat<T> operator*(const std::vector<T> &B, const SparseMat<T> &A) {
  return A.multiplyVecLeft(B);
}
template <typename T>
SparseMat<T> operator*(const SparseMat<T> &A, const T alpha) {
  return A.multiplySchalar(alpha);
}
template <typename T>
SparseMat<T> operator*(const T alpha, const SparseMat<T> &A) {
  return A.multiplySchalar(alpha);
}
}  // namespace Hongo

#endif /* SparseMat_hpp */
