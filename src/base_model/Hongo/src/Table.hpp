//
//  Table.hpp
//  Hongo
//
//  Created by 小川大智 on 2022/05/16.
//

#ifndef Table_hpp
#define Table_hpp

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Hongo {

class Table {
  std::unordered_map<std::string, size_t>
      header;  // key: columnName, value: columnNumber
  std::vector<std::vector<std::string>> contentVec;

 public:
  Table() {};
  Table(const std::string &filePath) {
    std::vector<std::string> filePathVec = split(filePath, '.');
    std::string suffix = filePathVec.at(filePathVec.size() - 1);
    if (suffix == "csv" || suffix == "CSV") {
      readCSV(filePath);
    }
  }
  Table(const std::unordered_map<std::string, size_t> &_header,
        const std::vector<std::vector<std::string>> &_contentVec)
      : header(_header), contentVec(_contentVec) {}
  Table(const Table &table) {
    for (const auto &pair : table.getHeader()) {
      header.emplace(pair.first, pair.second);
    }
    for (size_t i = 0; i < table.getContentVec().size(); i++) {
      std::vector<std::string> vec;
      for (const auto &itr : table.getContentVec()[i]) {
        vec.emplace_back(itr);
      }
      contentVec.emplace_back(vec);
    }
    header = table.getHeader();
    contentVec = table.getContentVec();
  }

  std::string at(size_t row, size_t col) const {
    if (row < contentVec.size()) {
      if (col < contentVec.at(row).size()) {
        return contentVec.at(row).at(col);
      } else {
        return "0";
      }
    } else {
      return "0";
    }
  }

  std::string at(size_t row, std::string col) const {
    if (header.find(col) != header.end()) {
      if (row < contentVec.size()) {
        return contentVec.at(row).at(header.at(col));
      } else {
        return "0";
      }
    } else {
      return "0";
    }
  }

  std::vector<std::string> at(const std::string &col) const {
    std::vector<std::string> result(contentVec.size(), "0");
    if (header.find(col) != header.end()) {
      size_t row = 0;
      for (const auto &vec : contentVec) {
        result.at(row++) = vec.at(header.at(col));
      }
    }
    return result;
  }

  void setVal(size_t row, const std::string &col, const std::string &val) {
    if (row < contentVec.size() && header.find(col) != header.end()) {
      contentVec.at(row).at(header.at(col)) = val;
    }
  }

  void concate(const Table &newCols) {
    if (contentVec.size() != newCols.length()) {
      std::cout << "Table length is not the same between original table and "
                   "new columns."
                << std::endl;
      return;
    }
    size_t prev_ncol = header.size();
    size_t ncol = header.size() + newCols.size().second;
    for (const auto &pair : newCols.getHeader()) {
      header.emplace(pair.first, prev_ncol + pair.second);
    }
    const std::vector<std::vector<std::string>> &newContentVec =
        newCols.getContentVec();
    for (size_t row = 0; row < contentVec.size(); row++) {
      contentVec.at(row).resize(ncol);
      contentVec.at(row).insert(contentVec.at(row).begin() + prev_ncol,
                                newContentVec.at(row).begin(),
                                newContentVec.at(row).end());
    }
  }

  void append(const Table &newRows) {
    size_t prev_length = contentVec.size();
    contentVec.resize(prev_length + newRows.length());
    for (size_t row = prev_length; row < prev_length + newRows.length(); row++)
      contentVec.at(row) = std::vector<std::string>(header.size(), "0");
    for (const auto &pair : newRows.getHeader()) {
      if (contain(pair.first)) {
        for (size_t row = 0; row < newRows.length(); row++) {
          contentVec.at(row + prev_length).at(header.at(pair.first)) =
              newRows.at(row, pair.first);
        }
      }
    }
  }

  Table multiply(size_t t) const {
    // 1行をt行に拡大する．行の順番は変化しない．
    std::vector<std::vector<std::string>> newContentVec(
        length() * t, std::vector<std::string>(width()));
    std::vector<std::string> tmpRow(width());
    for (size_t row = 0; row < length(); row++) {
      tmpRow = contentVec.at(row);
      for (size_t sampleNum = 0; sampleNum < t; sampleNum++) {
        newContentVec.at(row * t + sampleNum) = tmpRow;
      }
    }
    Table newTable = Table(header, newContentVec);
    return newTable;
  }

  bool contain(const std::string &col) const {
    // colがheaderに含まれているか
    return header.find(col) != header.end();
  }

  size_t getIndex(const std::string &col, const std::string &val) const {
    if (contain(col)) {
      size_t colIndex = header.at(col);
      for (size_t index = 0; index < length(); index++) {
        if (contentVec.at(index).at(colIndex) == val) return index;
      }
    }
    return length();
  }

  size_t length() const { return contentVec.size(); }

  size_t width() const { return header.size(); }

  std::pair<size_t, size_t> size() const {
    return std::make_pair(contentVec.size(), header.size());
  }

  void clear() {
    header.clear();
    contentVec.clear();
  }

  bool empty() { return header.empty(); }

  bool readCSV(const std::string &filePath) {
    clear();
    char delimiter = ',';
    std::ifstream ifs(filePath);
    std::string line;
    if (ifs.fail()) {
      std::cerr << "Failed to open " << filePath << "." << std::endl;
      return false;
    }
    if (getline(ifs, line)) {
      std::vector<std::string> values = split(line, delimiter);
      for (size_t i = 0; i < values.size(); i++) {
        header[values.at(i)] = i;
      }
    }
    while (getline(ifs, line)) {
      std::vector<std::string> values = split(line, delimiter);
      contentVec.push_back(values);
    }
    return true;
  }

  const std::unordered_map<std::string, size_t> &getHeader() const {
    return header;
  }

  const std::vector<std::vector<std::string>> &getContentVec() const {
    return contentVec;
  }

  std::vector<std::string> split(const std::string &input, char delimiter) {
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter)) {
      result.push_back(field);
    }
    field = stream.str();
    if (field[field.size() - 1] == delimiter) {  // 最後が空文字だった場合
      result.push_back("");
    }
    return result;
  }
};
}  // namespace Hongo

#endif /* Table_hpp */
