//
//  Kepler.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/11/09.
//

#ifndef Kepler_hpp
#define Kepler_hpp

#include <stdio.h>
#include <time.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace Hongo {

struct Location {
  double lon, lat, velocity;
};
struct Person {
  int id;
  long long startTime = std::numeric_limits<long long>::infinity();
  long long endTime = -1;
  std::map<long long, Location> data;

  void addData(long long time, double lat, double lon, double velocity) {
    if (time > 0 && lat > 0 && lon > 0) {
      Location location;
      location.lat = lat;
      location.lon = lon;
      location.velocity = velocity;
      data[time] = location;
      startTime = std::min(startTime, time);
      endTime = std::max(endTime, time);
    }
  }
};
struct Persons {
  std::map<int, Person> personMap;

  void addPerson(int id, long long time, double lat, double lon,
                 double velocity) {
    auto itr = personMap.find(id);
    if (itr != personMap.end()) {
      itr->second.addData(time, lat, lon, velocity);
    } else {
      personMap[id] = {id};
      personMap.at(id).addData(time, lat, lon, velocity);
    }
  }
  bool toKeplerOutput(std::string path) {
    std::ofstream fout;
    fout.open(path);
    if (!fout.is_open()) {
      return false;
    }
    fout << "{\"type\": \"FeatureCollection\", \"features\": [" << std::endl;
    for (auto person = personMap.begin(); person != personMap.end(); person++) {
      fout << "{" << std::endl;
      fout << " \"type\": \"Feature\",\"geometry\": { \"type\": \"LineString\","
           << std::endl;
      fout << "  \"coordinates\": [" << std::endl;
      const std::map<long long, Location> &data = person->second.data;
      long long prevTime = -1;
      double prevLon = 0;
      double prevLat = 0;
      int i = 0;
      for (auto loc = data.begin(); loc != data.end(); loc++) {
        if (loc == data.begin()) {
          prevTime = loc->first;
          prevLon = loc->second.lon;
          prevLat = loc->second.lat;
          continue;
        }
        if (i % 1 == 0) {
          if (prevLon == loc->second.lon && prevLat == loc->second.lat) {
            continue;
          } else {
            fout << "    [";
            fout << std::to_string(prevLon) + "," + std::to_string(prevLat) +
                        ",0," + std::to_string(prevTime);
            if (loc->first - prevTime >
                5000) {  // 次のデータまで時間が空いている場合、別のfeatureとして扱う。
              fout << "]" << std::endl;
              fout << "  ]," << std::endl;
              fout << "  \"properties\": {" << std::endl;
              fout << "    \"AgentID\": \"" << std::to_string(person->first)
                   << "\"" << std::endl;
              fout << "  }" << std::endl;
              fout << "  }" << std::endl;
              fout << "}," << std::endl;
              fout << "{" << std::endl;
              fout << " \"type\": \"Feature\",\"geometry\": { \"type\": "
                      "\"LineString\","
                   << std::endl;
              fout << "  \"coordinates\": [" << std::endl;
            } else {  // 次のデータが連続してある場合
              fout << "]," << std::endl;
            }
          }
          prevTime = loc->first;
          prevLon = loc->second.lon;
          prevLat = loc->second.lat;
        }
        i++;
      }
      fout << "    [";
      fout << std::to_string(prevLon) + "," + std::to_string(prevLat) + ",0," +
                  std::to_string(prevTime);
      fout << "]" << std::endl;
      fout << "  ]," << std::endl;
      fout << "  \"properties\": {" << std::endl;
      fout << "    \"AgentID\": \"" << std::to_string(person->first) << "\""
           << std::endl;
      fout << "  }" << std::endl;
      fout << "  }" << std::endl;
      if (person != --personMap.end()) {
        fout << "}," << std::endl;
      } else {
        fout << "}" << std::endl;
      }
    }
    fout << "]}" << std::endl;
    fout.close();
    return true;
  }
};

class Kepler {
  std::unordered_map<int, Persons> modes;

 public:
  Kepler(std::string inputPath, std::string outputPath) {
    if (readInput(inputPath)) {
      time_t t = time(nullptr);
      const tm *lt = localtime(&t);
      /*
      std::string path=outputPath+"KeplerOutput_"+std::to_string(lt->tm_year -
      100 )+std::to_string(lt->tm_mon + 1
      )+std::to_string(lt->tm_mday)+std::to_string(lt->tm_hour)+std::to_string(lt->tm_min);
       */
      std::string path = (outputPath / fs::path("KeplerOutput")).string();
      writeTripOutput(path);
    }
  }
  bool readInput(const std::string inputPath) {
    char delimiter = ',';
    /*struct tm tm;
    std::string strDate="2020-10-08 00:00:00.0";
    const char *datetime=strDate.c_str();
    strptime(datetime,"%Y-%m-%d %H:%M:%S",&tm);
    time_t t=mktime(&tm);*/
    long long t = (long long)1602082800000;
    std::ifstream ifs(inputPath);
    std::string line;
    if (ifs.fail()) {
      std::cerr << "Failed to open input file." << std::endl;
      return false;
    }
    std::unordered_map<std::string, std::size_t>
        header;  // key: column名, value: index
    std::size_t timeIndex, idIndex, latIndex, lonIndex, veloIndex, modeIndex;
    if (getline(ifs, line)) {
      std::vector<std::string> values = split(line, delimiter);
      for (std::size_t i = 0; i < values.size(); i++) {
        header[values.at(i)] = i;
      }
      timeIndex = header.at("SimulationTime");
      idIndex = header.at("AgentID");
      latIndex = header.at("Lat");
      lonIndex = header.at("Lon");
      veloIndex = header.at("Velocity(km/h)");
      modeIndex = header.at("Mode");
    }

    while (getline(ifs, line)) {
      // time,agentID,linkID,laneID,lat,lon,velocity,linkLength,positionRate,mode,
      std::vector<std::string> values = split(line, delimiter);
      long long time = std::stol(values[timeIndex]);
      time = (long long)(time * 1000) + t + (long long)(9 * 3600 * 1000);
      int id = std::stoi(values[idIndex]);
      double lat = std::stod(values[latIndex]);
      double lon = std::stod(values[lonIndex]);
      double velocity = std::stod(values[veloIndex]);
      int mode = std::stoi(values[modeIndex]);
      if (modes.find(mode) != modes.end()) {
        modes.at(mode).addPerson(id, time, lat, lon, velocity);
      } else {
        modes[mode] = Persons();
        modes.at(mode).addPerson(id, time, lat, lon, velocity);
      }
    }
    return true;
  }

  void writeTripOutput(std::string outputPath) {
    for (auto mode = modes.begin(); mode != modes.end(); mode++) {
      if (mode->first >= 100) {
        continue;
      }
      std::cout << "Starting mode" << mode->first << std::endl;
      std::string path =
          outputPath + "_" + std::to_string(mode->first) + ".json";
      if (!mode->second.toKeplerOutput(path)) {
        std::cerr << "Failed to write output file of mode " << mode->first
                  << std::endl;
      } else {
        std::cout << "Finish mode" << mode->first << std::endl;
      }
    }
  }

  std::vector<std::string> split(std::string &input, char delimiter) {
    std::stringstream stream(input);
    std::string value;
    std::vector<std::string> result;
    while (getline(stream, value, delimiter)) {
      result.push_back(value);
    }
    return result;
  }
};
}  // namespace Hongo

#endif /* Kepler_hpp */
