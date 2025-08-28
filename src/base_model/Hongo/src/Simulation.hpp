//
//  Simulation.hpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/06.
//

#ifndef Simulation_hpp
#define Simulation_hpp

#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Table.hpp"

namespace Hongo {

class Link;
class Lane;
class Node;
class Agent;
class Station;
class Signal;
class MigrationAttribute;
class Simulation {
  // mode  0:car, 2: bicycle, 3: walk, 5: train, 6: bus
  const static int maxAgent;  // エージェント発生人数の最大値

  double simTime;
  double startTime;
  double endTime;
  double timestep;
  bool finish = false;

  bool noOutput;

  std::filesystem::path outputDir;

  // データ読み込み用
  Table linkTable, nodeTable, connectivityTable, linkCrossTable, signalTable,
      facilityTable, agentTable, busTable, busRouteTable, stationTable,
      busPathTable, busLaneTable, linkPropTable, linkCoeffTable;

  std::unordered_map<int, std::shared_ptr<Link>> linkMap;
  std::unordered_map<int, std::shared_ptr<Node>> nodeMap;
  std::unordered_map<int, std::shared_ptr<Signal>> signalMap;
  std::unordered_map<int, std::shared_ptr<Station>> stationMap;
  std::unordered_map<int, std::shared_ptr<Link>>
      busLinkMap;  // 駅と駅との接続関係を表す
  std::unordered_map<int, std::vector<std::shared_ptr<Link>>>
      busPathMap;  // バスの経路 key: busRouteID, value: vector(linkID)
  std::unordered_map<int, std::vector<std::shared_ptr<Agent>>>
      agentTimeMap;  // key: time

  std::unordered_map<int, std::map<double, std::shared_ptr<Station>>>
      busRouteMap;  // key: routeID, value: (key: 出発時間からの差, value:
                    // station)

  std::unordered_map<int, std::shared_ptr<MigrationAttribute>>
      facilityMap;  // key: nodeID, value: ノードごとの属性

  std::vector<std::shared_ptr<Agent>>
      tmpAgents;  // 現在アクティブなエージェント

  std::ofstream foutAgentLoc;     // 毎秒のエージェントの状態
  std::ofstream foutDestination;  // エージェントの目的地
  std::ofstream foutLink;         // リンクの集計料
  std::ofstream foutAgent;        // エージェントごとの集計料
  std::string agentLocationPath;
  std::string destinationPath;
  std::string linkOutputPath;
  std::string agentOutputPath;

  bool foutAgentLocUsed = false;
  bool foutDestinationUsed = false;
  bool foutLinkUsed = false;
  bool foutAgentUsed = false;

 public:
  Simulation(std::string _outputDir, double _startTime, double _endTime,
             double _timestep, bool _noOutput, int _rectPlaneNum);

  void readData(std::string _inputDir);

  void initialize();
  void setNetwork();
  void setPublicTrans();
  void setFacilities();
  void setAgent();
  void setTransportationAgent();

  void addAgent(
      std::string agentPath);  // シミュレーション途中でエージェントを追加する
  void addBus(std::string busPath, std::string busRoutePath,
              std::string stationPath,
              std::string busPathPath);  // 途中でバスを追加する．
  std::vector<
      std::pair<std::pair<int, std::pair<int, int>>, std::pair<int, int>>>
  getTmpAgents();  // トリップ終了前のエージェントのID,ONodeID,DNodeID,coordを取得

  std::vector<std::pair<int, std::pair<int, int>>> calculation();
  void createAgent();

  void writeAgentOutput(bool destination, bool location);
  void writeAgentDestination();
  void writeAgentLocation();
  void writeAggregateAgentOutput(
      const std::vector<std::shared_ptr<Agent>> &finishAgents);

  void writeLinkOutput();

  void updateNetwork();
  void close();
  bool readCSV(std::string _file, std::unordered_map<std::string, int> &header,
               std::vector<std::vector<std::string>> &vec);

  void reset();

  bool isFinish() { return finish; }
};
}  // namespace Hongo

#endif /* Simulation_hpp */
