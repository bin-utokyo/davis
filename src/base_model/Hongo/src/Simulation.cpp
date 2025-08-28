//
//  Simulation.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/06.
//

#include "Simulation.hpp"

#include <filesystem>

#include "Agent.hpp"
#include "Bus.hpp"
#include "Kepler.hpp"
#include "Lane.hpp"
#include "Link.hpp"
#include "Node.hpp"
#include "Signal.hpp"
#include "Station.hpp"
#include "Utility.hpp"

namespace Hongo {

const bool Agent::ecursion = false;
const int Simulation::maxAgent = 50000;
int rectPlaneNum;
namespace fs = std::filesystem;

Simulation::Simulation(std::string _outputDir, double _startTime,
                       double _endTime, double _timestep, bool _noOutput,
                       int _rectPlaneNum) {
  outputDir = _outputDir;
  simTime = _startTime;
  startTime = _startTime;
  endTime = _endTime;
  timestep = _timestep;
  noOutput = _noOutput;
  rectPlaneNum = _rectPlaneNum;  // defined in Utility.hpp

  time_t t = time(nullptr);
  const tm *lt = localtime(&t);

  if (noOutput) return;

  /*
  agentLocationPath = _outputDir+"AgentLocation_"+std::to_string(lt->tm_year -
  100 )+std::to_string(lt->tm_mon + 1
  )+std::to_string(lt->tm_mday)+std::to_string(lt->tm_hour)+std::to_string(lt->tm_min)+".csv";
  destinationPath = _outputDir+"AgentDestination_"+std::to_string(lt->tm_year -
  100 )+std::to_string(lt->tm_mon + 1
  )+std::to_string(lt->tm_mday)+std::to_string(lt->tm_hour)+std::to_string(lt->tm_min)+".csv";
  linkOutputPath = _outputDir+"LinkOutput_"+std::to_string(lt->tm_year - 100
  )+std::to_string(lt->tm_mon + 1
  )+std::to_string(lt->tm_mday)+std::to_string(lt->tm_hour)+std::to_string(lt->tm_min)+".csv";
  */
  fs::create_directories(outputDir);
  agentLocationPath = (outputDir / fs::path("AgentLocation.csv")).string();
  destinationPath = (outputDir / fs::path("AgentDestination.csv")).string();
  linkOutputPath = (outputDir / fs::path("LinkOutput.csv")).string();
  agentOutputPath = (outputDir / fs::path("AgentOutput.csv")).string();

  foutAgentLoc.open(agentLocationPath);
  foutDestination.open(destinationPath);
  foutLink.open(linkOutputPath);
  foutAgent.open(agentOutputPath);
  if (!foutAgentLoc.is_open()) {
    std::cerr << "Invalid path : " << agentLocationPath << std::endl;
    finish = true;
  } else {
    foutAgentLoc << "Time,SimulationTime,AgentID,Mode,Lon,Lat,Velocity(km/"
                    "h),Acceleration(m/"
                    "s),LinkID,LaneID,X,InNodeLane,CarFollowing,SignalStop,"
                    "Transportation"
                 << std::endl;
  }
  if (!foutDestination.is_open()) {
    std::cerr << "Invalid path : " << destinationPath << std::endl;
    finish = true;
  } else {
    foutDestination
        << "Time,SimulationTime,AgentID,Mode,StayTime(min),DUtility,ONodeID,"
           "DNodeID,O_Lon,O_Lat,D_Lon,D_Lat,Purpose,LastTrip"
        << std::endl;
  }
  if (!foutLink.is_open()) {
    std::cerr << "Invalid path : " << linkOutputPath << std::endl;
    finish = true;
  } else {
    foutLink
        << "Time,SimulationTime,LinkID,CarTrafficVolume(OD),CarTrafficVolume("
           "DO),CarVelocity(km/h)(OD),CarVelocity(km/h)(DO),CarVelocity(km/"
           "h),TrafficVolume,Velocity(km/"
           "h),ONodeID,DNodeID,O_Lon,O_Lat,D_Lon,D_Lat"
        << std::endl;
  }
  if (!foutAgent.is_open()) {
    std::cerr << "Invalid path : " << agentOutputPath << std::endl;
    finish = true;
  } else {
    foutAgent
        << "AgentID,Mode,ONodeID,DNodeID,O_Lon,O_Lat,D_Lon,D_Lat,StartTime,"
           "EndTime,TotalUtility,TotalDistance(m)"
        << std::endl;
  }
}

void Simulation::readData(std::string _inputDir) {
  fs::path inputDir = _inputDir;
  if (linkTable.readCSV((inputDir / fs::path("link.csv")).string())) {
    // ID,ONodeID,DNodeID,Velocity,LeftCarNum,LeftPedNum,LeftBeltNum,RightCarNum,RightPedNum,RightBeltNum
    std::cout << "Link file is read." << std::endl;
  }
  if (nodeTable.readCSV((inputDir / fs::path("node.csv")).string())) {
    // ID,Lon,Lat
    std::cout << "Node file is read." << std::endl;
  }
  if (connectivityTable.readCSV(
          (inputDir / fs::path("connectivity.csv")).string())) {
    // NodeID,UpLinkID,UpLaneID,DnLinkID,DnLaneID
    std::cout << "connectivity file is read." << std::endl;
  }
  if (linkCrossTable.readCSV((inputDir / fs::path("linkCross.csv")).string())) {
    // LinkID,PriorLinkID
    std::cout << "link cross file is read." << std::endl;
  }
  if (signalTable.readCSV((inputDir / fs::path("signal.csv")).string())) {
    // ID,NodeID,RightTurn
    std::cout << "signal file is read." << std::endl;
  }
  if (facilityTable.readCSV((inputDir / fs::path("facility.csv")).string())) {
    std::cout << "facility file is read." << std::endl;
  }
  if (agentTable.readCSV((inputDir / fs::path("agent.csv")).string())) {
    // ID,StartTime,Mode,ONodeID,DNodeID
    std::cout << "agent file is read." << std::endl;
  }
  if (busTable.readCSV((inputDir / fs::path("bus.csv")).string())) {
    // RouteID,StartTime,ONodeID,DNodeID
    std::cout << "bus file is read." << std::endl;
  }
  if (busRouteTable.readCSV((inputDir / fs::path("busRoute.csv")).string())) {
    // ID,StationID,TimeOffset
    std::cout << "busRoute file is read." << std::endl;
  }
  if (stationTable.readCSV((inputDir / fs::path("station.csv")).string())) {
    // ID,LinkID,LaneID,Train,Bus
    std::cout << "station file is read." << std::endl;
  }
  if (busPathTable.readCSV((inputDir / fs::path("busPath.csv")).string())) {
    // RouteID,LinkID
    std::cout << "busPath file is read." << std::endl;
  }
  if (busLaneTable.readCSV((inputDir / fs::path("busLane.csv")).string())) {
    // LinkID,LaneID
    std::cout << "busLane file is read." << std::endl;
  }
  if (linkPropTable.readCSV((inputDir / fs::path("linkProp.csv")).string())) {
    // LinkID,prop1,porp2,...
    std::cout << "LinkProp file is read." << std::endl;
  }
  if (linkCoeffTable.readCSV((inputDir / fs::path("linkCoeff.csv")).string())) {
    // PropName,Mode,Coeff
    std::cout << "LinkCoeff file is read." << std::endl;
  }
}

void Simulation::initialize() {
  std::cout << "Setting network start." << std::endl;
  setNetwork();
  std::cout << "Setting network finish." << std::endl;
  std::cout << "Setting public transport start." << std::endl;
  setPublicTrans();
  std::cout << "Setting public transport finish." << std::endl;
  std::cout << "Setting facilities start." << std::endl;
  setFacilities();
  std::cout << "Setting facilities finish." << std::endl;
  std::cout << "Setting agents start." << std::endl;
  setAgent();
  std::cout << "Setting agents finish." << std::endl;
}

void Simulation::setNetwork() {
  using std::stod;
  using std::stoi;
  // nodeMap
  nodeMap.clear();
  for (size_t row = 0; row < nodeTable.length(); row++) {
    nodeMap.emplace(stoi(nodeTable.at(row, "ID")),
                    std::make_shared<Node>(stoi(nodeTable.at(row, "ID")),
                                           stod(nodeTable.at(row, "Lon")),
                                           stod(nodeTable.at(row, "Lat"))));
  }
  nodeTable.clear();
  // linkMap
  linkMap.clear();
  int linkId, oNodeId, dNodeId, lpedNum, lbeltNum, rpedNum, rbeltNum;
  for (size_t row = 0; row < linkTable.length(); row++) {
    linkId = stoi(linkTable.at(row, "ID"));
    oNodeId = stoi(linkTable.at(row, "ONodeID"));
    dNodeId = stoi(linkTable.at(row, "DNodeID"));
    lpedNum = linkTable.contain("LeftPedNum")
                  ? stoi(linkTable.at(row, "LeftPedNum"))
                  : 0;
    lbeltNum = linkTable.contain("LeftBeltNum")
                   ? stoi(linkTable.at(row, "LeftBeltNum"))
                   : 0;
    rpedNum = linkTable.contain("RightPedNum")
                  ? stoi(linkTable.at(row, "RightPedNum"))
                  : 0;
    rbeltNum = linkTable.contain("RightBeltNum")
                   ? stoi(linkTable.at(row, "RightBeltNum"))
                   : 0;
    if (nodeMap.find(oNodeId) != nodeMap.end() &&
        nodeMap.find(dNodeId) != nodeMap.end()) {
      linkMap.emplace(
          linkId, std::make_shared<Link>(linkId, nodeMap.at(oNodeId),
                                         nodeMap.at(dNodeId),
                                         stod(linkTable.at(row, "Velocity"))));
      linkMap.at(linkId)->setLanes(
          stoi(linkTable.at(row, "LeftCarNum")), lpedNum, lbeltNum,
          stoi(linkTable.at(row, "RightCarNum")), rpedNum, rbeltNum);

      nodeMap.at(oNodeId)->addLink(linkMap.at(linkId));
      nodeMap.at(dNodeId)->addLink(linkMap.at(linkId));
    }
  }
  for (const auto &pair : nodeMap) {
    pair.second->setOffset();
  }
  linkTable.clear();
  // linkProp
  std::vector<std::string> props;  // 経路選択用のlink属性
  for (const auto &pair : linkPropTable.getHeader()) {
    if (pair.first != "LinkID") {
      props.emplace_back(pair.first);
    }
  }
  for (size_t row = 0; row < linkPropTable.length(); row++) {
    linkId = stoi(linkPropTable.at(row, "LinkID"));
    if (linkMap.find(linkId) != linkMap.end()) {
      int propVal;
      std::unordered_map<std::string, double> properties;
      for (const auto &prop : props) {
        const auto val = linkPropTable.at(row, prop);
        if (val.size() > 0) {
          properties[prop] = stod(val);
        }
      }
      auto link = linkMap.at(linkId);
      link->setProperties(properties);
    }
  }
  linkPropTable.clear();
  // linkCoeff
  for (size_t row = 0; row < linkCoeffTable.length(); row++) {
    int mode = stoi(linkCoeffTable.at(row, "Mode"));
    std::string propName = linkCoeffTable.at(row, "PropName");
    double coeff = stod(linkCoeffTable.at(row, "Coeff"));
    linkMap.begin()->second->setUtilityCoefficient(mode, propName, coeff);
  }

  // busLane
  int laneId;
  for (size_t row = 0; row < busLaneTable.length(); row++) {
    linkId = stoi(busLaneTable.at(row, "LinkID"));
    laneId = stoi(busLaneTable.at(row, "LaneID"));
    if (linkMap.find(linkId) != linkMap.end()) {
      auto lane = linkMap.at(linkId)->getLane(laneId);
      if (lane != nullptr) {
        lane->setType("bus");
      }
    }
  }
  busLaneTable.clear();
  // connectivity

  int upLinkId, dnLinkId, nodeId;
  for (size_t row = 0; row < connectivityTable.length(); row++) {
    nodeId = stoi(connectivityTable.at(row, "NodeID"));
    upLinkId = stoi(connectivityTable.at(row, "UpLinkID"));
    dnLinkId = stoi(connectivityTable.at(row, "DnLinkID"));
    if (linkMap.find(upLinkId) != linkMap.end() &&
        linkMap.find(dnLinkId) != linkMap.end()) {
      auto upLane = linkMap.at(upLinkId)->getLane(
          stoi(connectivityTable.at(row, "UpLaneID")));
      auto dnLane = linkMap.at(dnLinkId)->getLane(
          stoi(connectivityTable.at(row, "DnLaneID")));
      if (upLane != nullptr && dnLane != nullptr) {
        upLane->addDnLane(nodeId, dnLane);
      }
    }
  }
  connectivityTable.clear();
  // link cross
  int plinkId;
  for (size_t row = 0; row < linkCrossTable.length(); row++) {
    linkId = stoi(linkCrossTable.at(row, "LinkID"));
    plinkId = stoi(linkCrossTable.at(row, "PriorLinkID"));
    if (linkMap.find(linkId) != linkMap.end() &&
        linkMap.find(plinkId) != linkMap.end()) {
      linkMap.at(linkId)->addPriorityLink(linkMap.at(plinkId));
    }
  }
  linkCrossTable.clear();
  // signal
  signalMap.clear();

  std::unordered_map<int, std::vector<std::shared_ptr<Node>>>
      intersectionMap;  // key: signalID, value: vector(nodeID)
  bool setLaneProp =
      signalTable.contain("LinkID") && signalTable.contain("LaneID") &&
      signalTable.contain("Cycle") && signalTable.contain("Blue") &&
      signalTable.contain("Offset");

  if (setLaneProp) {  // レーンごとの信号現示を指定する場合
    int signalId, nodeId, linkId, laneId, dLinkId;
    double cycle, blue, offset;
    // int _id, std::vector<std::shared_ptr<Node>> _nodeVec,
    // std::vector<std::shared_ptr<Lane>> _laneVec, double cycle, double blue,
    // std::vector<double> _offsetVec
    std::unordered_map<int, std::vector<std::shared_ptr<Lane>>>
        laneVecMap;  // key: signalID, value: laneVec
    std::unordered_map<int, std::vector<int>>
        dLinkMap;  // key: signalID, key:dLinkVec
    std::unordered_map<int, std::vector<double>> offsetVecMap;  // key: signalID
    std::unordered_map<int, std::vector<double>> blueVecMap;    // key: signalID
    std::unordered_map<int, double> cycleMap, blueMap;          // key: signalID

    bool contain_dlink = signalTable.contain("DLinkID");
    for (size_t row = 0; row < signalTable.length(); row++) {
      signalId = stoi(signalTable.at(row, "ID"));
      nodeId = stoi(signalTable.at(row, "NodeID"));
      linkId = stoi(signalTable.at(row, "LinkID"));
      laneId = stoi(signalTable.at(row, "LaneID"));
      cycle = stod(signalTable.at(row, "Cycle"));
      blue = stod(signalTable.at(row, "Blue"));
      offset = stod(signalTable.at(row, "Offset"));
      if (nodeMap.find(nodeId) != nodeMap.end() &&
          linkMap.find(linkId) != linkMap.end()) {
        intersectionMap[signalId].emplace_back(nodeMap.at(nodeId));
        const auto link = linkMap.at(linkId);
        const auto lane = link->getLane(laneId);
        laneVecMap[signalId].emplace_back(lane);
        offsetVecMap[signalId].emplace_back(offset);
        blueVecMap[signalId].emplace_back(blue);
        cycleMap[signalId] = cycle;
        if (contain_dlink && signalTable.at(row, "DLinkID").size() >
                                 0) {  // 制御対象リンクdLinkの情報のみ追加
          dLinkId = stoi(signalTable.at(row, "DLinkID"));
          dLinkMap[signalId].emplace_back(dLinkId);
        } else {
          dLinkMap[signalId].emplace_back(-1);
        }
      }
    }
    for (const auto &pair : intersectionMap) {
      signalMap[pair.first] = std::make_shared<Signal>(
          pair.first, pair.second, laneVecMap.at(pair.first),
          dLinkMap.at(pair.first), cycleMap.at(pair.first),
          blueVecMap.at(pair.first), offsetVecMap.at(pair.first));
      for (const auto &node : pair.second) {
        node->setSignal(signalMap.at(pair.first));
      }
    }
  } else {  // デフォルト値を使って信号を設定する場合
    std::unordered_map<int, bool> rightTurnMap;
    std::unordered_map<int, std::tuple<double, double, double>>
        signalPropMap;  // key: signalID, value: tuple(blue, yellow, rightBlue)
    bool setSignalProp = signalTable.contain("Blue") &&
                         signalTable.contain("Yellow") &&
                         signalTable.contain("RightBlue");

    int signalId, rightTurnNum;
    for (size_t row = 0; row < signalTable.length(); row++) {
      signalId = stoi(signalTable.at(row, "ID"));
      nodeId = stoi(signalTable.at(row, "NodeID"));
      rightTurnNum = signalTable.contain("RightTurn")
                         ? stoi(signalTable.at(row, "RightTurn"))
                         : stoi(signalTable.at(row, "RightBlue"));
      if (setSignalProp)
        signalPropMap[signalId] =
            std::make_tuple(stod(signalTable.at(row, "Blue")),
                            stod(signalTable.at(row, "Yellow")),
                            stod(signalTable.at(row, "RightBlue")));
      if (nodeMap.find(nodeId) != nodeMap.end()) {
        auto node = nodeMap.at(nodeId);
        intersectionMap[signalId].emplace_back(node);
        rightTurnMap[signalId] = rightTurnNum > 0;
      }
    }
    double blue, yellow, rightBlue;
    for (const auto &pair : intersectionMap) {
      if (!setSignalProp) {
        signalMap[pair.first] = std::make_shared<Signal>(
            pair.first, pair.second, rightTurnMap.at(pair.first));
      } else {
        auto [blue, yellow, rightBlue] = signalPropMap.at(pair.first);
        signalMap[pair.first] = std::make_shared<Signal>(
            pair.first, pair.second, blue, yellow, rightBlue);
      }
      for (const auto &node : pair.second) {
        node->setSignal(signalMap.at(pair.first));
      }
    }
  }

  signalTable.clear();
}

void Simulation::setPublicTrans() {
  if (stationTable.empty()) {
    return;
  }

  busLinkMap.clear();
  busRouteMap.clear();
  busPathMap.clear();

  setTransportationAgent();
}

void Simulation::setTransportationAgent() {
  using std::stod;
  using std::stoi;

  int id, linkId, laneId;
  for (size_t row = 0; row < stationTable.length(); row++) {
    std::unordered_set<int> modeSet;
    id = stoi(stationTable.at(row, "ID"));
    linkId = stoi(stationTable.at(row, "LinkID"));
    laneId = stoi(stationTable.at(row, "LaneID"));
    if (stoi(stationTable.at(row, "Train")) == 1) {
      modeSet.emplace(5);
    }
    if (stoi(stationTable.at(row, "Bus")) == 1) {
      modeSet.emplace(6);
    }
    if (linkMap.find(linkId) != linkMap.end()) {
      std::shared_ptr<Link> link = linkMap.at(linkId);
      std::shared_ptr<Lane> lane = link->getLane(laneId);
      if (lane != nullptr) {
        stationMap.emplace(
            id, std::make_shared<Station>(
                    id, lane, stod(stationTable.at(row, "X")), modeSet));
        link->addStation(stationMap.at(id));
      }
    }
  }
  stationTable.clear();

  int stationId;
  for (size_t row = 0; row < busRouteTable.length(); row++) {
    id = stoi(busRouteTable.at(row, "ID"));
    stationId = stoi(busRouteTable.at(row, "StationID"));
    if (stationMap.find(stationId) != stationMap.end()) {
      std::shared_ptr<Station> station = stationMap.at(stationId);
      busRouteMap[id].emplace(stod(busRouteTable.at(row, "TimeOffset")),
                              station);
    }
  }
  std::unordered_map<int, std::shared_ptr<Node>> stationNode;
  double x, lon, lat;
  std::shared_ptr<Lane> lane;
  for (const auto &pair : stationMap) {
    x = pair.second->getX();
    lane = pair.second->getLane();
    std::tie(lon, lat) = lane->getLonLatinLane(x, 3);
    stationNode[pair.first] = std::make_shared<Node>(pair.first, lon, lat);
  }
  std::unordered_map<int, std::vector<std::shared_ptr<Link>>> stationDLinkMap;
  std::shared_ptr<Link> routeLink;
  int routeLinkId = 1;
  for (const auto &pair : busRouteMap) {
    int prevNodeId = -1;
    int tmpNodeId = -1;
    double prevTime;
    for (const auto &pair2 : pair.second) {
      if (prevNodeId < 0) {
        prevNodeId = pair2.second->getId();
        prevTime = pair2.first;
        continue;
      }
      tmpNodeId = pair2.second->getId();
      routeLink = std::make_shared<Link>(
          routeLinkId, stationNode.at(prevNodeId), stationNode.at(tmpNodeId));
      routeLink->setTravelTime(pair2.first - prevTime);
      busLinkMap[routeLinkId] = routeLink;
      stationDLinkMap[prevNodeId].emplace_back(routeLink);
      routeLinkId++;
      prevNodeId = tmpNodeId;
      prevTime = pair2.first;
    }
  }
  // dnLinkの設定
  for (const auto &pair : busLinkMap) {
    int dNodeId = pair.second->getDNodeId();
    if (stationDLinkMap.find(dNodeId) != stationDLinkMap.end()) {
      for (const auto &stLink : stationDLinkMap.at(dNodeId)) {
        pair.second->addDnLink(dNodeId, stLink);
      }
    }
  }
  busRouteTable.clear();

  int routeId, startTime, oNodeId, dNodeId;
  for (size_t row = 0; row < busTable.length(); row++) {
    routeId = stoi(busTable.at(row, "RouteID"));
    startTime = stoi(busTable.at(row, "StartTime"));
    oNodeId = stoi(busTable.at(row, "ONodeID"));
    dNodeId = stoi(busTable.at(row, "DNodeID"));
    startTime = ((startTime - simTime) / timestep) * timestep + simTime;
    if (nodeMap.find(oNodeId) != nodeMap.end() &&
        nodeMap.find(dNodeId) != nodeMap.end() &&
        busRouteMap.find(routeId) != busRouteMap.end()) {
      std::shared_ptr<Node> oNode = nodeMap.at(oNodeId);
      std::shared_ptr<Node> dNode = nodeMap.at(dNodeId);
      std::map<double, std::shared_ptr<Station>> table =
          busRouteMap.at(routeId);
      std::shared_ptr<Agent> agent = std::make_shared<Agent>(
          6000000 + stoi(busTable.at(row, "ID")), startTime, 6, oNode, dNode);
      std::shared_ptr<Bus> bus = std::make_shared<Bus>(agent, routeId);
      for (const auto &pair : table) {
        bus->insertTimetable(pair.first + startTime, pair.second);
      }
      agent->setBus(bus);
      agentTimeMap[startTime].emplace_back(agent);
    }
  }
  busTable.clear();

  if (!busPathTable.empty()) {  // busの経路が指定されている場合
    // 同一RouteIDのものについては，通る順番でLinkIDを指定
    for (size_t row = 0; row < busPathTable.length(); row++) {
      routeId = stoi(busPathTable.at(row, "RouteID"));
      linkId = stoi(busPathTable.at(row, "LinkID"));
      busPathMap[routeId].emplace_back(
          linkMap.at(linkId));  // key: busRouteID, value: std::shared_ptr(link)
    }
  }
  busPathTable.clear();
}

void Simulation::setFacilities() {
  using std::stod;
  using std::stoi;

  facilityMap.clear();
  if (facilityTable.empty()) return;  // 施設インプットがない場合

  int nodeId;
  for (size_t row = 0; row < facilityTable.length(); row++) {
    //(int _startTime, int _endTime, double _spa, double _shop, double _famous)
    nodeId = stoi(facilityTable.at(row, "NodeID"));
    if (nodeMap.find(nodeId) != nodeMap.end()) {
      facilityMap[nodeId] = std::make_shared<MigrationAttribute>(
          nodeMap.at(nodeId), stoi(facilityTable.at(row, "StartTime")),
          stoi(facilityTable.at(row, "EndTime")),
          stod(facilityTable.at(row, "Spa")),
          stod(facilityTable.at(row, "Shop")),
          stod(facilityTable.at(row, "Famous")),
          stod(facilityTable.at(row, "Art")));
    }
  }
  facilityTable.clear();
}

void Simulation::setAgent() {
  using std::stod;
  using std::stoi;

  // 起終点と移動開始時刻、移動終了時刻を指定
  // 移動開始、終了時間を変化させた場合など
  int tmpAgentNum = 0;

  int id, startTime, dTime, oNodeId, dNodeId, mode, expansion;
  std::string purpose;
  std::shared_ptr<Node> oNode, dNode;
  for (size_t row = 0; row < agentTable.length(); row++) {
    startTime = stoi(agentTable.at(row, "StartTime"));
    startTime = ((startTime - simTime) / timestep) * timestep + simTime;
    oNodeId = stoi(agentTable.at(row, "ONodeID"));
    dNodeId = stoi(agentTable.at(row, "DNodeID"));
    mode = stoi(agentTable.at(row, "Mode"));

    dTime = agentTable.contain("EndTime") ? stoi(agentTable.at(row, "EndTime"))
                                          : startTime;
    purpose =
        agentTable.contain("Purpose") ? agentTable.at(row, "Purpose") : "";
    expansion = agentTable.contain("Expansion")
                    ? stoi(agentTable.at(row, "Expansion"))
                    : 1;
    //(int _id, int _oNodeId, int _dNodeId, int _startTime, int _endTime,
    // std::string _purpose)
    if (startTime >= simTime && nodeMap.find(oNodeId) != nodeMap.end() &&
        nodeMap.find(dNodeId) != nodeMap.end()) {
      id = stoi(agentTable.at(row, "ID"));
      oNode = nodeMap.at(oNodeId);
      dNode = nodeMap.at(dNodeId);

      for (int i = 0; i < 1; i++) {
        // agentTimeMap[startTime].emplace_back(std::make_shared<Agent>(id *
        // 10000 + i, startTime, endTime, purpose, oNode, dNode, facilityMap));
        agentTimeMap[startTime].emplace_back(std::make_shared<Agent>(
            id, startTime, dTime, purpose, mode, oNode, dNode, facilityMap));
      }
    }
    if (++tmpAgentNum >= maxAgent) {
      break;
    }
  }
  std::cout << "maxAgent remains: " << maxAgent - tmpAgentNum << std::endl;
  agentTable.clear();
}

void Simulation::addAgent(std::string agentPath) {
  if (agentTable.readCSV(agentPath)) {
    // ID,StartTime,Mode,ONodeID,DNodeID
    std::cout << "agent file is read." << std::endl;

    int id, startTime, dTime, oNodeId, dNodeId, mode, expansion;
    std::string purpose;
    std::shared_ptr<Node> oNode, dNode;
    for (size_t row = 0; row < agentTable.length(); row++) {
      startTime = stoi(agentTable.at(row, "StartTime"));
      startTime = ((startTime - simTime) / timestep) * timestep + simTime;
      oNodeId = stoi(agentTable.at(row, "ONodeID"));
      dNodeId = stoi(agentTable.at(row, "DNodeID"));
      mode = stoi(agentTable.at(row, "Mode"));

      dTime = agentTable.contain("EndTime")
                  ? stoi(agentTable.at(row, "EndTime"))
                  : startTime;
      purpose =
          agentTable.contain("Purpose") ? agentTable.at(row, "Purpose") : "";
      expansion = agentTable.contain("Expansion")
                      ? stoi(agentTable.at(row, "Expansion"))
                      : 1;
      //(int _id, int _oNodeId, int _dNodeId, int _startTime, int _endTime,
      // std::string _purpose)
      if (startTime >= simTime && nodeMap.find(oNodeId) != nodeMap.end() &&
          nodeMap.find(dNodeId) != nodeMap.end()) {
        id = stoi(agentTable.at(row, "ID"));
        oNode = nodeMap.at(oNodeId);
        dNode = nodeMap.at(dNodeId);

        for (int i = 0; i < 1; i++) {
          // agentTimeMap[startTime].emplace_back(std::make_shared<Agent>(id *
          // 10000 + i, startTime, endTime, purpose, oNode, dNode,
          // facilityMap));
          agentTimeMap[startTime].emplace_back(std::make_shared<Agent>(
              id, startTime, dTime, purpose, mode, oNode, dNode, facilityMap));
        }
      }
    }
    agentTable.clear();
  }
}

void Simulation::addBus(std::string busPath, std::string busRoutePath,
                        std::string stationPath, std::string busPathPath) {
  if (busTable.readCSV(busPath)) {
    std::cout << "bus file is read." << std::endl;
  }
  if (busRouteTable.readCSV(busRoutePath)) {
    std::cout << "busRoute file is read." << std::endl;
  }
  if (stationTable.readCSV(stationPath)) {
    std::cout << "station file is read." << std::endl;
  }
  if (busPathTable.readCSV(busPathPath)) {
    std::cout << "bus link file is read." << std::endl;
  }

  setTransportationAgent();
}

std::vector<std::pair<std::pair<int, std::pair<int, int>>, std::pair<int, int>>>
Simulation::getTmpAgents() {
  std::vector<
      std::pair<std::pair<int, std::pair<int, int>>, std::pair<int, int>>>
      result;  // AgentID, ONodeID, DNodeID
  for (const auto &a : tmpAgents) {
    result.emplace_back(std::make_pair(std::make_pair(a->getId(), a->getOD()),
                                       a->getGlobalCoord()));
  }
  return result;
}

std::vector<std::pair<int, std::pair<int, int>>> Simulation::calculation() {
  createAgent();

  for (const auto &agent : tmpAgents) {
    agent->setPrevPosition();
  }

  for (const auto &agent : tmpAgents) {
    agent->updateAcceleration(timestep, simTime);
  }
  for (const auto &agent : tmpAgents) {
    agent->updateVelocity(timestep);
  }

  std::vector<std::shared_ptr<Agent>> finishAgents;
  std::unordered_set<int> finishAgentIds;
  for (const auto &agent : tmpAgents) {
    auto finishAgent = agent->moveAgent(timestep, simTime);
    if (finishAgent != nullptr) {
      finishAgentIds.insert(finishAgent->getId());
      finishAgents.emplace_back(finishAgent);
    }
  }
  writeAggregateAgentOutput(finishAgents);
  auto itr =
      std::remove_if(tmpAgents.begin(), tmpAgents.end(), [&](const auto &a) {
        return finishAgentIds.find(a->getId()) != finishAgentIds.end();
      });

  std::vector<std::pair<int, std::pair<int, int>>>
      result;  // AgentID, ONodeID, DNodeID
  for (const auto &a : finishAgents) {
    std::cout << a->getId() << std::endl;
    std::pair<int, int> od = a->getOD();
    std::pair<int, std::pair<int, int>> val = std::make_pair(a->getId(), od);
    result.emplace_back(val);
  }
  if (finishAgents.size() > 0) {
    std::cout << "Agents finish: t=" << simTime << "," << finishAgents.size()
              << std::endl;
  }

  tmpAgents.erase(itr, tmpAgents.end());

  simTime += timestep;
  if (simTime >= endTime) {
    finish = true;
  }
  return result;
}

void Simulation::createAgent() {
  if (agentTimeMap.find(simTime) != agentTimeMap.end()) {
    int routeId;
    for (const auto &a : agentTimeMap.at(simTime)) {
      if (a->isTransportation()) {
        routeId = a->getBus()->getRouteId();
        if (busPathMap.find(routeId) != busPathMap.end()) {
          std::unordered_map<int, std::shared_ptr<Link>> tmpLinkMap;
          for (const auto &busLink : busPathMap.at(routeId)) {
            tmpLinkMap.emplace(busLink->getId(), busLink);
          }
          a->initializeLowerRL(tmpLinkMap);
        } else {
          a->initializeLowerRL(linkMap);
        }
      } else {
        a->initializeLowerRL(linkMap);
      }
      a->initializeTransRL(busLinkMap);
      a->chooseDestination();
      a->setLowerRL(simTime);
      if (a->getLane() != nullptr) {
        tmpAgents.emplace_back(a);
        std::cout << "Agent " << a->getId() << " is created." << std::endl;
      } else {
        std::cout << "Agent " << a->getId() << " is not created." << std::endl;
      }
    }
    agentTimeMap.erase(simTime);
  }
}

void Simulation::writeAgentOutput(bool destination, bool location) {
  // lane==nullptrのagentは含まれない
  if (destination) writeAgentDestination();
  if (location) writeAgentLocation();
}

void Simulation::writeAgentDestination() {
  //"Time,SimulationTime,AgentID,Mode,StayTime(min),DUtility,ONodeID,DNodeID,OLon,OLat,DLon,DLat,Purpose,LastTrip"
  if (!foutDestination.is_open()) return;
  foutDestinationUsed = true;

  double ox, oy, dx, dy;
  int oNodeId, dNodeId;
  double dUtil;
  for (const auto &agent : tmpAgents) {
    if (agent->destinationChanged()) {
      std::tie(oNodeId, dNodeId) = agent->getOD();
      const auto itrO = nodeMap.find(oNodeId);
      const auto itrD = nodeMap.find(dNodeId);
      if (itrO != nodeMap.end() && itrD != nodeMap.end()) {
        std::tie(ox, oy) =
            Utility::xy2LonLat(itrO->second->getX(), itrO->second->getY());
        std::tie(dx, dy) =
            Utility::xy2LonLat(itrD->second->getX(), itrD->second->getY());
        if (agent->getDestinationAttribute() != nullptr) {
          dUtil = agent->getDestinationAttribute()->getUtility(
              simTime, agent->getMigrationCoefficient());
        } else {
          dUtil = 0.0;
        }
        if (agent->isLastTrip()) {
          foutDestination << (long long)simTime * 1000 + 1602082800000 +
                                 9 * 3600 * 1000
                          << "," << (int)simTime << "," << agent->getId() << ","
                          << agent->getMode() << ","
                          << agent->getStayStep() * (int)timestep / 60 << ","
                          << 0 << "," << oNodeId << "," << dNodeId << ","
                          << std::fixed << std::setprecision(8) << ox << ","
                          << oy << "," << dx << "," << dy << ","
                          << agent->getPurpose() << "," << agent->isLastTrip()
                          << std::endl;
        } else {
          foutDestination << (long long)simTime * 1000 + 1602082800000 +
                                 9 * 3600 * 1000
                          << "," << (int)simTime << "," << agent->getId() << ","
                          << agent->getMode() << ","
                          << agent->getStayStep() * (int)timestep / 60 << ","
                          << dUtil << "," << oNodeId << "," << dNodeId << ","
                          << std::fixed << std::setprecision(8) << ox << ","
                          << oy << "," << dx << "," << dy << ","
                          << agent->getPurpose() << "," << agent->isLastTrip()
                          << std::endl;
        }
      }
    }
  }
}

void Simulation::writeAgentLocation() {
  //"Time,SimulationTime,AgentID,Mode,Lon,Lat,Velocity(km/h),Acceleration(m/s),LinkID,LaneID,X,InNodeLane,CarFollowing,SignalStop,Transportation"
  if (!foutAgentLoc.is_open()) return;
  foutAgentLocUsed = true;

  double x, y;
  for (const auto &agent : tmpAgents) {
    if (agent->isStay() || agent->isStop() || agent->isOnBord()) {
      continue;
    }
    std::tie(x, y) = agent->getGlobalCoord();
    foutAgentLoc << (long long)simTime * 1000 + 1602082800000 + 9 * 3600 * 1000
                 << "," << (int)simTime << "," << agent->getId() << ","
                 << agent->getMode() << "," << std::fixed
                 << std::setprecision(8) << x << "," << y << ","
                 << agent->getVelocity() * 3.6 << ","
                 << agent->getAcceleration() << "," << agent->getLink()->getId()
                 << "," << agent->getLane()->getId() << ","
                 << agent->getPosition().x << "," << agent->inNodeLane() << ","
                 << agent->isCarFollwing() << "," << agent->isSignalStop()
                 << "," << agent->isTransportation() << std::endl;
  }
}

void Simulation::writeAggregateAgentOutput(
    const std::vector<std::shared_ptr<Agent>> &finishAgents) {
  // AgentID,Mode,ONodeID,DNodeID,OLon,OLat,DLon,DLat,StartTime,EndTime,TotalUtility,TotalDistance(m)
  if (!foutAgent.is_open()) return;
  foutAgentUsed = true;

  int oNodeId, dNodeId;
  double oLon, oLat, dLon, dLat;
  for (const auto &agent : finishAgents) {
    std::tie(oNodeId, dNodeId) = agent->getOD();
    const auto oNode = nodeMap.at(oNodeId);
    const auto dNode = nodeMap.at(dNodeId);
    oLon = oNode->getLon();
    oLat = oNode->getLat();
    dLon = dNode->getLon();
    dLat = dNode->getLat();

    foutAgent << agent->getId() << "," << agent->getMode() << "," << oNodeId
              << "," << dNodeId << "," << oLon << "," << oLat << "," << dLon
              << "," << dLat << "," << agent->getStartTime() << "," << simTime
              << "," << agent->getUtility() << "," << agent->getTravelLength()
              << std::endl;
  }
}

void Simulation::writeLinkOutput() {
  //"Time,SimulationTime,LinkID,CarTrafficVolume(OD),CarTrafficVolume(DO),CarVelocity(km/h)(OD),CarVelocity(km/h)(DO),CarVelocity(km/h),TrafficVolume,Velocity(km/h),ONodeID,DNodeID,OLon,OLat,DLon,DLat"
  if (!foutLink.is_open()) return;
  foutLinkUsed = true;

  int trafficVolume;
  std::pair<int, int> carTrafficVolume;
  double velocity, carVelocity, ox, oy, dx, dy;
  std::pair<double, double> carVelocities;

  for (const auto &pair : linkMap) {
    trafficVolume = pair.second->getTrafficVolume();
    carTrafficVolume = pair.second->getCarTrafficVolume();
    velocity = pair.second->getVelocity(3);
    carVelocity = pair.second->getVelocity(0);
    carVelocities = pair.second->getCarVelocities();
    if (trafficVolume > 0 ||
        carTrafficVolume.first + carTrafficVolume.second > 0) {
      std::tie(ox, oy) = pair.second->getLonLat(pair.second->getONodeId());
      std::tie(dx, dy) = pair.second->getLonLat(pair.second->getDNodeId());
      foutLink << (long long)simTime * 1000 + 1602082800000 + 9 * 3600 * 1000
               << "," << (int)simTime << "," << pair.first << ","
               << carTrafficVolume.first << "," << carTrafficVolume.second
               << "," << carVelocities.first * 3.6 << ","
               << carVelocities.second * 3.6 << "," << carVelocity * 3.6 << ","
               << trafficVolume << "," << velocity * 3.6 << ","
               << pair.second->getONodeId() << "," << pair.second->getDNodeId()
               << "," << std::fixed << std::setprecision(8) << ox << "," << oy
               << "," << dx << "," << dy << std::endl;
    }
  }
}

void Simulation::updateNetwork() {
  for (const auto &pair : linkMap) {
    pair.second->clearVelocity();
  }
}

void Simulation::close() {
  if (foutAgentLoc.is_open()) {
    foutAgentLoc.close();
    if (!foutAgentLocUsed) {
      std::remove(destinationPath.c_str());
    } else
      Kepler(agentLocationPath, outputDir);
  }
  if (foutDestination.is_open()) {
    foutDestination.close();
    if (!foutDestinationUsed) std::remove(destinationPath.c_str());
  }
  if (foutLink.is_open()) {
    foutLink.close();
    if (!foutLinkUsed) std::remove(linkOutputPath.c_str());
  }
  if (foutAgent.is_open()) {
    foutAgent.close();
    if (!foutAgentUsed) std::remove(agentOutputPath.c_str());
  }
}

void Simulation::reset() {
  simTime = startTime;
  finish = false;
}
}  // namespace Hongo
