//
//  main.cpp
//  Pedestrian
//
//  Created by 小川大智 on 2021/10/06.
//

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "Simulation.hpp"

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <inputDir> <outputDir>" << std::endl;
    return 1;
  }
  std::string inputDir = argv[1];
  std::string outputDir = argv[2];
  double timestep = 1.0;
  double simulationStartTime = 3600.0 * 0;
  double simulationEndTime = 3600.0 * 16;

  int planeNum = 4;  // 平面直角座標の番号

  int outputInterval = 1;
  int update_interval =
      600 /
      timestep;  // 時空間ネットワークおよび旅行時間更新タイミングの時間間隔（ステップ）

  Hongo::Simulation sim(outputDir, simulationStartTime, simulationEndTime,
                        timestep, false, planeNum);
  std::cout << "Reading data start." << std::endl;

  sim.readData(inputDir);
  sim.initialize();

  int stepNum = 0;

  while (!sim.isFinish()) {
    sim.calculation();
    if (stepNum % outputInterval == 0) {
      sim.writeAgentOutput(true, true);  // destination, location
    }
    if (stepNum % update_interval == 0) {
      sim.writeLinkOutput();
      sim.updateNetwork();
    }
    stepNum++;
  }
  sim.close();

  return 0;
}
