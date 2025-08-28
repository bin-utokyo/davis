%module hongo//run python setup.py build_ext --inplace
%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"
%include <std_shared_ptr.i>
%include <Agent.i>

%{
#define SWIG_FILE_WITH_INIT
#include "Simulation.hpp"
#include "Node.hpp"
#include "Link.hpp"
#include "Lane.hpp"
#include "Signal.hpp"
#include "Station.hpp"
#include "Agent.hpp"
#include "Bus.hpp"
#include "Utility.hpp"
#include "Kepler.hpp"
%}

namespace std{
    %template(p) pair<int, int>;
    %template(pp) pair<int, pair<int, int>>;
    %template(vpp) vector<pair<int, pair<int, int>>>;
    %template(TmpAgentVec) std::vector<std::pair<std::pair<int, std::pair<int, int>>, std::pair<int, int>>>;
}


namespace Hongo{
class Simulation{
public:
    Simulation(std::string _outputDir, double _startTime, double _endTime, double _timestep, bool _noOutput, int rectPlaneNum);
    void readData(std::string _inputDir);
    void addAgent(std::string agentPath);
    std::vector<std::pair<std::pair<int, std::pair<int, int>>, std::pair<int, int>>>getTmpAgents();
    void initialize();
    std::vector<std::pair<int, std::pair<int, int>>> calculation();
    bool isFinish();
    void close();
    
    void writeAgentOutput(bool destination, bool location);
    void writeAgentDestination();
    void writeAgentLocation();
    void writeAggregateAgentOutput(std::vector<std::shared_ptr<Agent>> agents);
    
    void writeLinkOutput();
    void updateNetwork();
};
}
