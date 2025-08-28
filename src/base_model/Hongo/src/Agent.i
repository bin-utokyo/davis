%module hongo
%include "std_string.i"
%include "std_unordered_map.i"
%include "std_pair.i"
%include <std_shared_ptr.i>

%include <Node.i>

%shared_ptr(Hongo::Node);
%shared_ptr(Hongo::Agent);
%shared_ptr(Hongo::MigrationAttribute);

%{
#define SWIG_FILE_WITH_INIT
#include "Agent.hpp"
#include "RL.hpp"
#include "Link.hpp"
#include "Lane.hpp"
#include "Node.hpp"
#include "Pedestrian.hpp"
#include "Vehicle.hpp"
#include "Bus.hpp"
#include "Station.hpp"
#include "Utility.hpp"
%}

namespace std{
%template(vector_sp_Agent) vector<shared_ptr<Hongo::Agent>>;
}

namespace Hongo{
class MigrationAttribute{
public:
    MigrationAttribute(const std::shared_ptr<Node> &_node, int _startTime, int _endTime, double _spa, double _shop, double _famous, double _art);
};
    
class Agent{
public:
    Agent(int _id, int _startTime, int _endTime, std::string _purpose, int _mode, const std::shared_ptr<Node> &oNode, const std::shared_ptr<Node> &dNode,  const std::unordered_map<int, std::shared_ptr<MigrationAttribute>> &_facilityMap);
    Agent(int _id, int _startTime, int _mode, const std::shared_ptr<Node> &oNode, const std::shared_ptr<Node> &dNode);//公共交通など用
    
    int getMode();
    std::pair<int, int> getOD();
};
}

