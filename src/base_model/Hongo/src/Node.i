%module hongo
%include <std_shared_ptr.i>

%shared_ptr(Hongo::Node);

%{
#define SWIG_FILE_WITH_INIT
#include "Node.hpp"
#include "Link.hpp"
#include "Utility.hpp"

%}

namespace std{
}

namespace Hongo{
class Node{
public:
    Node(int _id, double _lon, double _lat);
    Node(int _id, double _lon, double _lat, double _x, double _y);
};
}


