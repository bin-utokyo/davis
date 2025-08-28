cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++)
project(Hongo CXX)
set(CMAKE_GENERATOR "Unix Makefiles")

set(CMAKE_CXX_STANDARD 17)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)
endif()

cmake_policy(SET CMP0076 NEW)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(Simulation)

target_sources(Simulation
    PRIVATE
    ${SRC_DIR}/Agent.cpp
    ${SRC_DIR}/Bus.cpp
    ${SRC_DIR}/ChoiceSet.cpp
    ${SRC_DIR}/Dijkstra.cpp
    ${SRC_DIR}/Kepler.cpp
    ${SRC_DIR}/Lane.cpp
    ${SRC_DIR}/Link.cpp
    ${SRC_DIR}/Node.cpp
    ${SRC_DIR}/Pedestrian.cpp
    ${SRC_DIR}/Physarum.cpp
    ${SRC_DIR}/RL.cpp
    ${SRC_DIR}/Signal.cpp
    ${SRC_DIR}/Simulation.cpp
    ${SRC_DIR}/SparseMat.cpp
    ${SRC_DIR}/Station.cpp
    ${SRC_DIR}/Table.cpp
    ${SRC_DIR}/Utility.cpp
    ${SRC_DIR}/Vehicle.cpp
)
add_executable(Hongo
    ${SRC_DIR}/main.cpp
)
target_link_libraries(Hongo Simulation)

##for python
set(PYTHON ON)

if(${PYTHON})

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${SRC_DIR})

set(CMAKE_SWIG_FLAGS "")

set_source_files_properties(${SRC_DIR}/Simulation.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${SRC_DIR}/Simulation.i PROPERTIES SWIG_FLAGS "-includeall")
swig_add_library(hongo
    LANGUAGE python
    SOURCES ${SRC_DIR}/Simulation.i
    ${SRC_DIR}/Agent.cpp
    ${SRC_DIR}/Bus.cpp
    ${SRC_DIR}/ChoiceSet.cpp
    ${SRC_DIR}/Dijkstra.cpp
    ${SRC_DIR}/Kepler.cpp
    ${SRC_DIR}/Lane.cpp
    ${SRC_DIR}/Link.cpp
    ${SRC_DIR}/Node.cpp
    ${SRC_DIR}/Pedestrian.cpp
    ${SRC_DIR}/Physarum.cpp
    ${SRC_DIR}/RL.cpp
    ${SRC_DIR}/Signal.cpp
    ${SRC_DIR}/Simulation.cpp
    ${SRC_DIR}/SparseMat.cpp
    ${SRC_DIR}/Station.cpp
    ${SRC_DIR}/Table.cpp
    ${SRC_DIR}/Utility.cpp
    ${SRC_DIR}/Vehicle.cpp
)
swig_link_libraries(hongo ${PYTHON_LIBRARIES})
endif()
