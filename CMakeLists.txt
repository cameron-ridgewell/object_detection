cmake_minimum_required(VERSION 2.4.3)
project(object_detection)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_detection
#  CATKIN_DEPENDS cv_bridge image_transport opencv2 roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories(
  ${Open_CV_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(image_converter_node src/image_converter_node.cpp)


target_link_libraries(image_converter_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
