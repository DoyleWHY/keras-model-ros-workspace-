cmake_minimum_required(VERSION 2.8.3)
project(pythonpacktest)

set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pythonpacktest
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

########################################################
# python2 configuration
########################################################
## directory of python headers
#include_directories(${PYTHON_INCLUDE_DIRS})
#include_directories("./")

#link_directories(${PYTHON_LIBRARY_DIRS})

#include_directories(${PYTHON_INCLUDE_DIRS})
#include_directories(${PYTHON_DIRECTORIES})

#add_executable(main src/main.cpp)
#add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(main ${catkin_LIBRARIES} ${PYTHON_LIBRARIES})


########################################################
# python3 configuration
########################################################
include_directories(/usr/include/python3.5m)#添加头文件目录，相当于g++ -I参数
link_directories(/usr/lib/python3.5)#动态链接库或静态链接库的搜索路径，相当于gcc的-L参数
## main_gui节点的构建选项。
add_executable(main src/main.cpp)
add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(main ${catkin_LIBRARIES} python3.5m)#添加链接库,相同于指定gcc -l参数
