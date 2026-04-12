cmake_minimum_required(VERSION 3.5)
project(go2_navigation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/waypoint_manager_node.py
  RENAME waypoint_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
