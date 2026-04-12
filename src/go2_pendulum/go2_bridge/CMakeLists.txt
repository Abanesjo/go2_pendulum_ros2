cmake_minimum_required(VERSION 3.5)
project(go2_bridge)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/go2_bridge_node.py
  RENAME go2_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
