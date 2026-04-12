cmake_minimum_required(VERSION 3.14)
project(go2_bringup)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(go2_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

# C++ RL controller node
add_executable(rl_controller_node src/rl_controller_node.cpp)
ament_target_dependencies(rl_controller_node
  rclcpp sensor_msgs geometry_msgs std_msgs std_srvs go2_interfaces)
target_link_libraries(rl_controller_node onnxruntime)

install(TARGETS rl_controller_node DESTINATION lib/${PROJECT_NAME})

# Directories
install(
  DIRECTORY launch config rviz model
  DESTINATION share/${PROJECT_NAME}
)

# Utility Python scripts (kept for debugging/testing)
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/sweep_joint_command.py
  scripts/publish_gait_cycle.py
  scripts/record_policy_debug.py
  scripts/plot_policy_debug_csv.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
