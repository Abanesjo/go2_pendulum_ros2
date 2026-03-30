cmake_minimum_required(VERSION 3.8)
project(go2_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(
  DIRECTORY launch config rviz model
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/odom_to_tf.py
  scripts/imu_to_projected_gravity.py
  scripts/publish_gait_cycle.py
  scripts/rl_controller.py
  scripts/sweep_joint_command.py
  scripts/record_policy_debug.py
  scripts/plot_policy_debug_csv.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
