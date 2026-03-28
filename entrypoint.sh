#!/bin/bash
source /opt/ros/humble/setup.bash
cd /workspace

# Prevent colcon from rebuilding dependencies already installed in the image
touch /workspace/src/unitree_ros2/dependencies/cyclonedds/COLCON_IGNORE
touch /workspace/src/unitree_ros2/dependencies/unitree_sdk2/COLCON_IGNORE
touch /workspace/src/unitree_ros2/dependencies/unitree_sdk2_python/COLCON_IGNORE

colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))
source /workspace/install/setup.bash

echo "source /workspace/install/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export CYCLONEDDS_URI=file:///workspace/src/unitree_ros2/cyclonedds.xml" >> ~/.bashrc

exec bash
