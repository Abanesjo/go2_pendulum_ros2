#!/bin/bash
source /opt/ros/humble/setup.bash

cd /workspace/ros2_ws
colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))
source /workspace/ros2_ws/install/setup.bash

echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export CYCLONEDDS_URI=file:///workspace/dependencies/cyclonedds.xml" >> ~/.bashrc

cd /workspace/ros2_ws/src

exec bash
