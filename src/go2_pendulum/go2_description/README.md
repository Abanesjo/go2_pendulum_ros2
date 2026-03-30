# go2_description
ROS 2 description package for the Unitree Go2.

## Build (ROS 2)
```bash
colcon build --packages-select go2_description
source install/setup.bash
```

## Run
```bash
ros2 launch go2_description go2_description.launch.xml
```

This launches `robot_state_publisher` with `robot_description` loaded from
`urdf/go2_description.urdf`. The node subscribes to `/joint_states` and publishes
TF for the model.

## Notes
- URDF: `urdf/go2_description.urdf`
- Xacro sources: `xacro/`
