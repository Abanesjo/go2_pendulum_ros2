# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS2 Humble workspace for controlling a Unitree Go2 quadruped robot with an inverted pendulum attachment. Combines RL policy control (ONNX Runtime, C++), Unitree Go2 hardware interfaces, and MuJoCo/ISAAC Gym simulations.

## Build & Run

**Docker (primary workflow):**
```bash
./build_and_run.sh   # builds image and runs container with GPU, X11, host networking
```

**Inside the container (or local with ROS2 Humble):**
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///workspace/dependencies/cyclonedds.xml
```

**Build a single package:**
```bash
colcon build --packages-select go2_bringup
```

**Launch with Isaac Sim (no bridge):**
```bash
ros2 launch go2_bringup bringup.launch.xml
```

**Launch with Unitree MuJoCo sim (with bridge):**
```bash
ros2 launch go2_bringup bringup.launch.xml bridge:=true
```

**IMPORTANT: Do not build on the host machine.** The host runs a different ROS version. All builds happen inside the Docker container.

## Architecture

### Topic Flow

Two simulator modes controlled by `bridge` launch arg:

- **Isaac Sim (bridge=false):** Sim publishes `/joint_states`, `/imu`, `/pose/base_link`, `/pose/pendulum_ee`. RL controller publishes `/joint_commands` at 500Hz.
- **Unitree MuJoCo (bridge=true):** Sim uses `/lowstate`/`/lowcmd`. The `go2_bridge` node converts between Unitree messages and standard ROS2 topics. Sim also publishes `/pose/base_link`, `/pose/pendulum_ee`.

All topics use BEST_EFFORT QoS, depth 1. No sim_time — always wall clock.

### Packages (under `src/go2_pendulum/`)

- **go2_bringup** — Main package. C++ RL controller node (`src/rl_controller_node.cpp`) with ONNX inference at 50Hz, publishing `/joint_commands` at 500Hz. Subscribes to `/joint_states`, `/imu`, `/pose/base_link`, `/pose/pendulum_ee`. Config in `config/config.yaml`.
- **go2_bridge** — Python bridge between Unitree Go2 `/lowstate`/`/lowcmd` and ROS2 `/joint_states`/`/joint_commands`/`/imu`. Has CRC computation for Go2 LowCmd (toggled via `enable_crc` param). Gains in `config/gains.yaml` (default kp=25.0, kd=0.6). Republishes `/lowcmd` at 500Hz.
- **go2_controller** — C++ ROS2 services wrapping Unitree SDK2: `/go2_sport_mode` and `/go2_damp`. For real robot control.
- **go2_interfaces** — Custom service: `SetGoal.srv`.
- **go2_description** — URDF/xacro model for the Go2 robot with meshes.

### RL Controller Key Details

The C++ RL controller (`go2_bringup/src/rl_controller_node.cpp`):
- 56-dim observation, 12-dim action (leg joints only)
- ONNX Runtime inference (policy at `go2_bringup/model/policy.onnx`)
- Action pipeline: clamp -> LPF (8Hz cutoff) -> clamp -> scale (0.25) -> hard clamp
- Base linear velocity: numerical differentiation of `/pose/base_link`, rotated to body frame
- Base angular velocity: from `/imu` gyroscope
- Pendulum angles: XY intrinsic Euler from relative rotation between `/pose/base_link` and `/pose/pendulum_ee`
- Gait clock: freq=3.0, phase=0.5, 4 sinusoidal foot indices
- Standup phase (3s default) interpolates to default pose before policy activates
- Services: `/set_goal`, `/toggle_policy_mode`

### Go2 Motor Index Mapping (for bridge)

```
FR_hip=0, FR_thigh=1, FR_calf=2
FL_hip=3, FL_thigh=4, FL_calf=5
RR_hip=6, RR_thigh=7, RR_calf=8
RL_hip=9, RL_thigh=10, RL_calf=11
```

### Dependencies (git submodules under `dependencies/`)

- **unitree_sdk2** — C++ SDK, built with CMake and installed system-wide before colcon build.
- **unitree_sdk2_python** — Python SDK, installed via pip with `CYCLONEDDS_HOME=/usr/local`.
- **cyclonedds** — DDS middleware, also built and installed before colcon.
- **ONNX Runtime** — C++ library installed in Docker from pre-built release.

### DDS Configuration

Uses CycloneDDS as RMW implementation (required for Unitree SDK2 communication). Config at `dependencies/cyclonedds.xml`. The `--network host` Docker flag is required for robot communication.
