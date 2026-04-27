#!/usr/bin/env python3
"""
Joint States Logger: logs pendulum joint state and latest /lowcmd leg targets.

Output: /workspace/ros2_ws/src/go2_pendulum/go2_bridge/data/<timestamp>.csv
"""

import csv
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from unitree_go.msg import LowCmd


OUTPUT_DIR = Path('/workspace/ros2_ws/src/go2_pendulum/go2_bridge/data')

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

LOGGED_JOINTS = ['pendulum_joint1', 'pendulum_joint2']
LOWCMD_JOINT_MAP = {
    'FR_hip_joint': 0,
    'FR_thigh_joint': 1,
    'FR_calf_joint': 2,
    'FL_hip_joint': 3,
    'FL_thigh_joint': 4,
    'FL_calf_joint': 5,
    'RR_hip_joint': 6,
    'RR_thigh_joint': 7,
    'RR_calf_joint': 8,
    'RL_hip_joint': 9,
    'RL_thigh_joint': 10,
    'RL_calf_joint': 11,
}
LOWCMD_JOINT_NAMES = sorted(LOWCMD_JOINT_MAP.keys(), key=lambda n: LOWCMD_JOINT_MAP[n])


class JointStatesLoggerNode(Node):
    def __init__(self):
        super().__init__('joint_states_logger_node')

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        self._csv_path = OUTPUT_DIR / filename

        self._file = open(self._csv_path, 'w', newline='')
        self._writer = csv.writer(self._file)

        header = ['timestamp']
        for n in LOGGED_JOINTS:
            header.append(f'{n}_pos')
            header.append(f'{n}_vel')
        for n in LOWCMD_JOINT_NAMES:
            header.append(f'{n}_lowcmd_target')
        self._writer.writerow(header)
        self._latest_lowcmd_targets = {n: '' for n in LOWCMD_JOINT_NAMES}

        self._sub = self.create_subscription(
            JointState, '/joint_states', self._cb, SENSOR_QOS)
        self._lowcmd_sub = self.create_subscription(
            LowCmd, '/lowcmd', self._lowcmd_cb, SENSOR_QOS)

        self.get_logger().info(
            f'Logging /joint_states and /lowcmd targets to {self._csv_path}')

    def _lowcmd_cb(self, msg: LowCmd):
        for name, idx in LOWCMD_JOINT_MAP.items():
            self._latest_lowcmd_targets[name] = float(msg.motor_cmd[idx].q)

    def _cb(self, msg: JointState):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        row = [t]
        for n in LOGGED_JOINTS:
            idx = name_to_idx.get(n)
            if idx is None:
                row.extend(['', ''])
                continue
            pos = msg.position[idx] if idx < len(msg.position) else ''
            vel = msg.velocity[idx] if idx < len(msg.velocity) else ''
            row.extend([pos, vel])
        for n in LOWCMD_JOINT_NAMES:
            row.append(self._latest_lowcmd_targets[n])
        self._writer.writerow(row)

    def destroy_node(self):
        self._file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JointStatesLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
