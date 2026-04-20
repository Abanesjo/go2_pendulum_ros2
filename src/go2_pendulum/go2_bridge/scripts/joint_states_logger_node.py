#!/usr/bin/env python3
"""
Joint States Logger: subscribes to /joint_states and writes each message to a CSV.

Output: /workspace/ros2_ws/src/go2_pendulum/go2_bridge/data/<timestamp>.csv
"""

import csv
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState


OUTPUT_DIR = Path('/workspace/ros2_ws/src/go2_pendulum/go2_bridge/data')

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class JointStatesLoggerNode(Node):
    def __init__(self):
        super().__init__('joint_states_logger_node')

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        self._csv_path = OUTPUT_DIR / filename

        self._file = open(self._csv_path, 'w', newline='')
        self._writer = csv.writer(self._file)

        self._joint_names = None

        self._sub = self.create_subscription(
            JointState, '/joint_states', self._cb, SENSOR_QOS)

        self.get_logger().info(f'Logging /joint_states to {self._csv_path}')

    def _cb(self, msg: JointState):
        if self._joint_names is None:
            self._joint_names = list(msg.name)
            header = ['timestamp']
            for n in self._joint_names:
                header.append(f'{n}_pos')
                header.append(f'{n}_vel')
            self._writer.writerow(header)

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        row = [t]
        for n in self._joint_names:
            idx = name_to_idx.get(n)
            if idx is None:
                row.extend(['', ''])
                continue
            pos = msg.position[idx] if idx < len(msg.position) else ''
            vel = msg.velocity[idx] if idx < len(msg.velocity) else ''
            row.extend([pos, vel])
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
