#!/usr/bin/env python3

import csv
import os
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray


def _sanitize_depth(depth: int) -> int:
    return depth if depth > 0 else 1


class PolicyDebugCsvLogger(Node):
    def __init__(self) -> None:
        super().__init__("policy_debug_csv_logger")

        self.declare_parameter("observation_topic", "/debug/policy_observation")
        self.declare_parameter("commanded_joint_position_topic", "/debug/commanded_joint_position")
        self.declare_parameter("actual_joint_position_topic", "/debug/actual_joint_position")
        self.declare_parameter("output_csv", "")
        self.declare_parameter("write_rate_hz", 60.0)
        self.declare_parameter("qos_depth", 0)
        self.declare_parameter("obs_dim", 56)
        self.declare_parameter("joint_dim", 12)
        self.declare_parameter("obs_prev_action_start", 40)
        self.declare_parameter("obs_prev_action_dim", 12)
        self.declare_parameter("flush_every_n", 25)

        self.observation_topic = str(self.get_parameter("observation_topic").value)
        self.commanded_joint_position_topic = str(self.get_parameter("commanded_joint_position_topic").value)
        self.actual_joint_position_topic = str(self.get_parameter("actual_joint_position_topic").value)
        self.write_rate_hz = max(0.1, float(self.get_parameter("write_rate_hz").value))
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.obs_dim = max(1, int(self.get_parameter("obs_dim").value))
        self.joint_dim = max(1, int(self.get_parameter("joint_dim").value))
        self.obs_prev_action_start = int(self.get_parameter("obs_prev_action_start").value)
        self.obs_prev_action_dim = max(1, int(self.get_parameter("obs_prev_action_dim").value))
        self.flush_every_n = max(1, int(self.get_parameter("flush_every_n").value))

        self._lock = threading.Lock()
        self._latest_obs = None
        self._latest_commanded_joint_position = None
        self._latest_actual_joint_position = None
        self._write_count = 0

        output_csv = str(self.get_parameter("output_csv").value).strip()
        if not output_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"/tmp/go2_policy_debug_{ts}.csv"
        output_csv = os.path.abspath(os.path.expandvars(os.path.expanduser(output_csv)))
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.output_csv = output_csv

        self._csv_file = open(self.output_csv, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._build_header())
        self._csv_file.flush()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=_sanitize_depth(self.qos_depth),
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Float32MultiArray,
            self.observation_topic,
            self._on_observation,
            qos,
        )
        self.create_subscription(
            Float32MultiArray,
            self.commanded_joint_position_topic,
            self._on_commanded_joint_position,
            qos,
        )
        self.create_subscription(
            Float32MultiArray,
            self.actual_joint_position_topic,
            self._on_actual_joint_position,
            qos,
        )

        self.create_timer(1.0 / self.write_rate_hz, self._on_write_timer)

        self.get_logger().info(f"Writing policy debug CSV to: {self.output_csv}")
        self.get_logger().info(
            "Subscribing to: %s, %s, %s"
            % (self.observation_topic, self.commanded_joint_position_topic, self.actual_joint_position_topic)
        )
        self.get_logger().info(
            "write_rate_hz=%.2f, qos_depth=%d -> qos_depth_effective=%d"
            % (self.write_rate_hz, self.qos_depth, _sanitize_depth(self.qos_depth))
        )

    def _build_header(self) -> list[str]:
        header = ["time_sec", "obs_msg_len", "commanded_joint_position_msg_len", "actual_joint_position_msg_len"]
        header.extend([f"obs_{i}" for i in range(self.obs_dim)])
        header.extend([f"obs_prev_action_{i}" for i in range(self.obs_prev_action_dim)])
        header.extend([f"commanded_joint_position_{i}" for i in range(self.joint_dim)])
        header.extend([f"actual_joint_position_{i}" for i in range(self.joint_dim)])
        header.extend([f"commanded_minus_actual_joint_position_{i}" for i in range(self.joint_dim)])
        return header

    @staticmethod
    def _pad_or_truncate(values: list[float], dim: int) -> list[float]:
        if len(values) >= dim:
            return [float(v) for v in values[:dim]]
        out = [float(v) for v in values]
        out.extend([float("nan")] * (dim - len(out)))
        return out

    def _extract_prev_action_from_obs(self, obs_values: list[float]) -> list[float]:
        out = []
        for i in range(self.obs_prev_action_dim):
            idx = self.obs_prev_action_start + i
            if 0 <= idx < len(obs_values):
                out.append(float(obs_values[idx]))
            else:
                out.append(float("nan"))
        return out

    def _on_observation(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._latest_obs = list(msg.data)

    def _on_commanded_joint_position(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._latest_commanded_joint_position = list(msg.data)

    def _on_actual_joint_position(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._latest_actual_joint_position = list(msg.data)

    def _on_write_timer(self) -> None:
        with self._lock:
            obs_values = self._latest_obs
            commanded_joint_position = self._latest_commanded_joint_position
            actual_joint_position = self._latest_actual_joint_position

        if obs_values is None or commanded_joint_position is None or actual_joint_position is None:
            return

        obs_row = self._pad_or_truncate(obs_values, self.obs_dim)
        prev_action_from_obs = self._extract_prev_action_from_obs(obs_values)
        commanded_row = self._pad_or_truncate(commanded_joint_position, self.joint_dim)
        actual_row = self._pad_or_truncate(actual_joint_position, self.joint_dim)
        diff_row = [commanded_row[i] - actual_row[i] for i in range(self.joint_dim)]

        time_sec = self.get_clock().now().nanoseconds * 1e-9
        row = [
            time_sec,
            len(obs_values),
            len(commanded_joint_position),
            len(actual_joint_position),
        ]
        row.extend(obs_row)
        row.extend(prev_action_from_obs)
        row.extend(commanded_row)
        row.extend(actual_row)
        row.extend(diff_row)
        self._csv_writer.writerow(row)
        self._write_count += 1

        if self._write_count % self.flush_every_n == 0:
            self._csv_file.flush()

    def destroy_node(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyDebugCsvLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
