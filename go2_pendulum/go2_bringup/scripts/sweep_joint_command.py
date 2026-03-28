#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState


DEFAULT_HARD_JOINT_LIMITS_BY_NAME = {
    "FL_hip_joint": (-1.0472, 1.0472),
    "FL_thigh_joint": (-1.5708, 3.4907),
    "FL_calf_joint": (-2.7227, -0.83776),
    "FR_hip_joint": (-1.0472, 1.0472),
    "FR_thigh_joint": (-1.5708, 3.4907),
    "FR_calf_joint": (-2.7227, -0.83776),
    "RL_hip_joint": (-1.0472, 1.0472),
    "RL_thigh_joint": (-0.5236, 4.5379),
    "RL_calf_joint": (-2.7227, -0.83776),
    "RR_hip_joint": (-1.0472, 1.0472),
    "RR_thigh_joint": (-0.5236, 4.5379),
    "RR_calf_joint": (-2.7227, -0.83776),
}


class JointSweepPublisher(Node):
    def __init__(self) -> None:
        super().__init__("joint_sweep_publisher")

        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("sweep_period_sec", 20.0)
        self.declare_parameter(
            "leg_joint_names",
            [
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "RL_thigh_joint",
                "RR_thigh_joint",
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
            ],
        )

        self.joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        self.publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.sweep_period_sec = max(1e-3, float(self.get_parameter("sweep_period_sec").value))
        self.leg_joint_names = [str(x) for x in self.get_parameter("leg_joint_names").value]

        if len(self.leg_joint_names) != 12:
            raise ValueError(f"leg_joint_names must be length 12, got {len(self.leg_joint_names)}")

        for name in self.leg_joint_names:
            if name not in DEFAULT_HARD_JOINT_LIMITS_BY_NAME:
                raise ValueError(f"Unsupported joint name in sweep list: {name}")

        self.joint_lows = [DEFAULT_HARD_JOINT_LIMITS_BY_NAME[name][0] for name in self.leg_joint_names]
        self.joint_highs = [DEFAULT_HARD_JOINT_LIMITS_BY_NAME[name][1] for name in self.leg_joint_names]

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(JointState, self.joint_command_topic, qos)

        self.start_sec = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(1.0 / self.publish_rate_hz, self._on_timer)

        self.get_logger().info(
            "Publishing joint sweep on %s at %.2f Hz, period %.2f s (min->max->min)"
            % (self.joint_command_topic, self.publish_rate_hz, self.sweep_period_sec)
        )
        self.get_logger().info(f"Joint order: {self.leg_joint_names}")

    def _triangle_0_to_1(self, t: float) -> float:
        phase = (t / self.sweep_period_sec) % 1.0
        return 1.0 - abs(2.0 * phase - 1.0)

    def _on_timer(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        t = max(0.0, now_sec - self.start_sec)
        u = self._triangle_0_to_1(t)

        positions = [self.joint_lows[i] + (self.joint_highs[i] - self.joint_lows[i]) * u for i in range(12)]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.leg_joint_names)
        msg.position = positions
        msg.velocity = [0.0] * 12
        msg.effort = [0.0] * 12
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointSweepPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
