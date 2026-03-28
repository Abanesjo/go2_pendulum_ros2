#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class PublishGaitCycleNode(Node):
    def __init__(self):
        super().__init__("publish_gait_cycle")

        self.declare_parameter("clock_topic", "/clock_inputs")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("frequency", 3.0)
        self.declare_parameter("phase", 0.5)
        self.declare_parameter("offset", 0.0)
        self.declare_parameter("bound", 0.0)
        self.declare_parameter("duration", 0.5)

        self.clock_topic = self.get_parameter("clock_topic").value
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.frequency = float(self.get_parameter("frequency").value)
        self.phase = float(self.get_parameter("phase").value)
        self.offset = float(self.get_parameter("offset").value)
        self.bound = float(self.get_parameter("bound").value)
        self.duration = float(self.get_parameter("duration").value)

        self.start_time_sec = self.get_clock().now().nanoseconds * 1e-9
        self.publisher = self.create_publisher(Float32MultiArray, self.clock_topic, 10)
        self.create_timer(1.0 / self.publish_rate_hz, self.publish_clock_inputs)

    def _remap_phase(self, x: float) -> float:
        r = x % 1.0
        if r < self.duration:
            return r * (0.5 / self.duration)
        return 0.5 + (r - self.duration) * (0.5 / (1.0 - self.duration))

    def publish_clock_inputs(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        gait_index = ((now_sec - self.start_time_sec) * self.frequency) % 1.0

        foot_indices = [
            gait_index + self.phase + self.offset + self.bound,  # FL
            gait_index + self.offset,  # FR
            gait_index + self.bound,  # RL
            gait_index + self.phase,  # RR
        ]
        mapped = [self._remap_phase(x) for x in foot_indices]

        msg = Float32MultiArray()
        msg.data = [float(math.sin(2.0 * math.pi * x)) for x in mapped]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PublishGaitCycleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
