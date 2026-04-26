#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

class JoyOrchestratorNode(Node):
    def __init__(self):
        super().__init__('joy_orchestrator_node')

        self.declare_parameter('start_button_index', 10)
        self.declare_parameter('damp_button_index', 9)
        self._start_button_index = int(
            self.get_parameter('start_button_index').value)
        self._damp_button_index = int(
            self.get_parameter('damp_button_index').value)

        self._prev_start_pressed = False
        self._prev_damp_pressed = False
        self._damp_latched = False

        self._policy_start_pub = self.create_publisher(
            Bool, '/policy_start_request', SENSOR_QOS)
        self._emergency_damp_pub = self.create_publisher(
            Bool, '/emergency_damp', SENSOR_QOS)
        self._joy_sub = self.create_subscription(
            Joy, '/joy', self._joy_cb, SENSOR_QOS)

        self.get_logger().info(
            'Joy orchestrator ready '
            f'(start=button[{self._start_button_index}], '
            f'damp=button[{self._damp_button_index}])')

    def _button_pressed(self, msg, index):
        return index < len(msg.buttons) and msg.buttons[index] == 1

    def _joy_cb(self, msg):
        start_pressed = self._button_pressed(msg, self._start_button_index)
        damp_pressed = self._button_pressed(msg, self._damp_button_index)

        if damp_pressed and not self._prev_damp_pressed and not self._damp_latched:
            self._damp_latched = True
            out = Bool()
            out.data = True
            self._emergency_damp_pub.publish(out)
            self.get_logger().fatal(
                f'Emergency damping requested by button[{self._damp_button_index}]')

        if start_pressed and not self._prev_start_pressed and not self._damp_latched:
            out = Bool()
            out.data = True
            self._policy_start_pub.publish(out)
            self.get_logger().info(
                f'Policy start requested by button[{self._start_button_index}]')

        self._prev_start_pressed = start_pressed
        self._prev_damp_pressed = damp_pressed


def main(args=None):
    rclpy.init(args=args)
    node = JoyOrchestratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
