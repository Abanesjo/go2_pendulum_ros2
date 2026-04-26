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

        self.declare_parameter('policy_toggle_button_index', 10)
        self.declare_parameter('stand_button_index', 11)
        self.declare_parameter('sit_button_index', 12)
        self.declare_parameter('damp_button_index', 9)
        self._policy_toggle_button_index = int(
            self.get_parameter('policy_toggle_button_index').value)
        self._stand_button_index = int(
            self.get_parameter('stand_button_index').value)
        self._sit_button_index = int(
            self.get_parameter('sit_button_index').value)
        self._damp_button_index = int(
            self.get_parameter('damp_button_index').value)

        self._prev_policy_toggle_pressed = False
        self._prev_stand_pressed = False
        self._prev_sit_pressed = False
        self._prev_damp_pressed = False
        self._damp_latched = False

        self._policy_toggle_pub = self.create_publisher(
            Bool, '/policy_toggle_request', SENSOR_QOS)
        self._stand_pub = self.create_publisher(
            Bool, '/stand_request', SENSOR_QOS)
        self._sit_pub = self.create_publisher(
            Bool, '/sit_request', SENSOR_QOS)
        self._emergency_damp_pub = self.create_publisher(
            Bool, '/emergency_damp', SENSOR_QOS)
        self._joy_sub = self.create_subscription(
            Joy, '/joy', self._joy_cb, SENSOR_QOS)

        self.get_logger().info(
            'Joy orchestrator ready '
            f'(policy_toggle=button[{self._policy_toggle_button_index}], '
            f'stand=button[{self._stand_button_index}], '
            f'sit=button[{self._sit_button_index}], '
            f'damp=button[{self._damp_button_index}])')

    def _button_pressed(self, msg, index):
        return index < len(msg.buttons) and msg.buttons[index] == 1

    def _publish_event(self, publisher):
        out = Bool()
        out.data = True
        publisher.publish(out)

    def _joy_cb(self, msg):
        policy_toggle_pressed = self._button_pressed(
            msg, self._policy_toggle_button_index)
        stand_pressed = self._button_pressed(msg, self._stand_button_index)
        sit_pressed = self._button_pressed(msg, self._sit_button_index)
        damp_pressed = self._button_pressed(msg, self._damp_button_index)

        if damp_pressed and not self._prev_damp_pressed and not self._damp_latched:
            self._damp_latched = True
            self._publish_event(self._emergency_damp_pub)
            self.get_logger().fatal(
                f'Emergency damping requested by button[{self._damp_button_index}]')

        if self._damp_latched:
            self._prev_policy_toggle_pressed = policy_toggle_pressed
            self._prev_stand_pressed = stand_pressed
            self._prev_sit_pressed = sit_pressed
            self._prev_damp_pressed = damp_pressed
            return

        if policy_toggle_pressed and not self._prev_policy_toggle_pressed:
            self._publish_event(self._policy_toggle_pub)
            self.get_logger().info(
                'Policy toggle requested by '
                f'button[{self._policy_toggle_button_index}]')

        if stand_pressed and not self._prev_stand_pressed:
            self._publish_event(self._stand_pub)
            self.get_logger().info(
                f'Stand requested by button[{self._stand_button_index}]')

        if sit_pressed and not self._prev_sit_pressed:
            self._publish_event(self._sit_pub)
            self.get_logger().info(
                f'Sit requested by button[{self._sit_button_index}]')

        self._prev_policy_toggle_pressed = policy_toggle_pressed
        self._prev_stand_pressed = stand_pressed
        self._prev_sit_pressed = sit_pressed
        self._prev_damp_pressed = damp_pressed


def main(args=None):
    rclpy.init(args=args)
    node = JoyOrchestratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
