#!/usr/bin/env python3
"""
Go2 Bridge (passive): Vicon-only bridge for when the robot is deactivated.

Subscriptions:
  /pose/base_link   (geometry_msgs/PoseStamped) -> TF world->base, pendulum angle input
  /pose/pendulum_ee (geometry_msgs/PoseStamped) -> pendulum angle input

Publishes /joint_states on a timer: leg joints locked to a configurable crouched
pose, pendulum joints from Vicon. No /lowstate, no /lowcmd.
"""

import math
from collections import deque

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from scipy.signal import savgol_coeffs
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


def _quat_to_rotation_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y*y + z*z),     2 * (x*y - z*w),     2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x*x + z*z),     2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

JOINT_MAP = {
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

JOINT_NAMES = sorted(JOINT_MAP.keys(), key=lambda n: JOINT_MAP[n])

PENDULUM_JOINT_NAMES = ['pendulum_joint1', 'pendulum_joint2']


class Go2BridgePassiveNode(Node):
    def __init__(self):
        super().__init__('go2_bridge_passive_node')

        self._default_joint_pos = []
        for name in JOINT_NAMES:
            self.declare_parameter(f'default_joint_pos.{name}', 0.0)
            self._default_joint_pos.append(
                float(self.get_parameter(f'default_joint_pos.{name}').value))

        self.declare_parameter('joint_states_rate', 50.0)
        rate_hz = float(self.get_parameter('joint_states_rate').value)

        self.declare_parameter('sg_window_length', 21)
        self.declare_parameter('sg_poly_order', 3)
        self.declare_parameter('sg_delta', 1.0 / 240.0)
        sg_wl = self.get_parameter('sg_window_length').value
        sg_po = self.get_parameter('sg_poly_order').value
        sg_delta = self.get_parameter('sg_delta').value

        self._sg_coeffs_smooth = savgol_coeffs(
            sg_wl, sg_po, deriv=0, delta=sg_delta, pos=sg_wl // 2, use='dot')
        self._sg_coeffs_vel = savgol_coeffs(
            sg_wl, sg_po, deriv=1, delta=sg_delta, pos=sg_wl // 2, use='dot')
        self._sg_window = sg_wl

        self.declare_parameter('pendulum_hinge_offset', [-0.05, 0.0, 0.06])
        self._hinge_offset = np.array(
            self.get_parameter('pendulum_hinge_offset').value, dtype=float)

        self._angle_buf = [deque(maxlen=sg_wl), deque(maxlen=sg_wl)]
        self._pendulum_pos = [0.0, 0.0]
        self._pendulum_vel = [0.0, 0.0]

        self._joint_states_pub = self.create_publisher(JointState, '/joint_states', SENSOR_QOS)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._base_tf_sub = self.create_subscription(
            PoseStamped, '/pose/base_link', self._base_tf_cb, SENSOR_QOS)

        self._base_sync_sub = message_filters.Subscriber(
            self, PoseStamped, '/pose/base_link', qos_profile=SENSOR_QOS)
        self._ee_sync_sub = message_filters.Subscriber(
            self, PoseStamped, '/pose/pendulum_ee', qos_profile=SENSOR_QOS)
        self._pose_sync = message_filters.ApproximateTimeSynchronizer(
            [self._base_sync_sub, self._ee_sync_sub], queue_size=10, slop=0.005)
        self._pose_sync.registerCallback(self._synced_pose_cb)

        self._publish_timer = self.create_timer(1.0 / rate_hz, self._publish_joint_states)

        self.get_logger().info(
            f'Go2 bridge passive node started (joint_states_rate={rate_hz} Hz)')

    def _base_tf_cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self._tf_broadcaster.sendTransform(t)

    def _synced_pose_cb(self, base_msg: PoseStamped, ee_msg: PoseStamped):
        o = base_msg.pose.orientation
        n = math.sqrt(o.w * o.w + o.x * o.x + o.y * o.y + o.z * o.z)
        if n < 1e-9:
            return
        R_base = _quat_to_rotation_matrix(o.w / n, o.x / n, o.y / n, o.z / n)

        p_base = np.array([base_msg.pose.position.x,
                           base_msg.pose.position.y,
                           base_msg.pose.position.z])
        p_ee = np.array([ee_msg.pose.position.x,
                         ee_msg.pose.position.y,
                         ee_msg.pose.position.z])

        v = R_base.T @ (p_ee - p_base) - self._hinge_offset

        joint1 = math.atan2(-v[1], v[2])
        joint2 = math.atan2(v[0], math.hypot(v[1], v[2]))

        if not self._angle_buf[0]:
            for _ in range(self._sg_window):
                self._angle_buf[0].append(joint1)
                self._angle_buf[1].append(joint2)
        else:
            self._angle_buf[0].append(joint1)
            self._angle_buf[1].append(joint2)

        if len(self._angle_buf[0]) >= self._sg_window:
            for j in range(2):
                buf = np.array(self._angle_buf[j])
                self._pendulum_pos[j] = float(
                    np.dot(self._sg_coeffs_smooth, buf))
                self._pendulum_vel[j] = float(
                    np.dot(self._sg_coeffs_vel, buf))

    def _publish_joint_states(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()

        for i, name in enumerate(JOINT_NAMES):
            js.name.append(name)
            js.position.append(self._default_joint_pos[i])
            js.velocity.append(0.0)
            js.effort.append(0.0)

        for i, name in enumerate(PENDULUM_JOINT_NAMES):
            js.name.append(name)
            js.position.append(self._pendulum_pos[i])
            js.velocity.append(self._pendulum_vel[i])
            js.effort.append(0.0)

        self._joint_states_pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = Go2BridgePassiveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
