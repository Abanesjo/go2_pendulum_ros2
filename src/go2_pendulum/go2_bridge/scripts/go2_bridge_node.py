#!/usr/bin/env python3
"""
Go2 Bridge Node: bridges Unitree Go2 lowstate/lowcmd to standard ROS2 interfaces.

Subscriptions:
  /lowstate       (unitree_go/msg/LowState) -> publishes /joint_states, /imu
  /joint_commands (sensor_msgs/JointState)   -> publishes /lowcmd

"""

import math
from collections import deque

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from scipy.signal import savgol_coeffs
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from unitree_go.msg import LowState, LowCmd

from go2_bridge.crc import compute_crc


def _quat_to_rotation_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y*y + z*z),     2 * (x*y - z*w),     2 * (x*z + y*w)],
        [    2 * (x*y + z*w), 1 - 2 * (x*x + z*z),     2 * (y*z - x*w)],
        [    2 * (x*z - y*w),     2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])

GO2_NUM_MOTOR = 20
GO2_NUM_LEG_MOTOR = 12

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Best-effort, volatile, depth 1 — matches Unitree SDK DDS defaults
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# URDF joint name -> motor index (from motor_crc.h)
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

# Ordered joint names (sorted by motor index)
JOINT_NAMES = sorted(JOINT_MAP.keys(), key=lambda n: JOINT_MAP[n])

PENDULUM_JOINT_NAMES = ['pendulum_joint1', 'pendulum_joint2']


class Go2BridgeNode(Node):
    def __init__(self):
        super().__init__('go2_bridge_node')

        # CRC toggle
        self.declare_parameter('enable_crc', True)
        self._enable_crc = self.get_parameter('enable_crc').value

        # Load per-joint policy gains from parameters
        self._policy_gains = {}
        for name in JOINT_NAMES:
            self.declare_parameter(f'gains.{name}.kp', 25.0)
            self.declare_parameter(f'gains.{name}.kd', 0.6)
            kp = self.get_parameter(f'gains.{name}.kp').value
            kd = self.get_parameter(f'gains.{name}.kd').value
            self._policy_gains[name] = (kp, kd)

        # Standup gains (higher stiffness for standing up reliably)
        self.declare_parameter('standup_kp', 60.0)
        self.declare_parameter('standup_kd', 5.0)
        standup_kp = self.get_parameter('standup_kp').value
        standup_kd = self.get_parameter('standup_kd').value
        self._standup_gains = {name: (standup_kp, standup_kd) for name in JOINT_NAMES}

        # Savitzky-Golay filter for pendulum angle/velocity (centered, 240 Hz)
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

        # Start in standup mode; switch when /policy_active becomes True
        self._policy_active = False

        # State tracking
        self._has_state = False
        self._latest_lowcmd = None

        # Publishers
        self._joint_states_pub = self.create_publisher(JointState, '/joint_states', SENSOR_QOS)
        self._imu_pub = self.create_publisher(Imu, '/imu', SENSOR_QOS)
        self._lowcmd_pub = self.create_publisher(LowCmd, '/lowcmd', SENSOR_QOS)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self._lowstate_sub = self.create_subscription(
            LowState, '/lowstate', self._lowstate_cb, SENSOR_QOS)
        self._joint_cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self._joint_cmd_cb, SENSOR_QOS)
        self._policy_active_sub = self.create_subscription(
            Bool, '/policy_active', self._policy_active_cb, SENSOR_QOS)

        self._base_tf_sub = self.create_subscription(
            PoseStamped, '/pose/base_link', self._base_tf_cb, SENSOR_QOS)

        self._base_sync_sub = message_filters.Subscriber(
            self, PoseStamped, '/pose/base_link', qos_profile=SENSOR_QOS)
        self._ee_sync_sub = message_filters.Subscriber(
            self, PoseStamped, '/pose/pendulum_ee', qos_profile=SENSOR_QOS)
        self._pose_sync = message_filters.ApproximateTimeSynchronizer(
            [self._base_sync_sub, self._ee_sync_sub], queue_size=10, slop=0.005)
        self._pose_sync.registerCallback(self._synced_pose_cb)

        # 500 Hz republish timer for stable PD control
        self._republish_timer = self.create_timer(0.002, self._republish_lowcmd)

        self.get_logger().info(
            f'Go2 bridge node started (enable_crc={self._enable_crc})')

    def _policy_active_cb(self, msg: Bool):
        if msg.data != self._policy_active:
            self._policy_active = msg.data
            mode = 'policy' if self._policy_active else 'standup'
            gains = self._policy_gains if self._policy_active else self._standup_gains
            kp, kd = next(iter(gains.values()))
            self.get_logger().info(
                f'Switched to {mode} gains (kp={kp}, kd={kd})')

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

    def _lowstate_cb(self, msg: LowState):
        self._has_state = True

        stamp = self.get_clock().now().to_msg()

        # Build and publish JointState (12 leg joints + 2 pendulum joints)
        js = JointState()
        js.header.stamp = stamp

        for name in JOINT_NAMES:
            idx = JOINT_MAP[name]
            motor = msg.motor_state[idx]
            js.name.append(name)
            js.position.append(float(motor.q))
            js.velocity.append(float(motor.dq))
            js.effort.append(float(motor.tau_est))

        for i, name in enumerate(PENDULUM_JOINT_NAMES):
            js.name.append(name)
            js.position.append(self._pendulum_pos[i])
            js.velocity.append(self._pendulum_vel[i])
            js.effort.append(0.0)

        self._joint_states_pub.publish(js)

        # Build and publish Imu
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'base_link'
        # Unitree quaternion convention: [w, x, y, z]
        imu.orientation.w = float(msg.imu_state.quaternion[0])
        imu.orientation.x = float(msg.imu_state.quaternion[1])
        imu.orientation.y = float(msg.imu_state.quaternion[2])
        imu.orientation.z = float(msg.imu_state.quaternion[3])
        imu.angular_velocity.x = float(msg.imu_state.gyroscope[0])
        imu.angular_velocity.y = float(msg.imu_state.gyroscope[1])
        imu.angular_velocity.z = float(msg.imu_state.gyroscope[2])
        imu.linear_acceleration.x = float(msg.imu_state.accelerometer[0])
        imu.linear_acceleration.y = float(msg.imu_state.accelerometer[1])
        imu.linear_acceleration.z = float(msg.imu_state.accelerometer[2])

        self._imu_pub.publish(imu)

    def _joint_cmd_cb(self, msg: JointState):
        if not self._has_state:
            self.get_logger().warn(
                'Received /joint_commands before any /lowstate — ignoring')
            return

        cmd = LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        # Initialize all 20 motors to safe defaults
        for i in range(GO2_NUM_MOTOR):
            cmd.motor_cmd[i].mode = 0x01   # servo (PMSM) mode
            cmd.motor_cmd[i].q = POS_STOP_F
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = VEL_STOP_F
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        # Apply commanded values from JointState message
        has_pos = len(msg.position) > 0
        has_vel = len(msg.velocity) > 0
        has_eff = len(msg.effort) > 0

        gains = self._policy_gains if self._policy_active else self._standup_gains

        for i, name in enumerate(msg.name):
            if name not in JOINT_MAP:
                continue
            idx = JOINT_MAP[name]
            kp, kd = gains.get(name, (25.0, 0.6))

            cmd.motor_cmd[idx].kp = float(kp)
            cmd.motor_cmd[idx].kd = float(kd)

            if has_pos and i < len(msg.position):
                cmd.motor_cmd[idx].q = float(msg.position[i])
            if has_vel and i < len(msg.velocity):
                cmd.motor_cmd[idx].dq = float(msg.velocity[i])
            else:
                cmd.motor_cmd[idx].dq = 0.0
            if has_eff and i < len(msg.effort):
                cmd.motor_cmd[idx].tau = float(msg.effort[i])
            else:
                cmd.motor_cmd[idx].tau = 0.0

        if self._enable_crc:
            compute_crc(cmd)

        self._latest_lowcmd = cmd
        self._lowcmd_pub.publish(cmd)

    def _republish_lowcmd(self):
        if self._latest_lowcmd is not None:
            self._lowcmd_pub.publish(self._latest_lowcmd)


def main(args=None):
    rclpy.init(args=args)
    node = Go2BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
