#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def quat_from_yaw(yaw):
    class Q:
        pass
    q = Q()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def wrap_to_pi(a):
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


class WaypointManagerNode(Node):
    def __init__(self):
        super().__init__('waypoint_manager_node')

        self.declare_parameter('waypoint_spacing', 0.05)
        self.declare_parameter('waypoint_reach_threshold', 0.1)
        self.declare_parameter('rate', 10.0)

        self._spacing = self.get_parameter('waypoint_spacing').value
        self._threshold = self.get_parameter('waypoint_reach_threshold').value
        rate = self.get_parameter('rate').value

        self._base_x = 0.0
        self._base_y = 0.0
        self._base_yaw = 0.0
        self._has_base = False

        self._waypoints = []
        self._wp_index = 0
        self._policy_active = False

        self._goal_pub = self.create_publisher(PoseStamped, '/goal', SENSOR_QOS)
        self._path_pub = self.create_publisher(Path, '/plan', SENSOR_QOS)

        self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_pose_cb, SENSOR_QOS)
        self.create_subscription(
            PoseStamped, '/pose/base_link', self._base_pose_cb, SENSOR_QOS)
        self.create_subscription(
            Bool, '/policy_active', self._policy_active_cb, SENSOR_QOS)

        self._timer = self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'Waypoint manager ready (spacing={self._spacing}m, '
            f'threshold={self._threshold}m, rate={rate}Hz)')

    def _base_pose_cb(self, msg: PoseStamped):
        self._base_x = msg.pose.position.x
        self._base_y = msg.pose.position.y
        self._base_yaw = yaw_from_quat(msg.pose.orientation)
        self._has_base = True

    def _goal_pose_cb(self, msg: PoseStamped):
        if not self._has_base:
            self.get_logger().warn('No base pose yet, ignoring goal')
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        goal_yaw = yaw_from_quat(msg.pose.orientation)

        start_x = self._base_x
        start_y = self._base_y
        start_yaw = self._base_yaw

        dx = goal_x - start_x
        dy = goal_y - start_y
        dist = math.sqrt(dx * dx + dy * dy)
        delta_yaw = wrap_to_pi(goal_yaw - start_yaw)

        if dist < self._spacing:
            self._waypoints = [(goal_x, goal_y, goal_yaw)]
        else:
            n = max(1, int(math.ceil(dist / self._spacing)))
            self._waypoints = []
            for i in range(1, n + 1):
                t = i / n
                wx = start_x + t * dx
                wy = start_y + t * dy
                wyaw = start_yaw + t * delta_yaw
                self._waypoints.append((wx, wy, wyaw))

        self._wp_index = 0

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'world'
        for wx, wy, wyaw in self._waypoints:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            q = quat_from_yaw(wyaw)
            ps.pose.orientation.x = q.x
            ps.pose.orientation.y = q.y
            ps.pose.orientation.z = q.z
            ps.pose.orientation.w = q.w
            path.poses.append(ps)
        self._path_pub.publish(path)

        self.get_logger().info(
            f'New path: {len(self._waypoints)} waypoints over {dist:.2f}m')

    def _policy_active_cb(self, msg: Bool):
        was_active = self._policy_active
        self._policy_active = msg.data
        if was_active and not self._policy_active and self._waypoints:
            self._waypoints = []
            self._wp_index = 0
            self._publish_empty_path()
            self.get_logger().info(
                'Policy stopped; cleared active waypoint plan')

    def _publish_empty_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'world'
        self._path_pub.publish(path)

    def _tick(self):
        if not self._waypoints:
            return

        wx, wy, wyaw = self._waypoints[self._wp_index]

        dx = wx - self._base_x
        dy = wy - self._base_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self._threshold and self._wp_index < len(self._waypoints) - 1:
            self._wp_index += 1
            wx, wy, wyaw = self._waypoints[self._wp_index]
            self.get_logger().info(
                f'Waypoint {self._wp_index + 1}/{len(self._waypoints)}')

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = wx
        msg.pose.position.y = wy
        q = quat_from_yaw(wyaw)
        msg.pose.orientation.x = q.x
        msg.pose.orientation.y = q.y
        msg.pose.orientation.z = q.z
        msg.pose.orientation.w = q.w
        self._goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
