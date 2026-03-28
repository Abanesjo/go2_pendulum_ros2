#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class OdomToTfNode(Node):
    def __init__(self) -> None:
        super().__init__("odom_to_tf")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("qos_depth", 1)
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        qos_depth = max(1, int(self.get_parameter("qos_depth").value))
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, qos)

    def odom_callback(self, msg: Odometry) -> None:
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = msg.child_frame_id
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomToTfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
