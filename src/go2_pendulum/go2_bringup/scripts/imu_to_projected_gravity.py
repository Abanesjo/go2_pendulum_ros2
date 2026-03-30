#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from sensor_msgs.msg import Imu


def normalize_vector(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return (0.0, 0.0, -1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def normalize_quat(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector(v, q):
    q = normalize_quat(q)
    vq = (v[0], v[1], v[2], 0.0)
    return quat_multiply(quat_multiply(q, vq), quat_conjugate(q))[:3]


class ImuToProjectedGravityNode(Node):
    def __init__(self):
        super().__init__("imu_to_projected_gravity")

        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("gravity_topic", "/projected_gravity")

        self.imu_topic = self.get_parameter("imu_topic").value
        self.gravity_topic = self.get_parameter("gravity_topic").value

        self.gravity_world = normalize_vector((0.0, 0.0, -1.0))

        self.gravity_pub = self.create_publisher(Vector3, self.gravity_topic, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)

    def imu_callback(self, msg: Imu):
        q_imu_to_world = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        g_imu = rotate_vector(self.gravity_world, quat_conjugate(q_imu_to_world))
        g_imu = normalize_vector(g_imu)

        out = Vector3()
        out.x = g_imu[0]
        out.y = g_imu[1]
        out.z = g_imu[2]
        self.gravity_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImuToProjectedGravityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
