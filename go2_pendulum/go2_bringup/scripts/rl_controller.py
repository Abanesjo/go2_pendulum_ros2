#!/usr/bin/env python3

from __future__ import annotations

import math
import os

import rclpy
import torch
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from go2_interfaces.srv import SetGoal
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_quat(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_conjugate(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    return (-x, -y, -z, w)


def quat_multiply(
    ax: float,
    ay: float,
    az: float,
    aw: float,
    bx: float,
    by: float,
    bz: float,
    bw: float,
) -> tuple[float, float, float, float]:
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_world_to_body(
    vx_w: float,
    vy_w: float,
    vz_w: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
    cqx, cqy, cqz, cqw = quat_conjugate(qx, qy, qz, qw)
    rx, ry, rz, _ = quat_multiply(
        *quat_multiply(cqx, cqy, cqz, cqw, vx_w, vy_w, vz_w, 0.0),
        qx,
        qy,
        qz,
        qw,
    )
    return (rx, ry, rz)


def stamp_to_sec(stamp_msg) -> float:
    return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9


def sec_to_stamp(sec_value: float) -> TimeMsg:
    sec_int = int(math.floor(sec_value))
    nsec = int(round((sec_value - sec_int) * 1e9))
    if nsec >= 1_000_000_000:
        sec_int += 1
        nsec -= 1_000_000_000
    if nsec < 0:
        sec_int -= 1
        nsec += 1_000_000_000
    return TimeMsg(sec=sec_int, nanosec=nsec)


def make_best_effort_qos(depth: int) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


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


class RLController(Node):
    POLICY_RATE_HZ = 50.0
    ACTION_SCALE = 0.25
    ACTION_LPF_CUTOFF_HZ = 8.0
    ACTION_BOUND_MARGIN = 1.0
    SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
    MAX_INPUT_AGE_SEC = 0.05
    MAX_INPUT_SKEW_SEC = 0.01
    SENSOR_QOS_DEPTH = 1
    COMMAND_QOS_DEPTH = 1
    MODE_STAND = "stand"
    MODE_POLICY = "policy"

    def __init__(self) -> None:
        super().__init__("rl_controller")

        default_model = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model",
            "policy.pt",
        )

        self.declare_parameter("model_file", default_model)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("joint_command_topic", "/joint_command")
        self.declare_parameter("goal_topic", "/goal")
        self.declare_parameter("goal_frame", "odom")
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
        self.declare_parameter("pendulum_joint1_name", "pendulum_joint1")
        self.declare_parameter("pendulum_joint2_name", "pendulum_joint2")
        self.declare_parameter(
            "full_command_joint_names",
            [
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "RL_thigh_joint",
                "RR_thigh_joint",
                "pendulum_joint1",
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
                "pendulum_joint2",
            ],
        )
        self.declare_parameter("pendulum_command_value_1", 0.0)
        self.declare_parameter("pendulum_command_value_2", 0.0)

        self.model_file = str(self.get_parameter("model_file").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.leg_joint_names = [str(x) for x in self.get_parameter("leg_joint_names").value]
        self.pendulum_joint1_name = str(self.get_parameter("pendulum_joint1_name").value)
        self.pendulum_joint2_name = str(self.get_parameter("pendulum_joint2_name").value)
        self.full_command_joint_names = [str(x) for x in self.get_parameter("full_command_joint_names").value]
        self.pendulum_command_value_1 = float(self.get_parameter("pendulum_command_value_1").value)
        self.pendulum_command_value_2 = float(self.get_parameter("pendulum_command_value_2").value)

        if len(self.leg_joint_names) != 12:
            raise ValueError(f"leg_joint_names must be length 12, got {len(self.leg_joint_names)}")
        if len(self.full_command_joint_names) != 14:
            raise ValueError(
                f"full_command_joint_names must be length 14, got {len(self.full_command_joint_names)}"
            )
        if self.pendulum_joint1_name not in self.full_command_joint_names:
            raise ValueError(f"{self.pendulum_joint1_name} missing from full_command_joint_names")
        if self.pendulum_joint2_name not in self.full_command_joint_names:
            raise ValueError(f"{self.pendulum_joint2_name} missing from full_command_joint_names")
        for name in self.leg_joint_names:
            if name not in self.full_command_joint_names:
                raise ValueError(f"{name} missing from full_command_joint_names")

        self.policy_dt = 1.0 / self.POLICY_RATE_HZ
        self.action_lpf_alpha = math.exp(-2.0 * math.pi * self.ACTION_LPF_CUTOFF_HZ * self.policy_dt)

        default_joint_pos_by_name = {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        self.default_joint_pos = [float(default_joint_pos_by_name[name]) for name in self.leg_joint_names]

        self.action_dim = 12
        self.obs_dim = 56
        self.joint_pos_low, self.joint_pos_high = self._build_joint_limits()
        self.action_low, self.action_high = self._build_action_bounds()

        self.previous_action_for_obs = [0.0] * self.action_dim
        self.filtered_action = [0.0] * self.action_dim
        self.command_action = [0.0] * self.action_dim

        self.gait_index = 0.0
        self.clock_inputs = [0.0, 0.0, 0.0, 0.0]

        self.base_x = 0.0
        self.base_y = 0.0
        self.base_yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw = 0.0

        self.latest_joint_state: JointState | None = None
        self.latest_odom: Odometry | None = None
        self.latest_joint_recv_wall_sec: float | None = None
        self.latest_odom_recv_wall_sec: float | None = None
        self.last_sync_sec: float | None = None
        self.policy_accumulator = 0.0
        # self.mode = self.MODE_STAND
        self.mode = self.MODE_POLICY
        self.last_commanded_pos = list(self.default_joint_pos)

        self.last_warn_sec = 0.0
        self.warned_missing_joint_velocity = False
        self.adopted_joint_state_order = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            self.get_logger().warn("CUDA unavailable; running policy on CPU")

        self.policy = torch.jit.load(self.model_file, map_location=self.device)
        self.policy.eval()

        command_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.COMMAND_QOS_DEPTH,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        sensor_qos = make_best_effort_qos(self.SENSOR_QOS_DEPTH)

        self.joint_command_pub = self.create_publisher(JointState, self.joint_command_topic, command_qos)
        self.create_subscription(PoseStamped, self.goal_topic, self.on_goal, sensor_qos)
        self.create_service(SetGoal, "/set_goal", self.on_set_goal)
        self.create_service(Trigger, "/toggle_policy_mode", self.on_toggle_policy_mode)
        self.create_subscription(JointState, self.joint_states_topic, self.on_joint_state, sensor_qos)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, sensor_qos)
        self.create_timer(self.policy_dt, self.on_stand_timer)

        self.get_logger().info(f"Using leg_joint_names order: {self.leg_joint_names}")
        self.get_logger().info(f"policy_device={self.device.type}")
        self.get_logger().info(
            "policy_rate_hz=%.3f (policy_dt=%.6f), lpf_alpha=%.6f"
            % (self.POLICY_RATE_HZ, self.policy_dt, self.action_lpf_alpha)
        )
        self.get_logger().info("Action pipeline: action clamp -> LPF -> action clamp -> desired hard clamp")
        self.get_logger().info("Action delay: disabled")
        self.get_logger().info(
            f"Publishing 14-joint name-based position command on /joint_command.position using order: {self.full_command_joint_names}"
        )
        self.get_logger().info("Policy stepping uses synchronized /joint_states and /odom header timestamps")
        self.get_logger().info("Standalone stand-mode publisher runs at 50 Hz (wall time)")
        self.get_logger().info("Using latest /joint_states and /odom directly (queue depth 1)")
        self.get_logger().info("Assuming /odom twist is already body-frame (child_frame_id='base')")
        self.get_logger().info("Default target set to world/odom (x=0.0, y=0.0, yaw=0.0)")
        self.get_logger().info("Previous action for observation is stored in-memory at each policy step")
        self.get_logger().info("Startup mode: stand (publishing default joint positions at 50 Hz)")
        self.get_logger().info("Service /toggle_policy_mode (std_srvs/Trigger): toggles stand <-> policy")

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def warn_throttled(self, msg: str, interval_sec: float = 1.0) -> None:
        now = self.now_sec()
        if now - self.last_warn_sec >= interval_sec:
            self.last_warn_sec = now
            self.get_logger().warn(msg)

    def on_joint_state(self, msg: JointState) -> None:
        if not self.adopted_joint_state_order:
            required = set(self.leg_joint_names + [self.pendulum_joint1_name, self.pendulum_joint2_name])
            msg_names = [str(n) for n in msg.name]
            if len(msg_names) == 14 and required.issubset(set(msg_names)):
                self.full_command_joint_names = msg_names
                self.adopted_joint_state_order = True
                self.get_logger().info(
                    f"Adopted /joint_states joint order for /joint_command (14 joints): {self.full_command_joint_names}"
                )
        self.latest_joint_state = msg
        self.latest_joint_recv_wall_sec = self.now_sec()
        self._try_policy_update_from_sensor_sync()

    def on_odom(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.latest_odom_recv_wall_sec = self.now_sec()
        self._try_policy_update_from_sensor_sync()

    def _set_goal(self, msg: PoseStamped) -> tuple[bool, str]:
        frame = msg.header.frame_id if msg.header.frame_id else self.goal_frame
        if frame != self.goal_frame:
            return False, f"Goal frame must be '{self.goal_frame}', got '{frame}'"

        self.target_x = float(msg.pose.position.x)
        self.target_y = float(msg.pose.position.y)
        q = msg.pose.orientation
        self.target_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        return True, "Goal updated"

    def on_goal(self, msg: PoseStamped) -> None:
        self._set_goal(msg)

    def on_set_goal(self, request: SetGoal.Request, response: SetGoal.Response):
        ok, msg = self._set_goal(request.goal)
        response.success = ok
        response.message = msg
        return response

    def _reset_policy_state(self) -> None:
        self.previous_action_for_obs = [0.0] * self.action_dim
        self.filtered_action = [0.0] * self.action_dim
        self.command_action = [0.0] * self.action_dim
        self.gait_index = 0.0
        self.clock_inputs = [0.0, 0.0, 0.0, 0.0]
        self.policy_accumulator = 0.0
        self.last_sync_sec = None

    def on_toggle_policy_mode(self, _request: Trigger.Request, response: Trigger.Response):
        if self.mode == self.MODE_STAND:
            self.mode = self.MODE_POLICY
            self._reset_policy_state()
            response.success = True
            response.message = "mode=policy"
            self.get_logger().info("Mode switched to policy")
            return response

        self.mode = self.MODE_STAND
        self._reset_policy_state()
        response.success = True
        response.message = "mode=stand"
        self.get_logger().info("Mode switched to stand")
        return response

    def _extract_joint_data(self, js: JointState):
        idx = {name: i for i, name in enumerate(js.name)}
        required = list(self.leg_joint_names) + [self.pendulum_joint1_name, self.pendulum_joint2_name]
        if not all(name in idx for name in required):
            return None
        if len(js.position) == 0:
            return None

        leg_indices = [idx[name] for name in self.leg_joint_names]
        p1_idx = idx[self.pendulum_joint1_name]
        p2_idx = idx[self.pendulum_joint2_name]

        leg_pos = [float(js.position[i]) for i in leg_indices]
        pendulum_pos = [float(js.position[p1_idx]), float(js.position[p2_idx])]

        if len(js.velocity) > max(p2_idx, max(leg_indices)):
            leg_vel = [float(js.velocity[i]) for i in leg_indices]
            pendulum_vel = [float(js.velocity[p1_idx]), float(js.velocity[p2_idx])]
        else:
            if not self.warned_missing_joint_velocity:
                self.warned_missing_joint_velocity = True
                self.get_logger().warn("joint_states.velocity missing/incomplete; using zeros")
            leg_vel = [0.0] * 12
            pendulum_vel = [0.0, 0.0]

        return leg_pos, leg_vel, pendulum_pos, pendulum_vel

    def _build_joint_limits(self) -> tuple[list[float], list[float]]:
        joint_pos_low = []
        joint_pos_high = []
        for name in self.leg_joint_names:
            hard_low, hard_high = DEFAULT_HARD_JOINT_LIMITS_BY_NAME[name]
            center = 0.5 * (hard_low + hard_high)
            half = 0.5 * (hard_high - hard_low) * self.SOFT_JOINT_POS_LIMIT_FACTOR
            joint_pos_low.append(center - half)
            joint_pos_high.append(center + half)
        return joint_pos_low, joint_pos_high

    def _build_action_bounds(self) -> tuple[list[float], list[float]]:
        action_low = []
        action_high = []
        for i in range(self.action_dim):
            low = (self.joint_pos_low[i] - self.default_joint_pos[i]) / self.ACTION_SCALE
            high = (self.joint_pos_high[i] - self.default_joint_pos[i]) / self.ACTION_SCALE
            center = 0.5 * (low + high)
            half = 0.5 * (high - low) * self.ACTION_BOUND_MARGIN
            action_low.append(center - half)
            action_high.append(center + half)
        return action_low, action_high

    @staticmethod
    def _clamp(values: list[float], lows: list[float], highs: list[float]) -> list[float]:
        return [min(highs[i], max(lows[i], values[i])) for i in range(len(values))]

    def _apply_action_pipeline(self, raw_action: list[float]) -> list[float]:
        bounded_raw = self._clamp(raw_action, self.action_low, self.action_high)
        filtered = [
            self.action_lpf_alpha * self.filtered_action[i] + (1.0 - self.action_lpf_alpha) * bounded_raw[i]
            for i in range(self.action_dim)
        ]
        filtered = self._clamp(filtered, self.action_low, self.action_high)

        self.filtered_action = list(filtered)
        return filtered

    def _compute_desired_joint_pos(self) -> list[float]:
        desired = [self.default_joint_pos[i] + self.ACTION_SCALE * self.command_action[i] for i in range(self.action_dim)]
        desired = self._clamp(desired, self.joint_pos_low, self.joint_pos_high)
        return desired

    def _update_clock_inputs(self) -> None:
        freq = 3.0
        phase = 0.5
        offset = 0.0
        bound = 0.0
        duration = 0.5

        self.gait_index = (self.gait_index + self.policy_dt * freq) % 1.0
        foot_indices = [
            self.gait_index + phase + offset + bound,
            self.gait_index + offset,
            self.gait_index + bound,
            self.gait_index + phase,
        ]

        remapped = []
        for x in foot_indices:
            r = x % 1.0
            if r < duration:
                remapped.append(r * (0.5 / duration))
            else:
                remapped.append(0.5 + (r - duration) * (0.5 / (1.0 - duration)))

        self.clock_inputs = [float(math.sin(2.0 * math.pi * x)) for x in remapped]

    def _build_observation(
        self,
        base_lin_vel_b: list[float],
        base_ang_vel_b: list[float],
        projected_gravity_b: list[float],
        leg_pos: list[float],
        leg_vel: list[float],
        pendulum_pos: list[float],
        pendulum_vel: list[float],
    ) -> list[float]:
        dx_w = self.target_x - self.base_x
        dy_w = self.target_y - self.base_y
        c = math.cos(self.base_yaw)
        s = math.sin(self.base_yaw)
        err_x_b = c * dx_w + s * dy_w
        err_y_b = -s * dx_w + c * dy_w
        yaw_err = wrap_to_pi(self.target_yaw - self.base_yaw)

        obs = []
        obs.extend(base_lin_vel_b)
        obs.extend(base_ang_vel_b)
        obs.extend(projected_gravity_b)
        obs.extend([err_x_b, err_y_b, yaw_err])
        obs.extend([leg_pos[i] - self.default_joint_pos[i] for i in range(12)])
        obs.extend(leg_vel)
        obs.extend(pendulum_pos)
        obs.extend(pendulum_vel)
        obs.extend(self.previous_action_for_obs)
        obs.extend(self.clock_inputs)
        return obs

    def _policy_step(
        self,
        base_lin_vel_b: list[float],
        base_ang_vel_b: list[float],
        projected_gravity_b: list[float],
        leg_pos: list[float],
        leg_vel: list[float],
        pendulum_pos: list[float],
        pendulum_vel: list[float],
    ) -> None:
        obs = self._build_observation(
            base_lin_vel_b,
            base_ang_vel_b,
            projected_gravity_b,
            leg_pos,
            leg_vel,
            pendulum_pos,
            pendulum_vel,
        )
        if len(obs) != self.obs_dim:
            self.warn_throttled(f"observation length mismatch: {len(obs)} != {self.obs_dim}")
            return

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)

        if isinstance(action_tensor, tuple):
            action_tensor = action_tensor[0]

        raw_action = action_tensor.squeeze(0).detach().cpu().tolist()
        raw_action = [float(x) for x in raw_action[: self.action_dim]]
        if len(raw_action) != self.action_dim:
            self.warn_throttled(f"action length mismatch: {len(raw_action)} != {self.action_dim}")
            return

        executed_action = self._apply_action_pipeline(raw_action)
        self.command_action = list(executed_action)
        self.previous_action_for_obs = list(executed_action)
        self._update_clock_inputs()

    def _publish_command(self, desired: list[float], stamp_msg: TimeMsg | None = None) -> None:
        desired_by_name = {self.leg_joint_names[i]: float(desired[i]) for i in range(self.action_dim)}
        desired_by_name[self.pendulum_joint1_name] = self.pendulum_command_value_1
        desired_by_name[self.pendulum_joint2_name] = self.pendulum_command_value_2

        cmd = JointState()
        if stamp_msg is None:
            stamp_msg = self.get_clock().now().to_msg()
        cmd.header.stamp = stamp_msg
        cmd.name = list(self.full_command_joint_names)
        cmd.position = [float(desired_by_name[name]) for name in self.full_command_joint_names]
        # Position-only command mode: keep velocity/effort empty so the articulation
        # controller does not try to apply multiple control modes at once.
        cmd.velocity = []
        cmd.effort = []
        self.joint_command_pub.publish(cmd)
        self.last_commanded_pos = list(cmd.position)

    def on_stand_timer(self) -> None:
        if self.mode != self.MODE_STAND:
            return
        self._publish_command(list(self.default_joint_pos))

    def _policy_control_step(self, step_sec: float) -> None:
        if self.mode != self.MODE_POLICY:
            return
        if self.latest_joint_state is None or self.latest_odom is None:
            self.warn_throttled("missing /joint_states or /odom; skipping policy update")
            return

        js_sec = stamp_to_sec(self.latest_joint_state.header.stamp)
        odom_sec = stamp_to_sec(self.latest_odom.header.stamp)

        if abs(js_sec - odom_sec) > self.MAX_INPUT_SKEW_SEC:
            self.warn_throttled(
                "joint_states/odom stamp skew too large; skipping policy/control update "
                f"(skew={abs(js_sec - odom_sec):.6f}s > max_input_skew_sec={self.MAX_INPUT_SKEW_SEC:.6f}s)"
            )
            return

        joint_data = self._extract_joint_data(self.latest_joint_state)
        if joint_data is None:
            self.warn_throttled("missing required joint names/positions; skipping")
            return

        leg_pos, leg_vel, pendulum_pos, pendulum_vel = joint_data

        odom = self.latest_odom
        q = odom.pose.pose.orientation
        self.base_x = float(odom.pose.pose.position.x)
        self.base_y = float(odom.pose.pose.position.y)
        self.base_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        base_lin_vel_b = [
            float(odom.twist.twist.linear.x),
            float(odom.twist.twist.linear.y),
            float(odom.twist.twist.linear.z),
        ]
        base_ang_vel_b = [
            float(odom.twist.twist.angular.x),
            float(odom.twist.twist.angular.y),
            float(odom.twist.twist.angular.z),
        ]
        gx, gy, gz = rotate_world_to_body(0.0, 0.0, -1.0, q.x, q.y, q.z, q.w)
        projected_gravity_b = [gx, gy, gz]

        self._policy_step(
            base_lin_vel_b,
            base_ang_vel_b,
            projected_gravity_b,
            leg_pos,
            leg_vel,
            pendulum_pos,
            pendulum_vel,
        )
        desired = self._compute_desired_joint_pos()
        self._publish_command(desired, sec_to_stamp(step_sec))

    def _try_policy_update_from_sensor_sync(self) -> None:
        if self.mode != self.MODE_POLICY:
            return
        if self.latest_joint_state is None or self.latest_odom is None:
            return
        if self.latest_joint_recv_wall_sec is None or self.latest_odom_recv_wall_sec is None:
            return

        now_wall = self.now_sec()
        if now_wall - self.latest_joint_recv_wall_sec > self.MAX_INPUT_AGE_SEC:
            self.warn_throttled(
                "joint_states receive-age too old; skipping policy update "
                f"(age={now_wall - self.latest_joint_recv_wall_sec:.6f}s > max_input_age_sec={self.MAX_INPUT_AGE_SEC:.6f}s)"
            )
            return
        if now_wall - self.latest_odom_recv_wall_sec > self.MAX_INPUT_AGE_SEC:
            self.warn_throttled(
                "odom receive-age too old; skipping policy update "
                f"(age={now_wall - self.latest_odom_recv_wall_sec:.6f}s > max_input_age_sec={self.MAX_INPUT_AGE_SEC:.6f}s)"
            )
            return

        js_sec = stamp_to_sec(self.latest_joint_state.header.stamp)
        odom_sec = stamp_to_sec(self.latest_odom.header.stamp)
        if abs(js_sec - odom_sec) > self.MAX_INPUT_SKEW_SEC:
            self.warn_throttled(
                "joint_states/odom stamp skew too large; skipping policy update "
                f"(skew={abs(js_sec - odom_sec):.6f}s > max_input_skew_sec={self.MAX_INPUT_SKEW_SEC:.6f}s)"
            )
            return

        sync_sec = min(js_sec, odom_sec)
        if self.last_sync_sec is None:
            self.last_sync_sec = sync_sec
            return

        dt = sync_sec - self.last_sync_sec
        if dt <= 0.0:
            return

        self.last_sync_sec = sync_sec
        self.policy_accumulator += dt
        while self.policy_accumulator >= self.policy_dt:
            step_sec = sync_sec - self.policy_accumulator + self.policy_dt
            self.policy_accumulator -= self.policy_dt
            self._policy_control_step(step_sec)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
