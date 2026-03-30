#!/usr/bin/env python3
"""Standalone MuJoCo simulation for Go2 + pendulum with a trained RL policy.

Dependencies: mujoco, torch, numpy
Usage: cd mujoco && python run_policy.py
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
XML_PATH = SCRIPT_DIR / "go2_pendulum.xml"
POLICY_PATH = SCRIPT_DIR / "policy.pt"

# --- Sim constants (matching Isaac) ---
SIM_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = SIM_DT * DECIMATION
ACTION_SCALE = 0.25
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
ACTION_BOUND_MARGIN = 1.0
ACTION_LPF_CUTOFF_HZ = 8.0
LPF_ALPHA = math.exp(-2.0 * math.pi * ACTION_LPF_CUTOFF_HZ * CONTROL_DT)
ACTION_DELAY_MIN = 0
ACTION_DELAY_MAX = 2
BASE_TILT_RESET_RAD = math.radians(45.0)
PENDULUM_INIT_MIN_RAD = math.radians(0.0)
PENDULUM_INIT_MAX_RAD = math.radians(9.9)

# Policy joint order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ... FL_calf, ...
POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

DEFAULT_JOINT_POS_BY_NAME = {
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}

HARD_JOINT_LIMITS = {
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

PENDULUM_JOINT_NAMES = ["pendulum_joint1", "pendulum_joint2"]

TARGET_X = 0.0
TARGET_Y = 0.0
TARGET_YAW = 0.0


def build_index_maps(model: mujoco.MjModel):
    """Build mapping from policy order to MuJoCo qpos/qvel/ctrl indices."""
    # joint name -> mujoco joint id
    mj_joint_id = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            mj_joint_id[name] = i

    # actuator name -> mujoco actuator id, and the joint it drives
    mj_act_joint = {}
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jnt_id = model.actuator_trnid[i, 0]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if jnt_name:
            mj_act_joint[jnt_name] = i

    # Policy-order indices into qpos, qvel, ctrl
    leg_qpos_idx = []
    leg_qvel_idx = []
    leg_ctrl_idx = []
    for name in POLICY_JOINT_NAMES:
        jid = mj_joint_id[name]
        leg_qpos_idx.append(model.jnt_qposadr[jid])
        leg_qvel_idx.append(model.jnt_dofadr[jid])
        leg_ctrl_idx.append(mj_act_joint[name])

    pend_qpos_idx = []
    pend_qvel_idx = []
    for name in PENDULUM_JOINT_NAMES:
        jid = mj_joint_id[name]
        pend_qpos_idx.append(model.jnt_qposadr[jid])
        pend_qvel_idx.append(model.jnt_dofadr[jid])

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    return (
        np.array(leg_qpos_idx),
        np.array(leg_qvel_idx),
        np.array(leg_ctrl_idx),
        np.array(pend_qpos_idx),
        np.array(pend_qvel_idx),
        base_body_id,
    )


def compute_soft_limits_and_bounds():
    """Compute soft joint limits and per-joint action bounds (matching Isaac)."""
    default_pos = np.array([DEFAULT_JOINT_POS_BY_NAME[n] for n in POLICY_JOINT_NAMES])
    joint_pos_low = np.zeros(12)
    joint_pos_high = np.zeros(12)
    for i, name in enumerate(POLICY_JOINT_NAMES):
        lo, hi = HARD_JOINT_LIMITS[name]
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * SOFT_JOINT_POS_LIMIT_FACTOR
        joint_pos_low[i] = center - half
        joint_pos_high[i] = center + half

    action_low = (joint_pos_low - default_pos) / ACTION_SCALE
    action_high = (joint_pos_high - default_pos) / ACTION_SCALE
    action_center = 0.5 * (action_low + action_high)
    action_half = 0.5 * (action_high - action_low) * ACTION_BOUND_MARGIN
    action_low = action_center - action_half
    action_high = action_center + action_half

    return default_pos, joint_pos_low, joint_pos_high, action_low, action_high


def update_clock_inputs(gait_index: float, dt: float):
    """Compute 4D clock signal (matching Isaac gait generator)."""
    freq = 3.0
    phase = 0.5
    offset = 0.0
    bound = 0.0
    duration = 0.5

    gait_index = (gait_index + dt * freq) % 1.0
    foot_indices = [
        gait_index + phase + offset + bound,
        gait_index + offset,
        gait_index + bound,
        gait_index + phase,
    ]
    clock = np.zeros(4)
    for k, x in enumerate(foot_indices):
        r = x % 1.0
        if r < duration:
            mapped = r * (0.5 / duration)
        else:
            mapped = 0.5 + (r - duration) * (0.5 / (1.0 - duration))
        clock[k] = math.sin(2.0 * math.pi * mapped)

    return gait_index, clock


def compute_observation(
    data: mujoco.MjData,
    base_body_id: int,
    leg_qpos_idx: np.ndarray,
    leg_qvel_idx: np.ndarray,
    pend_qpos_idx: np.ndarray,
    pend_qvel_idx: np.ndarray,
    default_pos: np.ndarray,
    prev_action: np.ndarray,
    clock: np.ndarray,
) -> np.ndarray:
    """Build the 56D observation vector."""
    # Body rotation matrix (3x3, row-major in xmat)
    R = data.xmat[base_body_id].reshape(3, 3)

    # World-frame velocities from qvel[0:6] (freejoint: 3 lin + 3 ang, both world-frame)
    lin_vel_w = data.qvel[0:3]
    ang_vel_w = data.qvel[3:6]

    # Rotate to body frame
    lin_vel_b = R.T @ lin_vel_w
    ang_vel_b = R.T @ ang_vel_w

    # Projected gravity in body frame: R^T @ [0, 0, -1]
    gravity_b = R.T @ np.array([0.0, 0.0, -1.0])

    # Position/yaw error
    base_pos = data.qpos[0:3]
    base_quat = data.qpos[3:7]  # w, x, y, z
    yaw = quat_to_yaw(base_quat)

    dx_w = TARGET_X - base_pos[0]
    dy_w = TARGET_Y - base_pos[1]
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    err_x_b = cos_y * dx_w + sin_y * dy_w
    err_y_b = -sin_y * dx_w + cos_y * dy_w
    yaw_err = wrap_to_pi(TARGET_YAW - yaw)

    # Joint states in policy order
    leg_pos_rel = data.qpos[leg_qpos_idx] - default_pos
    leg_vel = data.qvel[leg_qvel_idx]
    pend_pos = data.qpos[pend_qpos_idx]
    pend_vel = data.qvel[pend_qvel_idx]

    obs = np.concatenate([
        lin_vel_b,          # 3
        ang_vel_b,          # 3
        gravity_b,          # 3
        [err_x_b, err_y_b, yaw_err],  # 3
        leg_pos_rel,        # 12
        leg_vel,            # 12
        pend_pos,           # 2
        pend_vel,           # 2
        prev_action,        # 12
        clock,              # 4
    ])
    assert obs.shape[0] == 56, f"Expected 56D obs, got {obs.shape[0]}"
    return obs


def quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from MuJoCo quaternion (w, x, y, z)."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_base_tilt(data: mujoco.MjData, base_body_id: int) -> float:
    """Angle between body z-axis and world z-axis."""
    R = data.xmat[base_body_id].reshape(3, 3)
    gravity_b = R.T @ np.array([0.0, 0.0, -1.0])
    return math.atan2(np.linalg.norm(gravity_b[:2]), -gravity_b[2])


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    default_pos: np.ndarray,
    leg_qpos_idx: np.ndarray,
    pend_qpos_idx: np.ndarray,
):
    """Reset to default standing pose."""
    mujoco.mj_resetData(model, data)

    # Base position (freejoint: qpos[0:3]=xyz, qpos[3:7]=quat wxyz)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.4
    data.qpos[3] = 1.0  # w
    data.qpos[4] = 0.0  # x
    data.qpos[5] = 0.0  # y
    data.qpos[6] = 0.0  # z

    # Set leg joints to default
    data.qpos[leg_qpos_idx] = default_pos

    # Small random pendulum perturbation
    for idx in pend_qpos_idx:
        sign = random.choice([-1, 1])
        mag = random.uniform(PENDULUM_INIT_MIN_RAD, PENDULUM_INIT_MAX_RAD)
        data.qpos[idx] = sign * mag

    mujoco.mj_forward(model, data)


def main():
    if not XML_PATH.is_file():
        raise FileNotFoundError(f"Missing MJCF: {XML_PATH}")
    if not POLICY_PATH.is_file():
        raise FileNotFoundError(f"Missing policy: {POLICY_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    (leg_qpos_idx, leg_qvel_idx, leg_ctrl_idx,
     pend_qpos_idx, pend_qvel_idx, base_body_id) = build_index_maps(model)

    default_pos, joint_pos_low, joint_pos_high, action_low, action_high = (
        compute_soft_limits_and_bounds()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.jit.load(str(POLICY_PATH), map_location=device)
    policy.eval()

    # Action pipeline state
    actions_filtered = np.zeros(12)
    prev_action_for_obs = np.zeros(12)
    gait_index = 0.0
    clock = np.zeros(4)

    # Action delay buffer
    max_delay = ACTION_DELAY_MAX
    delay_buf = np.zeros((12, max_delay + 1))
    delay_steps = random.randint(ACTION_DELAY_MIN, ACTION_DELAY_MAX)

    # Initial reset
    reset_state(model, data, default_pos, leg_qpos_idx, pend_qpos_idx)

    print(f"[INFO] Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"[INFO] Policy device: {device}")
    print(f"[INFO] LPF alpha: {LPF_ALPHA:.4f}, delay_steps: {delay_steps}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 135.0

        while viewer.is_running():
            step_start = time.time()

            # Check for reset
            if compute_base_tilt(data, base_body_id) > BASE_TILT_RESET_RAD:
                reset_state(model, data, default_pos, leg_qpos_idx, pend_qpos_idx)
                actions_filtered[:] = 0.0
                prev_action_for_obs[:] = 0.0
                gait_index = 0.0
                clock[:] = 0.0
                delay_buf[:] = 0.0
                delay_steps = random.randint(ACTION_DELAY_MIN, ACTION_DELAY_MAX)
                continue

            # Observe
            obs = compute_observation(
                data, base_body_id,
                leg_qpos_idx, leg_qvel_idx,
                pend_qpos_idx, pend_qvel_idx,
                default_pos, prev_action_for_obs, clock,
            )

            # Infer
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor = policy(obs_tensor)
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]
            raw_action = action_tensor.squeeze(0).cpu().numpy()[:12].astype(np.float64)

            # --- Action pipeline ---
            # 1. Clamp to action bounds
            bounded = np.clip(raw_action, action_low, action_high)

            # 2. Action delay
            delay_buf = np.roll(delay_buf, shift=1, axis=1)
            delay_buf[:, 0] = bounded
            d = min(delay_steps, max_delay)
            delayed = delay_buf[:, d]

            # 3. Low-pass filter
            actions_filtered = LPF_ALPHA * actions_filtered + (1.0 - LPF_ALPHA) * delayed

            # 4. Re-clamp after LPF
            actions_filtered = np.clip(actions_filtered, action_low, action_high)

            # 5. Desired joint positions
            desired_pos = default_pos + ACTION_SCALE * actions_filtered

            # 6. Hard-clamp to soft limits
            desired_pos = np.clip(desired_pos, joint_pos_low, joint_pos_high)

            # Set ctrl (actuator inputs = desired joint positions for PD actuators)
            data.ctrl[leg_ctrl_idx] = desired_pos

            # Step physics (decimation)
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            # Update state for next iteration
            prev_action_for_obs = actions_filtered.copy()
            gait_index, clock = update_clock_inputs(gait_index, CONTROL_DT)

            # Sync viewer
            viewer.sync()

            # Real-time sync
            elapsed = time.time() - step_start
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
