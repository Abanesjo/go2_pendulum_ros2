#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg


ROOT_DIR = Path(__file__).resolve().parent

USD_PATH = ROOT_DIR / "usd" / "go2_pendulum.usd"
POLICY_PATH = ROOT_DIR / "model" / "policy.pt"

SIM_DT = 1.0 / 200.0
DECIMATION = 4
ACTION_SCALE = 0.25

ENABLE_ACTION_LPF = True
ACTION_LPF_CUTOFF_HZ = 8.0
ENABLE_ACTION_DELAY = True
ACTION_DELAY_STEPS_MIN = 0
ACTION_DELAY_STEPS_MAX = 2
ACTION_DELAY_RANDOMIZE_PER_RESET = True
ENABLE_PER_JOINT_ACTION_BOUNDS = True
ACTION_BOUND_MARGIN = 1.0
ENABLE_DESIRED_JOINT_POS_HARD_CLAMP = True

PENDULUM_JOINT_NAMES = ["pendulum_joint1", "pendulum_joint2"]
PENDULUM_ANGLE_MIN_RAD = math.radians(0.0)
PENDULUM_ANGLE_MAX_RAD = math.radians(9.9)
PENDULUM_JOINT_LIMIT_MIN_RAD = math.radians(-90.0)
PENDULUM_JOINT_LIMIT_MAX_RAD = math.radians(90.0)

TARGET_X = 0.0
TARGET_Y = 0.0
TARGET_YAW = 0.0
BASE_TILT_RESET_RAD = math.radians(45.0)
RUN_REAL_TIME = True


def make_physics_material_cfg() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )


def build_robot_cfg() -> ArticulationCfg:
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.6,
                friction=0.0,
            ),
        },
    )
    cfg.articulation_root_prim_path = "/base"
    return cfg


def find_leg_joint_ids_from_robot_order(robot: Articulation) -> torch.Tensor:
    ids: list[int] = []
    for idx, name in enumerate(robot.joint_names):
        if name.endswith("_hip_joint") or name.endswith("_thigh_joint") or name.endswith("_calf_joint"):
            ids.append(idx)
    if len(ids) != 12:
        raise RuntimeError(f"Expected 12 leg joints, got {len(ids)}: {ids}")
    return torch.tensor(ids, dtype=torch.long, device=robot.device)


def find_named_joint_ids(robot: Articulation, names: list[str]) -> torch.Tensor:
    ids: list[int] = []
    for name in names:
        found, _ = robot.find_joints(name)
        if len(found) != 1:
            raise RuntimeError(f"Expected exactly one joint named '{name}', found {found}.")
        ids.append(found[0])
    return torch.tensor(ids, dtype=torch.long, device=robot.device)


def print_joint_orders(
    robot: Articulation,
    leg_joint_ids: torch.Tensor,
    pendulum_joint_ids: torch.Tensor,
) -> None:
    print("[INFO] Full robot joint order from USD:")
    for joint_idx, name in enumerate(robot.joint_names):
        print(f"  joint_idx={joint_idx:02d} name={name}")

    print("[INFO] Leg joint order used for policy action/observation (12D):")
    for policy_idx, joint_idx in enumerate(leg_joint_ids.tolist()):
        print(f"  policy_leg_idx={policy_idx:02d} joint_idx={joint_idx:02d} name={robot.joint_names[joint_idx]}")

    print("[INFO] Pendulum joint order used in observation:")
    for obs_idx, joint_idx in enumerate(pendulum_joint_ids.tolist()):
        print(f"  pendulum_obs_idx={obs_idx:02d} joint_idx={joint_idx:02d} name={robot.joint_names[joint_idx]}")


def apply_pendulum_joint_limits(robot: Articulation, pendulum_joint_ids: torch.Tensor) -> None:
    if pendulum_joint_ids.numel() == 0:
        return
    limits = torch.zeros(
        (robot.num_instances, pendulum_joint_ids.numel(), 2),
        device=robot.device,
        dtype=torch.float32,
    )
    limits[:, :, 0] = PENDULUM_JOINT_LIMIT_MIN_RAD
    limits[:, :, 1] = PENDULUM_JOINT_LIMIT_MAX_RAD
    robot.write_joint_position_limit_to_sim(
        limits,
        joint_ids=pendulum_joint_ids,
        warn_limit_violation=False,
    )


def sample_initial_joint_state(
    robot: Articulation,
    pendulum_joint_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    if pendulum_joint_ids.numel() > 0:
        for joint_idx in pendulum_joint_ids.tolist():
            signs = (torch.randint(0, 2, (robot.num_instances,), device=robot.device) * 2 - 1).to(dtype=torch.float32)
            magnitudes = (
                torch.rand((robot.num_instances,), device=robot.device)
                * (PENDULUM_ANGLE_MAX_RAD - PENDULUM_ANGLE_MIN_RAD)
                + PENDULUM_ANGLE_MIN_RAD
            )
            joint_pos[:, joint_idx] += signs * magnitudes

    return joint_pos, joint_vel


def reset_robot_state(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    pendulum_joint_ids: torch.Tensor,
    env_origins: torch.Tensor,
) -> None:
    robot.reset()
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += env_origins
    joint_pos, joint_vel = sample_initial_joint_state(robot, pendulum_joint_ids)
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()
    robot.update(SIM_DT)


def compute_action_and_joint_bounds(
    robot: Articulation,
    leg_joint_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    leg_joint_pos_limits = robot.data.soft_joint_pos_limits[:, leg_joint_ids, :]
    leg_default_joint_pos = robot.data.default_joint_pos[:, leg_joint_ids]
    joint_pos_low = leg_joint_pos_limits[:, :, 0].clone()
    joint_pos_high = leg_joint_pos_limits[:, :, 1].clone()

    if ENABLE_PER_JOINT_ACTION_BOUNDS:
        action_low = (joint_pos_low - leg_default_joint_pos) / ACTION_SCALE
        action_high = (joint_pos_high - leg_default_joint_pos) / ACTION_SCALE
        action_center = 0.5 * (action_low + action_high)
        action_half_range = 0.5 * (action_high - action_low) * ACTION_BOUND_MARGIN
        action_low = action_center - action_half_range
        action_high = action_center + action_half_range
    else:
        action_low = torch.full_like(leg_default_joint_pos, -float("inf"))
        action_high = torch.full_like(leg_default_joint_pos, float("inf"))

    return action_low, action_high, joint_pos_low, joint_pos_high


def update_clock_inputs(
    gait_index: torch.Tensor,
    clock_inputs: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 3.0
    phases = 0.5
    offsets = 0.0
    bounds = 0.0
    durations = 0.5 * torch.ones((clock_inputs.shape[0],), dtype=torch.float32, device=clock_inputs.device)

    gait_index = torch.remainder(gait_index + step_dt * frequencies, 1.0)

    foot_indices = [
        gait_index + phases + offsets + bounds,
        gait_index + offsets,
        gait_index + bounds,
        gait_index + phases,
    ]

    for idxs in foot_indices:
        stance_idxs = torch.remainder(idxs, 1.0) < durations
        swing_idxs = torch.remainder(idxs, 1.0) > durations
        idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1.0) * (0.5 / durations[stance_idxs])
        idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1.0) - durations[swing_idxs]) * (
            0.5 / (1.0 - durations[swing_idxs])
        )

    clock_inputs[:, 0] = torch.sin(2.0 * torch.pi * foot_indices[0])
    clock_inputs[:, 1] = torch.sin(2.0 * torch.pi * foot_indices[1])
    clock_inputs[:, 2] = torch.sin(2.0 * torch.pi * foot_indices[2])
    clock_inputs[:, 3] = torch.sin(2.0 * torch.pi * foot_indices[3])

    return gait_index, clock_inputs


def compute_observation(
    robot: Articulation,
    leg_joint_ids: torch.Tensor,
    pendulum_joint_ids: torch.Tensor,
    prev_action_for_obs: torch.Tensor,
    clock_inputs: torch.Tensor,
    target_xy: torch.Tensor,
    target_yaw: torch.Tensor,
) -> torch.Tensor:
    leg_joint_pos_rel = robot.data.joint_pos[:, leg_joint_ids] - robot.data.default_joint_pos[:, leg_joint_ids]
    leg_joint_vel = robot.data.joint_vel[:, leg_joint_ids]
    pendulum_joint_pos = robot.data.joint_pos[:, pendulum_joint_ids]
    pendulum_joint_vel = robot.data.joint_vel[:, pendulum_joint_ids]

    _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    base_pos_xy = robot.data.root_pos_w[:, :2]
    position_error_xy_world = target_xy - base_pos_xy
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    position_error_xy_body = torch.stack(
        (
            cos_yaw * position_error_xy_world[:, 0] + sin_yaw * position_error_xy_world[:, 1],
            -sin_yaw * position_error_xy_world[:, 0] + cos_yaw * position_error_xy_world[:, 1],
        ),
        dim=-1,
    )
    yaw_error = math_utils.wrap_to_pi(target_yaw - yaw).unsqueeze(-1)
    state_error = torch.cat([position_error_xy_body, yaw_error], dim=-1)

    obs = torch.cat(
        [
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            state_error,
            leg_joint_pos_rel,
            leg_joint_vel,
            pendulum_joint_pos,
            pendulum_joint_vel,
            prev_action_for_obs,
            clock_inputs,
        ],
        dim=-1,
    )
    if obs.shape[-1] != 56:
        raise RuntimeError(f"Expected 56D observation, got {obs.shape[-1]}.")
    return obs


def compute_base_tilt_rad(robot: Articulation) -> torch.Tensor:
    projected_gravity_b = robot.data.projected_gravity_b
    return torch.atan2(torch.linalg.norm(projected_gravity_b[:, :2], dim=1), -projected_gravity_b[:, 2])


def main() -> None:
    if not USD_PATH.is_file():
        raise FileNotFoundError(f"Missing USD: {USD_PATH}")
    if not POLICY_PATH.is_file():
        raise FileNotFoundError(f"Missing policy: {POLICY_PATH}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=SIM_DT,
            render_interval=DECIMATION,
            physics_material=make_physics_material_cfg(),
        )
    )
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.35])

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=make_physics_material_cfg(),
        num_envs=1,
        env_spacing=4.0,
        debug_vis=False,
    )
    terrain = terrain_cfg.class_type(terrain_cfg)
    env_origins = terrain.env_origins.to(dtype=torch.float32)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    robot = Articulation(cfg=build_robot_cfg())

    sim.reset()
    robot.reset()
    robot.update(SIM_DT)

    leg_joint_ids = find_leg_joint_ids_from_robot_order(robot)
    pendulum_joint_ids = find_named_joint_ids(robot, PENDULUM_JOINT_NAMES)
    print_joint_orders(robot, leg_joint_ids, pendulum_joint_ids)
    apply_pendulum_joint_limits(robot, pendulum_joint_ids)

    reset_robot_state(sim, robot, pendulum_joint_ids, env_origins)

    action_low, action_high, joint_pos_low, joint_pos_high = compute_action_and_joint_bounds(robot, leg_joint_ids)

    device = torch.device(robot.device)
    policy = torch.jit.load(str(POLICY_PATH), map_location=device)
    policy.eval()

    control_dt = SIM_DT * DECIMATION
    lpf_alpha = math.exp(-2.0 * math.pi * ACTION_LPF_CUTOFF_HZ * control_dt) if ENABLE_ACTION_LPF else 0.0
    max_action_delay_steps = ACTION_DELAY_STEPS_MAX if ENABLE_ACTION_DELAY else 0

    action_delay_buffer = torch.zeros(
        (robot.num_instances, 12, max_action_delay_steps + 1),
        dtype=torch.float32,
        device=device,
    )
    if ENABLE_ACTION_DELAY and ACTION_DELAY_RANDOMIZE_PER_RESET:
        action_delay_steps = torch.randint(
            ACTION_DELAY_STEPS_MIN,
            ACTION_DELAY_STEPS_MAX + 1,
            (robot.num_instances,),
            dtype=torch.long,
            device=device,
        )
    elif ENABLE_ACTION_DELAY:
        action_delay_steps = torch.full(
            (robot.num_instances,),
            ACTION_DELAY_STEPS_MAX,
            dtype=torch.long,
            device=device,
        )
    else:
        action_delay_steps = torch.zeros((robot.num_instances,), dtype=torch.long, device=device)

    actions_filtered = torch.zeros((robot.num_instances, 12), dtype=torch.float32, device=device)
    prev_action_for_obs = torch.zeros((robot.num_instances, 12), dtype=torch.float32, device=device)
    gait_index = torch.zeros((robot.num_instances,), dtype=torch.float32, device=device)
    clock_inputs = torch.zeros((robot.num_instances, 4), dtype=torch.float32, device=device)

    target_xy = torch.tensor([[TARGET_X, TARGET_Y]], dtype=torch.float32, device=device)
    target_yaw = torch.tensor([TARGET_YAW], dtype=torch.float32, device=device)
    is_rendering = sim.has_gui() or sim.has_rtx_sensors()
    sim_step_counter = 0

    while simulation_app.is_running():
        start_time = time.time()
        if torch.any(compute_base_tilt_rad(robot) > BASE_TILT_RESET_RAD):
            reset_robot_state(sim, robot, pendulum_joint_ids, env_origins)
            action_delay_buffer.zero_()
            actions_filtered.zero_()
            prev_action_for_obs.zero_()
            gait_index.zero_()
            clock_inputs.zero_()
            if ENABLE_ACTION_DELAY and ACTION_DELAY_RANDOMIZE_PER_RESET:
                action_delay_steps = torch.randint(
                    ACTION_DELAY_STEPS_MIN,
                    ACTION_DELAY_STEPS_MAX + 1,
                    (robot.num_instances,),
                    dtype=torch.long,
                    device=device,
                )
            elif ENABLE_ACTION_DELAY:
                action_delay_steps.fill_(ACTION_DELAY_STEPS_MAX)
            continue

        obs = compute_observation(
            robot=robot,
            leg_joint_ids=leg_joint_ids,
            pendulum_joint_ids=pendulum_joint_ids,
            prev_action_for_obs=prev_action_for_obs,
            clock_inputs=clock_inputs,
            target_xy=target_xy,
            target_yaw=target_yaw,
        )

        with torch.no_grad():
            actions_raw_policy = policy(obs)
        if isinstance(actions_raw_policy, tuple):
            actions_raw_policy = actions_raw_policy[0]
        if actions_raw_policy.ndim == 1:
            actions_raw_policy = actions_raw_policy.unsqueeze(0)
        actions_raw_policy = actions_raw_policy[:, :12]

        if ENABLE_PER_JOINT_ACTION_BOUNDS:
            actions_bounded = torch.clamp(actions_raw_policy, min=action_low, max=action_high)
        else:
            actions_bounded = actions_raw_policy.clone()

        if ENABLE_ACTION_DELAY:
            action_delay_buffer = torch.roll(action_delay_buffer, shifts=1, dims=2)
            action_delay_buffer[:, :, 0] = actions_bounded
            delay_idx = action_delay_steps.clamp(max=max_action_delay_steps).view(robot.num_instances, 1, 1)
            delay_idx = delay_idx.expand(-1, 12, 1)
            actions_delayed = torch.gather(action_delay_buffer, dim=2, index=delay_idx).squeeze(-1)
        else:
            actions_delayed = actions_bounded.clone()

        if ENABLE_ACTION_LPF:
            actions_filtered = lpf_alpha * actions_filtered + (1.0 - lpf_alpha) * actions_delayed
            if ENABLE_PER_JOINT_ACTION_BOUNDS:
                actions_filtered = torch.clamp(actions_filtered, min=action_low, max=action_high)
            actions_executed = actions_filtered.clone()
        else:
            actions_executed = actions_delayed.clone()

        desired_joint_pos = robot.data.default_joint_pos[:, leg_joint_ids] + ACTION_SCALE * actions_executed
        if ENABLE_DESIRED_JOINT_POS_HARD_CLAMP:
            desired_joint_pos = torch.clamp(desired_joint_pos, min=joint_pos_low, max=joint_pos_high)

        for _ in range(DECIMATION):
            robot.set_joint_position_target(desired_joint_pos, joint_ids=leg_joint_ids)
            robot.write_data_to_sim()
            sim.step(render=False)
            sim_step_counter += 1
            if is_rendering and sim_step_counter % DECIMATION == 0:
                sim.render()
            robot.update(SIM_DT)

        prev_action_for_obs = actions_executed.clone()
        gait_index, clock_inputs = update_clock_inputs(gait_index, clock_inputs, control_dt)

        if RUN_REAL_TIME:
            sleep_time = control_dt - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
