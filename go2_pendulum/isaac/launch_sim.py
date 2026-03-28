#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import omni.graph.core as og
import omni.usd
import torch
import usdrt.Sdf

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaacsim.core.utils.extensions import enable_extension


ROOT_DIR = Path(__file__).resolve().parent
USD_PATH = ROOT_DIR / "model" / "go2_pendulum.usd"

SIM_DT = 1.0 / 200.0
SIM_DEVICE = "cpu"
ROBOT_PRIM_PATH = "/World/Robot"
DEFAULT_ARTICULATION_PRIM_PATH = "/World/Robot/base"
DEFAULT_CHASSIS_PRIM_PATH = "/World/Robot/base"

TOPIC_JOINT_STATES = "/joint_states"
TOPIC_JOINT_COMMAND = "/joint_command"
TOPIC_ODOM = "/odom"


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
        prim_path=ROBOT_PRIM_PATH,
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


def find_leg_joint_ids(robot: Articulation) -> torch.Tensor:
    ids = [
        i
        for i, name in enumerate(robot.joint_names)
        if name.endswith("_hip_joint") or name.endswith("_thigh_joint") or name.endswith("_calf_joint")
    ]
    if len(ids) != 12:
        raise RuntimeError(f"Expected 12 leg joints, got {len(ids)} from joint list: {robot.joint_names}")
    return torch.tensor(ids, dtype=torch.long, device=robot.device)


def resolve_articulation_prim_path_fallback() -> str:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DEFAULT_ARTICULATION_PRIM_PATH).IsValid():
        return DEFAULT_ARTICULATION_PRIM_PATH
    if stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        return ROBOT_PRIM_PATH
    raise RuntimeError(
        f"Failed to resolve articulation prim from '{DEFAULT_ARTICULATION_PRIM_PATH}' or '{ROBOT_PRIM_PATH}'."
    )


def resolve_chassis_prim_path_fallback() -> str:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DEFAULT_CHASSIS_PRIM_PATH).IsValid():
        return DEFAULT_CHASSIS_PRIM_PATH
    if stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
        return ROBOT_PRIM_PATH
    raise RuntimeError(f"Failed to resolve chassis prim from '{DEFAULT_CHASSIS_PRIM_PATH}' or '{ROBOT_PRIM_PATH}'.")


def create_ros_action_graph(articulation_prim_path: str, chassis_prim_path: str) -> None:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishOdometry.inputs:context"),
                ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", TOPIC_JOINT_STATES),
                ("PublishJointState.inputs:queueSize", 1),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(articulation_prim_path)]),
                ("SubscribeJointState.inputs:topicName", TOPIC_JOINT_COMMAND),
                ("SubscribeJointState.inputs:queueSize", 1),
                ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(chassis_prim_path)]),
                ("PublishOdometry.inputs:topicName", TOPIC_ODOM),
                ("PublishOdometry.inputs:queueSize", 1),
                ("PublishOdometry.inputs:odomFrameId", "odom"),
                ("PublishOdometry.inputs:chassisFrameId", "base"),
                ("PublishOdometry.inputs:publishRawVelocities", False),
            ],
        },
    )


def main() -> None:
    if not USD_PATH.is_file():
        raise FileNotFoundError(f"Missing USD file: {USD_PATH}")

    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            device=SIM_DEVICE,
            dt=SIM_DT,
            render_interval=1,
            use_fabric=False,
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

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    robot = Articulation(cfg=build_robot_cfg())

    sim.reset()
    robot.reset()
    robot.update(SIM_DT)

    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += terrain.env_origins
    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    sim.forward()
    robot.update(SIM_DT)

    articulation_prim_path = None
    if robot.root_physx_view is not None and len(robot.root_physx_view.prim_paths) > 0:
        articulation_prim_path = str(robot.root_physx_view.prim_paths[0])
    if articulation_prim_path is None:
        articulation_prim_path = resolve_articulation_prim_path_fallback()

    chassis_prim_path = articulation_prim_path
    if not omni.usd.get_context().get_stage().GetPrimAtPath(chassis_prim_path).IsValid():
        chassis_prim_path = resolve_chassis_prim_path_fallback()

    create_ros_action_graph(articulation_prim_path, chassis_prim_path)

    leg_joint_ids = find_leg_joint_ids(robot)
    stand_target = robot.data.default_joint_pos[:, leg_joint_ids].clone()
    name_to_leg_policy_idx = {name: i for i, name in enumerate([robot.joint_names[j] for j in leg_joint_ids.tolist()])}
    current_leg_target = stand_target.clone()
    ros_command_seen = False

    # Send an initial standing command immediately so the robot does not drop before ROS bringup starts.
    robot.set_joint_position_target(current_leg_target, joint_ids=leg_joint_ids)
    robot.write_data_to_sim()

    print(f"[INFO] Robot prim: {ROBOT_PRIM_PATH}")
    print(f"[INFO] Simulation device: requested={SIM_DEVICE}, runtime={sim.device}")
    print(f"[INFO] Articulation prim for control/joint states: {articulation_prim_path}")
    print(f"[INFO] Chassis prim for odometry: {chassis_prim_path}")
    print(f"[INFO] ROS topics: {TOPIC_JOINT_STATES}, {TOPIC_ODOM}, {TOPIC_JOINT_COMMAND}")
    print("[INFO] Applying /joint_command in Python through IsaacLab actuators (position targets).")
    print("[INFO] Default stand target is held until /joint_command updates targets.")

    while simulation_app.is_running():
        cmd_names = list(og.Controller.get(og.Controller.attribute("/ActionGraph/SubscribeJointState.outputs:jointNames")))
        cmd_pos = list(og.Controller.get(og.Controller.attribute("/ActionGraph/SubscribeJointState.outputs:positionCommand")))

        applied = 0
        n = min(len(cmd_names), len(cmd_pos))
        for i in range(n):
            name = str(cmd_names[i])
            if name in name_to_leg_policy_idx:
                current_leg_target[0, name_to_leg_policy_idx[name]] = float(cmd_pos[i])
                applied += 1

        # Fallback: if names are not populated by the bridge but positions are,
        # assume policy leg order (same order used by rl_controller).
        if applied == 0 and len(cmd_pos) >= 12:
            for i in range(12):
                current_leg_target[0, i] = float(cmd_pos[i])
            applied = 12

        if (not ros_command_seen) and applied > 0:
            ros_command_seen = True
            print("[INFO] First /joint_command received. Updating leg targets from ROS.")

        robot.set_joint_position_target(current_leg_target, joint_ids=leg_joint_ids)
        robot.write_data_to_sim()

        sim.step(render=True)
        robot.update(SIM_DT)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
