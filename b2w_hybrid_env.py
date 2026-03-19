"""B2-W Hybrid Locomotion Environment for IsaacLab v1.2.0

Unified policy controlling all 16 joints:
- 12 leg joints (hip/thigh/calf) via position control
- 4 wheel joints via velocity control

This is the novel contribution: a single policy that learns to
coordinate leg posture and wheel spinning for locomotion.
"""

from __future__ import annotations

import math
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.envs.mdp as base_mdp
from omni.isaac.lab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

import sys
sys.path.insert(0, "/home/ubuntu/b2w_project")
from b2w_cfg import B2W_CFG


##
# Scene
##

@configclass
class B2WHybridSceneCfg(InteractiveSceneCfg):
    """Flat ground + B2-W robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = B2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


##
# Custom observation functions
##

def wheel_velocities(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wheel joint velocities [FL, FR, RL, RR]."""
    robot = env.scene["robot"]
    wheel_ids, _ = robot.find_joints(
        ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]
    )
    return robot.data.joint_vel[:, wheel_ids]


def leg_joint_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Leg joint positions [hip x4, thigh x4, calf x4] = 12 dims."""
    robot = env.scene["robot"]
    leg_ids, _ = robot.find_joints([
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ])
    return robot.data.joint_pos[:, leg_ids]


def leg_joint_velocities(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Leg joint velocities = 12 dims."""
    robot = env.scene["robot"]
    leg_ids, _ = robot.find_joints([
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ])
    return robot.data.joint_vel[:, leg_ids]


##
# Custom reward functions
##

def track_lin_vel_xy(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2]
            - env.scene["robot"].data.root_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_yaw_rate(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    yaw_rate_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - env.scene["robot"].data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-yaw_rate_error / std**2)


def penalize_lin_vel_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(env.scene["robot"].data.root_lin_vel_b[:, 2])


def penalize_ang_vel_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.scene["robot"].data.root_ang_vel_b[:, :2]), dim=1)


def penalize_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def penalize_flat_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]), dim=1
    )


def penalize_leg_torques(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large leg torques to encourage efficient locomotion."""
    robot = env.scene["robot"]
    leg_ids, _ = robot.find_joints([
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ])
    return torch.sum(torch.square(robot.data.applied_torque[:, leg_ids]), dim=1)


def wheel_engagement(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward wheel spinning when commanded to move forward.
    This is a key metric for the paper: how much do wheels contribute?
    """
    robot = env.scene["robot"]
    wheel_ids, _ = robot.find_joints(
        ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]
    )
    cmd_speed = torch.norm(
        env.command_manager.get_command("base_velocity")[:, :2], dim=1
    )
    wheel_speed = torch.mean(torch.abs(robot.data.joint_vel[:, wheel_ids]), dim=1)
    # Reward wheel spinning proportional to command speed
    return cmd_speed * wheel_speed * 0.1


def penalize_stand_still(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    actual_speed = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    should_move = (cmd_speed > 0.1).float()
    not_moving = (actual_speed < 0.1).float()
    return should_move * not_moving


##
# Terminations
##

def low_velocity_timeout(env: ManagerBasedRLEnv) -> torch.Tensor:
    cmd_speed = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    actual_speed = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    not_moving = (cmd_speed > 0.3) & (actual_speed < 0.05)
    return not_moving


##
# Observations
##

@configclass
class B2WHybridObsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # 3: base linear velocity
        base_lin_vel = ObsTerm(
            func=base_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # 3: base angular velocity
        base_ang_vel = ObsTerm(
            func=base_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        # 3: projected gravity
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 12: leg joint positions
        leg_pos = ObsTerm(
            func=leg_joint_positions,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 12: leg joint velocities
        leg_vel = ObsTerm(
            func=leg_joint_velocities,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # 4: wheel velocities
        wheel_vel = ObsTerm(
            func=wheel_velocities,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        # 3: velocity commands
        velocity_commands = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # Total: 3+3+3+12+12+4+3 = 40 dims

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Actions: 16-dim hybrid
##

@configclass
class B2WHybridActionsCfg:
    """16-dim action space: 12 leg positions + 4 wheel velocities."""

    # Leg joints: position control with PD gains from actuator config
    leg_positions = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ],
        scale=0.5,   # maps [-1,1] -> [-0.5, 0.5] rad offset from current
        use_default_offset=True,
    )

    # Wheel joints: velocity control
    wheel_velocities = base_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"],
        scale=10.0,  # maps [-1,1] -> [-10, 10] rad/s
    )


##
# Events
##

@configclass
class B2WHybridEventCfg:

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


##
# Rewards
##

@configclass
class B2WHybridRewardsCfg:

    # Primary: velocity tracking
    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy,
        weight=1.5,
        params={"std": 0.25, "command_name": "base_velocity"},
    )
    track_yaw_rate = RewTerm(
        func=track_yaw_rate,
        weight=0.75,
        params={"std": 0.25, "command_name": "base_velocity"},
    )

    # Novel metric: reward wheel engagement (key for paper)
    wheel_engagement = RewTerm(func=wheel_engagement, weight=0.5)

    # Penalties
    lin_vel_z_l2 = RewTerm(func=penalize_lin_vel_z, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=penalize_ang_vel_xy, weight=-0.05)
    action_rate_l2 = RewTerm(func=penalize_action_rate, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=penalize_flat_orientation, weight=-0.5)
    leg_torques_l2 = RewTerm(func=penalize_leg_torques, weight=-1e-4)
    stand_still_penalty = RewTerm(
        func=penalize_stand_still,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )


##
# Terminations
##

@configclass
class B2WHybridTerminationsCfg:

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    low_velocity = DoneTerm(func=low_velocity_timeout, time_out=False)


##
# Commands
##

@configclass
class B2WHybridCommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# Full env config
##

@configclass
class B2WHybridEnvCfg(ManagerBasedRLEnvCfg):

    scene: B2WHybridSceneCfg = B2WHybridSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: B2WHybridObsCfg = B2WHybridObsCfg()
    actions: B2WHybridActionsCfg = B2WHybridActionsCfg()
    rewards: B2WHybridRewardsCfg = B2WHybridRewardsCfg()
    terminations: B2WHybridTerminationsCfg = B2WHybridTerminationsCfg()
    events: B2WHybridEventCfg = B2WHybridEventCfg()
    commands: B2WHybridCommandsCfg = B2WHybridCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005       # 200 Hz physics
        self.decimation = 4       # 50 Hz policy
        self.episode_length_s = 20.0
        self.viewer.eye = (10.0, 0.0, 6.0)