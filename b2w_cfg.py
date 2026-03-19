"""Configuration for the Unitree B2-W wheeled-legged quadruped."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Robot configuration
##

B2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/b2w_asset_simple/b2w.usd",
        activate_contact_sensors=False,
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
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),  # spawned ~0.55m above ground
        joint_pos={
            # Legs: standing pose
            "FL_hip_joint":    0.0,
            "FR_hip_joint":    0.0,
            "RL_hip_joint":    0.0,
            "RR_hip_joint":    0.0,
            "FL_thigh_joint":  0.9,
            "FR_thigh_joint":  0.9,
            "RL_thigh_joint":  0.9,
            "RR_thigh_joint":  0.9,
            "FL_calf_joint":  -1.8,
            "FR_calf_joint":  -1.8,
            "RL_calf_joint":  -1.8,
            "RR_calf_joint":  -1.8,
            # Wheels: zero velocity
            "FL_wheel_joint":  0.0,
            "FR_wheel_joint":  0.0,
            "RL_wheel_joint":  0.0,
            "RR_wheel_joint":  0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # --- Hip joints (abduction/adduction, ±0.87 rad) ---
        "legs_hip": DCMotorCfg(
            joint_names_expr=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        ),
        # --- Thigh joints (forward/back swing, ±4.69 rad range) ---
        "legs_thigh": DCMotorCfg(
            joint_names_expr=["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        ),
        # --- Calf joints (knee, higher torque) ---
        "legs_calf": DCMotorCfg(
            joint_names_expr=["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            effort_limit=300.0,
            saturation_effort=300.0,
            velocity_limit=23.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        ),
        # --- Wheel joints (continuous rotation, velocity-like control) ---
        # Low stiffness, moderate damping: policy outputs torque commands
        "wheels": DCMotorCfg(
            joint_names_expr=["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"],
            effort_limit=20.0,
            saturation_effort=20.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=2.0,  # velocity control needs higher damping
            friction=0.0,
        ),
    },
)