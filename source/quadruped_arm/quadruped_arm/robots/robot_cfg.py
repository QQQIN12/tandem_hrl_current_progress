"""
configurations for zyb's quadurped armed robot

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from pathlib import Path
CURRENT_DIR = Path(__file__).parent



# TANDEM uses a private copy with wheel-foot mass properties corrected from the
# authored near-zero inertia.  The original shared USD remains untouched.
ZYB_QUADRUPED_ARM_USD = str(CURRENT_DIR/"assets/quadruped_arm_V3_tandem_inertia.usd")

ZYB_QUADRUPED_ARM_POINT_FOOT_USD = str(CURRENT_DIR/"assets/quadruped_arm_point_foot_v2.usd")

B2W_USD = str(CURRENT_DIR/"assets/b2w.usd")

B2W_FIXED_USD = str(CURRENT_DIR/"assets/B2W_fixed_wheel.usd")


# --------- raw B2W platform | TODO: need to check unitree's parameters ---------
B2W_Cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=B2W_USD,
        activate_contact_sensors=True,

        # rigid body properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # common
            disable_gravity=False,
            retain_accelerations=False,

            # damping
            linear_damping=0.0,
            angular_damping=0.0,

            # limitations
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        # articulation properties
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    
    # init states
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        # B2
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,

            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
        },

        joint_vel={".*": 0.0},
    ),

    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160,
            damping=5,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=20.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=0.2,
            friction=0.01,
        ),
    },
)


# --------- quadruped armed robot Cfg ---------
ZYB_QUADRUPED_ARM_Cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ZYB_QUADRUPED_ARM_USD,
        activate_contact_sensors=True,

        # rigid body properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # common
            disable_gravity=False,
            retain_accelerations=False,

            # damping
            linear_damping=0.0,
            angular_damping=0.0,

            # limitations
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        # articulation properties
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,  # 源自UnitreeUsdFileCfg参数
            solver_velocity_iteration_count=4,
        ),
    ),
    
    # init states
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),

        joint_pos={
            ".*R_hip_joint": -0.2,
            ".*L_hip_joint": 0.2,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,

            ".*_foot_wheel_joint": 0.0,
        },

        joint_vel={".*": 0.0},
    ),

    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160,
            damping=5,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),

        # Go2W的轮子关节设计
        # The shared USD has very small wheel inertia. An explicit IdealPD
        # velocity loop is numerically unstable at the 1/120 s physics step.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_wheel_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.01,
        ),

        # Franka Parameters
        "arm_proximal": ImplicitActuatorCfg(
            joint_names_expr=[r"^joint[1-3]$"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.175,
            stiffness=1000.0,
            damping=80.0,
        ),

        "arm_distal": ImplicitActuatorCfg(
            joint_names_expr=[r"^joint[4-6]$"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),

        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[r"^joint[78]$"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17,
            damping=0.02,
        ),
    },
)

ZYB_QUADRUPED_ARM_POINTFOOT_Cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ZYB_QUADRUPED_ARM_POINT_FOOT_USD,
        activate_contact_sensors=True,

        # rigid body properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # common
            disable_gravity=False,
            retain_accelerations=False,

            # damping
            linear_damping=0.0,
            angular_damping=0.0,

            # limitations
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        # articulation properties
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,  # 源自UnitreeUsdFileCfg参数
            solver_velocity_iteration_count=4,
        ),
    ),
    
    # init states
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),

        joint_pos={
            ".*R_hip_joint": -0.2,
            ".*L_hip_joint": 0.2,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },

        joint_vel={".*": 0.0},
    ),

    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160,
            damping=5,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),


        # Franka Parameters
        "arm_proximal": ImplicitActuatorCfg(
            joint_names_expr=[r"^joint[1-3]$"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.175,
            stiffness=1000.0,
            damping=80.0,
        ),

        "arm_distal": ImplicitActuatorCfg(
            joint_names_expr=[r"^joint[4-6]$"],
            effort_limit_sim=720.0,
            velocity_limit_sim=2.61,
            stiffness=1000.0,
            damping=80.0,
        ),

        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[r"^joint[78]$"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17,
            damping=0.02,
        ),
    },

    # fmt: off
    # joint_sdk_names=[
    #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    # ],
    # fmt: on
)
