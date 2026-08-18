"""Lower-body diagnostics using a bounded wheel torque teacher."""

from isaaclab.utils import configclass

from .mdp.safe_wheel_torque_action import SafeDifferentialWheelTorqueActionCfg
from .mobility_lower_env_cfg import MobilityLowerEnvCfg


def _torque_wheel_cfg(**overrides):
    values = dict(
        asset_name="robot",
        joint_names=[
            "FL_foot_wheel_joint",
            "FR_foot_wheel_joint",
            "RL_foot_wheel_joint",
            "RR_foot_wheel_joint",
        ],
        scale=1.0,
        preserve_order=True,
        command_name="locomotion",
        wheel_radius=0.11,
        track_width=0.4693,
        wheel_dir_signs=(1.0, 1.0, 1.0, 1.0),
        max_ref_vx=0.25,
        max_ref_wz=0.10,
        vx_feedback_gain=0.15,
        wz_feedback_gain=1.0,
        wheel_velocity_kp=3.0,
        yaw_torque_gain=12.0,
        torque_limit=8.0,
        residual_scale=0.10,
        max_torque_rate=100.0,
    )
    values.update(overrides)
    return SafeDifferentialWheelTorqueActionCfg(**values)


@configclass
class WheelTorqueLowerEnvCfg(MobilityLowerEnvCfg):
    """Torque-controlled wheel teacher with nominal yaw torque gain."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.max_policy_residual = 0.0
        self.actions.leg_pos.posture_feedback_enabled = False
        self.actions.leg_pos.safety_gate_enabled = True
        self.actions.leg_pos.turn_coordination_enabled = False

        # MobilityLower uses velocity-drive damping 6 for its wheel tests.
        # The torque diagnostic returns to the asset's low passive damping so
        # the commanded effort is not hidden by a large viscous brake.
        self.scene.robot.actuators["wheels"].damping = 0.5
        self.actions.wheel_vel = _torque_wheel_cfg()


@configclass
class WheelTorqueHighYawLowerEnvCfg(WheelTorqueLowerEnvCfg):
    """Torque teacher with a bounded stronger yaw moment ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.wheel_vel.yaw_torque_gain = 24.0
