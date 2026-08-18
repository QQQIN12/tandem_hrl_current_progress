# tasks/manager_based/maniploco/config/scale_cfg.py
from isaaclab.utils import configclass

@configclass
class BaseScale:
    VBC_LOCO_SCALES = {
        # The old VBC shaping mainly rewarded motion in the commanded
        # direction.  It did not sufficiently penalize under-speed, so the
        # policy could remain almost stationary and still survive.  Keep the
        # directional terms, but add symmetric absolute tracking terms below.
        "tracking_lin_vel": 2.0,
        "tracking_lin_vel_max": 1.5,
        "tracking_ang_vel": 1.0,
        "tracking_lin_vel_x_exp": 5.0,
        "tracking_ang_vel_yaw_exp": 5.0,
        "torques": -2.5e-5,
        "stand_still": None,#1.0,
        "walking_dof": 0.1,#1.5,
        "alive": 1.0,
        "lin_vel_z": -1.5,
        "roll": -1.0,
        "ang_vel_xy": -0.2,
        "dof_acc": -7.5e-7,
        "collision": -10.0,
        "action_rate": -0.015,
        "dof_pos_limits": -10.0,
        "hip_pos": -0.2,
        "feet_jerk": -0.0002,
        "feet_drag": -0.08,
        "feet_contact_forces": -0.001,
        "base_height": -2.0,

        "flat_orientation_l2": -2.0,
        "air_time_variance": -1.0,

        # 这些VBC里也有，默认为0可以保留用于日志
        "tracking_lin_vel_x_l1": None,#0.0,
        # Enabled above with a strong symmetric tracking weight.
        "tracking_contacts_shaped_force": None,#-2.0,   # observe_gait_commands=False 时函数里会返回0
        "tracking_contacts_shaped_vel": None,#-2.0,
        "feet_air_time": 0.05,
        # "feet_height": 0.1,
        "delta_torques": -1.0e-7,  
        "work": -0.003,            
        "energy_square": None, #0.0,
        "dof_default_pos": None,#0,0
        "dof_error": None,#0,0
        "orientation": None,#0,0
        "orientation_walking": None,#0,0
        "orientation_standing": None,#0,0
        "torques_walking": None,#0,0
        "torques_standing": None,#0,0
        "energy_square_walking": None,#0,0
        "energy_square_standing": None,#0,0
        "base_height_walking": None,#0,0
        "base_height_standing": None,#0,0
        "penalty_lin_vel_y": None,#0,0

        "hip_action_l2": None,#0,0
        "leg_energy_abs_sum": None,#0,0
        "leg_energy_sum_abs": None,#0,0
        "leg_action_l2": -0.01,#0,0
        "leg_energy": None,#0,0
    }

    VBC_ARM_SCALES = {
        "tracking_ee_sphere": None,#0,0
        "tracking_ee_world": None,#3.0,#0.8,
        "tracking_ee_sphere_walking": None,#0,0
        "tracking_ee_sphere_standing": None,#0,0
        "tracking_ee_cart": None,
        "tracking_ee_orn": None,#0.8,#None,#0,0
        "tracking_ee_orn_ry": None,#0.8,
        "arm_energy_abs_sum": None,#-0.01,#None,
    }

    WHEEL_SCALES = {
        # 你自己设计的B2W特化项（建议先用这组）
        "wheel_idle_speed": -0.003,
        "wheel_action_l2": -0.001,
        "wheel_action_rate": -0.002,
        "teacher_ensemble_match": 0.75,
        "wheel_forward_use": 0.6,
        "wheel_turn_support": 0.45,
    }

    TERMINATION_SCALES = {
        "term_bad_contact": -10000.0,  # 实际约 -10
        "term_tilt": -10000.0,         # 实际约 -10
        "term_low_height": -10000.0,    # 实际约 -5
    }
