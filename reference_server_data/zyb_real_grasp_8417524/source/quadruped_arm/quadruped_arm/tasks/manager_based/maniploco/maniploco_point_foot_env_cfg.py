# tasks/manager_based/maniploco/maniploco_point_foot_env_cfg.py
from isaaclab.utils import configclass

from quadruped_arm.robots.robot_cfg import ZYB_QUADRUPED_ARM_POINTFOOT_Cfg
from .maniploco_env_cfg import ManipLocoEnvCfg, LEG_JOINTS


POINTFOOT_OBS_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]


@configclass
class ManipLocoPointFootEnvCfg(ManipLocoEnvCfg):
    def __post_init__(self):
        # 先执行父类，把公共配置都建好
        super().__post_init__()

        # 1) 换 point-foot 机器人
        self.scene.robot = ZYB_QUADRUPED_ARM_POINTFOOT_Cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # 2) 去掉 wheel action
        self.actions.wheel_vel = None

        # 3) observation 去掉轮关节
        self.observations.policy.obs.params["obs_joint_names"] = POINTFOOT_OBS_JOINTS

        # ---------- helper ----------
        def _set_reward_joint_names(term_name: str, joint_names):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.params["joint_names"] = joint_names

        def _set_reward_param(term_name: str, key: str, value):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.params[key] = value

        def _disable_reward(term_name: str):
            if hasattr(self.rewards, term_name):
                setattr(self.rewards, term_name, None)

        # 4) 所有原来 BASE_JOINTS 的项，改成只看腿
        _set_reward_joint_names("torques", LEG_JOINTS)
        _set_reward_joint_names("energy_square", LEG_JOINTS)
        _set_reward_joint_names("work", LEG_JOINTS)
        _set_reward_joint_names("dof_acc", LEG_JOINTS)
        _set_reward_joint_names("delta_torques", LEG_JOINTS)
        _set_reward_joint_names("torques_walking", LEG_JOINTS)
        _set_reward_joint_names("torques_standing", LEG_JOINTS)
        _set_reward_joint_names("energy_square_walking", LEG_JOINTS)
        _set_reward_joint_names("energy_square_standing", LEG_JOINTS)

        # 5) policy action 只有 12 维腿动作
        _set_reward_param("action_rate", "action_dim", 12)

        # 6) wheel 专属 reward 全部禁用
        _disable_reward("wheel_idle_speed")
        _disable_reward("wheel_action_l2")
        _disable_reward("wheel_action_rate")
        _disable_reward("wheel_forward_use")
        _disable_reward("wheel_turn_support")

        # 7) 注意：这里不要再调用不存在的 _apply_reward_scales()
        # self._apply_reward_scales(include_wheel=False)   # 删掉



# =============== PLAY ================
@configclass
class ManipLocoPointFootPlayEnvCfg(ManipLocoPointFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0

        self.commands.ee_goal.debug_vis = True
        self.commands.locomotion.debug_vis = True

        self.events.reset_root.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_root.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_joints.params["velocity_range"] = (0.0, 0.0)

        self.episode_length_s = 30.0

        # self.scene.robot.spawn.articulation_props.fix_root_link = True

        # # # 只做ik验证
        # self.commands.locomotion.ranges.lin_vel_x = (0.8, 0.8)
        # self.commands.locomotion.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.locomotion.ranges.ang_vel_z = (0.0, 0.0)

        # self.actions.leg_pos.scale = {
        #     ".*_hip_.*": 0.0,
        #     ".*_thigh_.*": 0.0,
        #     ".*_calf_.*": 0.0,
        # }