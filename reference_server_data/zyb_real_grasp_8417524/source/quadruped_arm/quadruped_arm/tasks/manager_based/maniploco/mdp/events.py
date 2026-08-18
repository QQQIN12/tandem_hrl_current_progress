import torch

def set_vbc_priv_buf(env, env_ids, dim: int = 18):
    """reset 时初始化 privileged buffer"""
    import torch
    if not hasattr(env, "vbc_priv_buf"):
        env.vbc_priv_buf = torch.zeros(env.num_envs, dim, device=env.device)
    env.vbc_priv_buf[env_ids] = 0.0


def reset_reward_buffers(env, env_ids, feet_count: int = 4):
    """reset 时把 reward 的时序 buffer 清零，避免跨 episode 串扰。"""
    if hasattr(env, "vbc_last_actions"):
        env.vbc_last_actions.zero_()
    if hasattr(env, "vbc_last_dof_vel"):
        env.vbc_last_dof_vel.zero_()
    if hasattr(env, "vbc_last_torques"):
        env.vbc_last_torques.zero_()
    if hasattr(env, "vbc_last_contact_forces"):
        env.vbc_last_contact_forces.zero_()
    if hasattr(env, "feet_air_time"):
        env.feet_air_time.zero_()