# tasks/manager_based/maniploco/mdp/terminations.py
import torch
from isaaclab.managers import SceneEntityCfg


def bad_contacts(
    env,
    sensor_names: list[str],
    thresh: float = 50.0,
    use_filtered: bool = True,
) -> torch.Tensor:
    """对一组单独的 contact sensors 做 bad contact 判定。

    每个 sensor 对应一个 body，并且建议都配置了
    filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"]

    Args:
        env: 环境
        sensor_names: 需要监控的 sensor 名字列表
        thresh: 接触力阈值
        use_filtered: True 时优先用 force_matrix_w（对地过滤后的接触）；
                      False 时用 net_forces_w（总接触力）
    Returns:
        done: (N,) bool tensor
    """
    per_sensor_force = []
    pretty_names = []

    for sname in sensor_names:
        sensor = env.scene[sname]

        # ---------- 优先使用 ground-filtered contact ----------
        if use_filtered and sensor.data.force_matrix_w is not None:
            # 可能形状是 (N, 1, M, 3) 或类似，统一拉平成 (..., 3)
            f = sensor.data.force_matrix_w
            f_norm = torch.norm(f.reshape(f.shape[0], -1, 3), dim=-1)   # (N, K)
            f_val = torch.max(f_norm, dim=1).values                      # (N,)
        else:
            # fallback: 总接触力
            f = sensor.data.net_forces_w
            f_norm = torch.norm(f.reshape(f.shape[0], -1, 3), dim=-1)   # (N, K)
            f_val = torch.max(f_norm, dim=1).values                      # (N,)

        per_sensor_force.append(f_val)
        pretty_names.append(sname.replace("_contact", ""))

    # (N, num_sensors)
    force_table = torch.stack(per_sensor_force, dim=1)

    # 任意一个 sensor 超阈值就 done
    done = torch.any(force_table > thresh, dim=1)

    # 调试打印
    if env.common_step_counter < 10 or torch.any(done):
        env_id = torch.where(done)[0][0].item() if torch.any(done) else 0
        vals = force_table[env_id].detach().cpu().tolist()
        print(f"[step {env.common_step_counter}] bad contact:", list(zip(pretty_names, vals)))

    return done


def base_tilt(env, asset_cfg: SceneEntityCfg, limit: float = 0.6) -> torch.Tensor:
    """
    总倾角大于 limit 就 done
    """
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b  # (N,3)
    # rough proxy: sqrt(gx^2+gy^2) ~ sin(tilt)
    tilt = torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)
    return tilt > torch.sin(torch.tensor(limit, device=tilt.device))

def base_height_low(env, asset_cfg: SceneEntityCfg, z_min: float = 0.25) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # print(f"------------{asset.data.root_pos_w[:, 2]}")
    return asset.data.root_pos_w[:, 2] < z_min