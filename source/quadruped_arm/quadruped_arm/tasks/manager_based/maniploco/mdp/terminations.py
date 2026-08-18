# tasks/manager_based/maniploco/mdp/terminations.py
import torch
from isaaclab.managers import SceneEntityCfg


def bad_contacts(
    env,
    sensor_names: list[str],
    thresh: float = 50.0,
    use_filtered: bool = True,
) -> torch.Tensor:
    """Return environments with excessive force on monitored contact sensors."""

    per_sensor_force = []

    for sname in sensor_names:
        sensor = env.scene[sname]

        if use_filtered and sensor.data.force_matrix_w is not None:
            f = sensor.data.force_matrix_w
            f_norm = torch.norm(f.reshape(f.shape[0], -1, 3), dim=-1)
            f_val = torch.max(f_norm, dim=1).values
        else:
            f = sensor.data.net_forces_w
            f_norm = torch.norm(f.reshape(f.shape[0], -1, 3), dim=-1)
            f_val = torch.max(f_norm, dim=1).values

        per_sensor_force.append(f_val)

    force_table = torch.stack(per_sensor_force, dim=1)
    return torch.any(force_table > thresh, dim=1)


def base_tilt(env, asset_cfg: SceneEntityCfg, limit: float = 0.6) -> torch.Tensor:
    """Terminate when base roll/pitch exceeds the configured limit."""

    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    tilt = torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)
    return tilt > torch.sin(torch.tensor(limit, device=tilt.device))


def base_height_low(env, asset_cfg: SceneEntityCfg, z_min: float = 0.25) -> torch.Tensor:
    """Terminate when the base height drops below the allowed threshold."""

    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < z_min
