# tasks/manager_based/maniploco/mdp/utils.py
import torch
import math

# ====================== tcp common =========================
TCP_POS_OFFSET = (0.0, 0.0, 0.13)
TCP_QUAT_OFFSET = (1.0, 0.0, 0.0, 0.0)


# ======================= utils =============================
def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_apply(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_wxyz.unbind(-1)
    vx, vy, vz = v.unbind(-1)

    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)

    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([vpx, vpy, vpz], dim=-1)


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), 0.0 * half, 0.0 * half, torch.sin(half)], dim=-1)


def _quat_from_euler_xyz(roll, pitch, yaw):
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _sphere2cart(sph: torch.Tensor) -> torch.Tensor:
    l = sph[:, 0]
    p = sph[:, 1]
    y = sph[:, 2]
    x = l * torch.cos(p) * torch.cos(y)
    yy = l * torch.cos(p) * torch.sin(y)
    z = l * torch.sin(p)
    return torch.stack([x, yy, z], dim=-1)


def _skew_batch(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v.unbind(-1)
    zero = torch.zeros_like(vx)
    return torch.stack(
        [
            torch.stack([zero, -vz,   vy], dim=-1),
            torch.stack([  vz, zero, -vx], dim=-1),
            torch.stack([ -vy,  vx, zero], dim=-1),
        ],
        dim=1,
    )


def orientation_error(q_des_wxyz: torch.Tensor, q_cur_wxyz: torch.Tensor) -> torch.Tensor:
    q_des_wxyz = _normalize_quat(q_des_wxyz)
    q_cur_wxyz = _normalize_quat(q_cur_wxyz)

    w1, x1, y1, z1 = q_des_wxyz.unbind(-1)
    w2, x2, y2, z2 = q_cur_wxyz.unbind(-1)

    w =  w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    x = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    y = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    z = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2

    sign = torch.where(w.unsqueeze(-1) < 0.0, -1.0, 1.0)
    return 2.0 * sign * torch.stack([x, y, z], dim=-1)


def _quat_from_rotmat(R: torch.Tensor) -> torch.Tensor:
    N = R.shape[0]
    q = torch.zeros((N, 4), device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    mask = trace > 0.0
    if mask.any():
        t = torch.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * t
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / t
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / t
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / t

    mask1 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask1.any():
        t = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2.0
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / t
        q[mask1, 1] = 0.25 * t
        q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / t
        q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / t

    mask2 = (~mask) & (~mask1) & (R[:, 1, 1] > R[:, 2, 2])
    if mask2.any():
        t = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / t
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / t
        q[mask2, 2] = 0.25 * t
        q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / t

    mask3 = (~mask) & (~mask1) & (~mask2)
    if mask3.any():
        t = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2.0
        q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / t
        q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / t
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / t
        q[mask3, 3] = 0.25 * t

    return _normalize_quat(q)


def _quat_from_tool_z(tool_z_w: torch.Tensor, world_up: torch.Tensor | None = None) -> torch.Tensor:
    z_axis = _normalize(tool_z_w)

    if world_up is None:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=tool_z_w.device, dtype=tool_z_w.dtype).view(1, 3)
    up = world_up.expand_as(z_axis)

    bad = torch.abs((z_axis * up).sum(dim=-1, keepdim=True)) > 0.95
    alt_up = torch.tensor([1.0, 0.0, 0.0], device=tool_z_w.device, dtype=tool_z_w.dtype).view(1, 3).expand_as(z_axis)
    up = torch.where(bad, alt_up, up)

    y_axis = _normalize(torch.cross(z_axis, up, dim=-1))
    x_axis = _normalize(torch.cross(y_axis, z_axis, dim=-1))
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return _quat_from_rotmat(R)


def _quat_conj(q_wxyz: torch.Tensor) -> torch.Tensor:
    """求四元数的共轭"""
    w, x, y, z = q_wxyz.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def _quat_rotate_inverse(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """应用四元数到向量，并求其逆"""
    return _quat_apply(_quat_conj(q_wxyz), v)


def _quat_to_euler_xyz(q_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """wxyz 四元数 -> XYZ 欧拉角（roll, pitch, yaw）"""
    q = _normalize_quat(q_wxyz)
    w, x, y, z = q.unbind(-1)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2.0 * torch.pi) - torch.pi