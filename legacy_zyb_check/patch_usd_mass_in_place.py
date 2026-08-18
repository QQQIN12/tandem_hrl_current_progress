"""Patch wheel-foot mass attributes in an already-copied USD root layer."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--ix", type=float, default=0.00312)
parser.add_argument("--iy", type=float, default=0.00585)
parser.add_argument("--iz", type=float, default=0.00312)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pxr import Gf, Usd, UsdPhysics


def main() -> None:
    stage = Usd.Stage.Open(args.usd)
    if not stage:
        raise RuntimeError(f"failed to open {args.usd}")
    stage.Load()
    inertia = Gf.Vec3f(args.ix, args.iy, args.iz)
    coms = {
        "FL_foot": Gf.Vec3f(0.00064, 0.04319, 0.00079),
        "RL_foot": Gf.Vec3f(0.00064, 0.04319, 0.00079),
        "FR_foot": Gf.Vec3f(0.00062, -0.04262, 0.00043),
        "RR_foot": Gf.Vec3f(0.00062, -0.04262, 0.00043),
    }
    for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        prim = stage.GetPrimAtPath("/quadruped_arm/" + name)
        api = UsdPhysics.MassAPI(prim)
        if not api:
            api = UsdPhysics.MassAPI.Apply(prim)
        api.GetMassAttr().Set(0.918)
        api.GetDiagonalInertiaAttr().Set(inertia)
        api.GetCenterOfMassAttr().Set(coms[name])
    stage.GetRootLayer().Save()
    print("saved", args.usd, "root_layer", stage.GetRootLayer().identifier, flush=True)


try:
    main()
finally:
    simulation_app.close()
