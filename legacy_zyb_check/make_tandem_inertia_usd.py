"""Create a TANDEM-only USD overlay with physically plausible wheel-foot inertia."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--ix", type=float, default=0.0032)
parser.add_argument("--iy", type=float, default=0.0059)
parser.add_argument("--iz", type=float, default=0.0032)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pxr import Gf, Usd, UsdPhysics


def main() -> None:
    stage = Usd.Stage.Open(args.input)
    if not stage:
        raise RuntimeError(f"failed to open {args.input}")
    stage.Load()
    inertia = Gf.Vec3f(args.ix, args.iy, args.iz)
    com = Gf.Vec3f(0.0, 0.0, 0.0)
    for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        prim = stage.GetPrimAtPath("/quadruped_arm/" + name)
        if not prim.IsValid():
            raise RuntimeError(f"missing foot prim {prim.GetPath()}")
        api = UsdPhysics.MassAPI(prim)
        if not api:
            api = UsdPhysics.MassAPI.Apply(prim)
        api.GetMassAttr().Set(0.918)
        api.GetDiagonalInertiaAttr().Set(inertia)
        api.GetCenterOfMassAttr().Set(com)
        print("patched", prim.GetPath(), "mass", api.GetMassAttr().Get(),
              "inertia", api.GetDiagonalInertiaAttr().Get(),
              "com", api.GetCenterOfMassAttr().Get(), flush=True)
    stage.Export(args.output)
    print("exported", args.output, flush=True)


try:
    main()
finally:
    simulation_app.close()
