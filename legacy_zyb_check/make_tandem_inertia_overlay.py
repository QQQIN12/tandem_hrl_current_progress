"""Create a small USD sublayer that overrides only TANDEM wheel-foot inertia."""

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

from pxr import Gf, Sdf, Usd, UsdPhysics


def main() -> None:
    layer = Sdf.Layer.CreateNew(args.output)
    layer.subLayerPaths.append(args.input)
    stage = Usd.Stage.Open(layer)
    if not stage:
        raise RuntimeError("failed to open overlay stage")
    stage.Load()
    inertia = Gf.Vec3f(args.ix, args.iy, args.iz)
    com = Gf.Vec3f(0.0, 0.0, 0.0)
    for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        prim = stage.OverridePrim("/quadruped_arm/" + name)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.GetMassAttr().Set(0.918)
        api.GetDiagonalInertiaAttr().Set(inertia)
        api.GetCenterOfMassAttr().Set(com)
    default = stage.GetPrimAtPath("/quadruped_arm")
    if default.IsValid():
        stage.SetDefaultPrim(default)
    stage.GetRootLayer().Save()
    print("overlay", args.output, "size_bytes", stage.GetRootLayer().GetSize(),
          "default", stage.GetDefaultPrim().GetPath(), flush=True)


try:
    main()
finally:
    simulation_app.close()
