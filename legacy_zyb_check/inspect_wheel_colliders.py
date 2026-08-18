"""Inspect the authored foot/wheel collision geometry without opening a scene."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import omni.usd
from pxr import UsdGeom


USD_PATH = "/root/gpufree-data/tandem_hrl_620d798/source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3.usd"


def _value(prim, name):
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        return None
    try:
        return attr.Get()
    except Exception:
        return "<unreadable>"


def main() -> None:
    context = omni.usd.get_context()
    context.open_stage(USD_PATH)
    for _ in range(120):
        simulation_app.update()
    stage = context.get_stage()
    print("stage", stage.GetRootLayer().identifier, "default", stage.GetDefaultPrim().GetPath(), flush=True)
    prim_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        path = str(prim.GetPath())
        if not any(token in prim.GetName().lower() for token in ("wheel", "foot")):
            continue
        interesting = {}
        for name in (
            "radius", "height", "size", "extent", "points", "physics:approximation",
            "physics:collisionEnabled", "physics:rigidBodyEnabled", "physics:kinematicEnabled",
            "physics:mass", "physics:diagonalInertia", "xformOp:translate", "xformOp:scale",
        ):
            value = _value(prim, name)
            if value is not None:
                if name == "points":
                    value = f"<{len(value)} points>"
                interesting[name] = value
        relationships = {}
        for rel_name in prim.GetRelationshipNames():
            if "material" in rel_name.lower() or "physics" in rel_name.lower():
                relationships[rel_name] = [str(target) for target in prim.GetRelationship(rel_name).GetTargets()]
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if points:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                zs = [float(point[2]) for point in points]
                interesting["point_bounds"] = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
        print("prim", path, "type", prim.GetTypeName(), "attrs", interesting, "rels", relationships, flush=True)
    print("prim_count", prim_count, flush=True)


try:
    main()
finally:
    simulation_app.close()
