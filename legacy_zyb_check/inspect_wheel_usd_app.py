"""Inspect wheel joints and rigid-body mass properties with Isaac Sim's USD runtime."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import omni.usd
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def attr_value(prim, name):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.IsValid() else None


def rel_targets(prim, name):
    rel = prim.GetRelationship(name)
    if not rel or not rel.IsValid():
        return []
    return [str(path) for path in rel.GetTargets()]


def main():
    context = omni.usd.get_context()
    context.open_stage(args.usd)
    for _ in range(20):
        simulation_app.update()
    stage = context.get_stage()
    print("stage=", stage.GetRootLayer().identifier, flush=True)
    print("default_prim=", stage.GetDefaultPrim().GetPath(), flush=True)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render_, UsdGeom.Tokens.proxy_],
    )
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        typ = str(prim.GetTypeName())
        if "wheel" in name or "foot" in name or "joint" in name or typ in {"PhysicsRevoluteJoint", "PhysicsFixedJoint"}:
            print(
                "prim=", prim.GetPath(),
                "type=", typ,
                "body0=", rel_targets(prim, "physics:body0"),
                "body1=", rel_targets(prim, "physics:body1"),
                "axis=", attr_value(prim, "physics:axis"),
                "joint_pos=", attr_value(prim, "state:angular:physics:position"),
                "joint_vel=", attr_value(prim, "state:angular:physics:velocity"),
                flush=True,
            )
            for attr in prim.GetAttributes():
                attr_name = attr.GetName().lower()
                if any(key in attr_name for key in ("stiffness", "damping", "maxforce", "velocity", "drive", "limit")):
                    print("  attr=", attr.GetName(), "value=", attr.Get(), flush=True)
        if "wheel" in name or "foot" in name:
            if prim.IsA(UsdGeom.Gprim):
                try:
                    bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    print("  bbox_min=", bound.GetMin(), "bbox_max=", bound.GetMax(), flush=True)
                except Exception as exc:
                    print("  bbox_error=", repr(exc), flush=True)
            mass_api = UsdPhysics.MassAPI(prim)
            if mass_api:
                mass = mass_api.GetMassAttr().Get()
                inertia = mass_api.GetDiagonalInertiaAttr().Get()
                center = mass_api.GetCenterOfMassAttr().Get()
                if mass is not None or inertia is not None or center is not None:
                    print("  mass=", mass, "diag_inertia=", inertia, "com=", center, flush=True)
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if rigid_api:
                print("  rigid_body=", True, "kinematic=", rigid_api.GetKinematicEnabledAttr().Get(), flush=True)
    simulation_app.close()


try:
    main()
finally:
    if simulation_app.is_running():
        simulation_app.close()
