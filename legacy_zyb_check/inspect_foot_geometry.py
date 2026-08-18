"""Inspect the authored foot/wheel geometry and mass properties in the shared USD."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics


def _attr(prim, name):
    value = prim.GetAttribute(name)
    if not value or not value.IsValid():
        return None
    try:
        return value.Get()
    except Exception as exc:
        return f"<error {exc!r}>"


def _rel(prim, name):
    rel = prim.GetRelationship(name)
    if not rel or not rel.IsValid():
        return []
    return [str(path) for path in rel.GetTargets()]


def main() -> None:
    stage = Usd.Stage.Open(args.usd)
    if not stage:
        raise RuntimeError(f"failed to open {args.usd}")
    stage.Load()
    print("stage", stage.GetRootLayer().identifier, flush=True)
    print("default", stage.GetDefaultPrim().GetPath(), flush=True)
    print("root_children", [str(child.GetPath()) for child in stage.GetPseudoRoot().GetChildren()], flush=True)
    for path in ("/quadruped_arm", "/quadruped_arm/FL_foot", "/quadruped_arm/joints/FL_foot_wheel_joint",
                 "/colliders/FL_foot", "/colliders/FL_foot/FL_foot"):
        prim = stage.GetPrimAtPath(path)
        print("target", path, "valid", prim.IsValid(), "type", prim.GetTypeName(),
              "children", [str(child.GetPath()) for child in prim.GetChildren()], flush=True)
    for name in ("FL", "FR", "RL", "RR"):
        prim = stage.GetPrimAtPath(f"/quadruped_arm/joints/{name}_foot_wheel_joint")
        print(
            "joint", name,
            "axis", _attr(prim, "physics:axis"),
            "body0", _rel(prim, "physics:body0"),
            "body1", _rel(prim, "physics:body1"),
            "localPos0", _attr(prim, "physics:localPos0"),
            "localPos1", _attr(prim, "physics:localPos1"),
            "localRot0", _attr(prim, "physics:localRot0"),
            "localRot1", _attr(prim, "physics:localRot1"),
            flush=True,
        )
    count = 0
    roots = [stage.GetPrimAtPath(path) for path in (
        "/quadruped_arm/FL_foot", "/quadruped_arm/FR_foot",
        "/quadruped_arm/RL_foot", "/quadruped_arm/RR_foot",
        "/colliders/FL_foot", "/colliders/FR_foot",
        "/colliders/RL_foot", "/colliders/RR_foot",
        "/meshes/FL_foot", "/meshes/FR_foot",
        "/meshes/RL_foot", "/meshes/RR_foot")]

    def walk(prim):
        if not prim or not prim.IsValid():
            return
        yield prim
        for child in prim.GetChildren():
            yield from walk(child)

    for root in roots:
        for prim in walk(root):
            path = str(prim.GetPath())
            if count > 200:
                break
            count += 1
            line = ["prim", path, "type", prim.GetTypeName(), "active", prim.IsActive()]
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get() or []
                line += ["points", len(points), "local_extent", _attr(prim, "extent")]
            line += [
                "radius", _attr(prim, "radius"),
                "height", _attr(prim, "height"),
                "size", _attr(prim, "size"),
                "translate", _attr(prim, "xformOp:translate"),
                "scale", _attr(prim, "xformOp:scale"),
                "rotateXYZ", _attr(prim, "xformOp:rotateXYZ"),
                "orient", _attr(prim, "xformOp:orient"),
                "xformOrder", _attr(prim, "xformOpOrder"),
                "body0", _rel(prim, "physics:body0"),
                "body1", _rel(prim, "physics:body1"),
            ]
            print(*line, flush=True)
    print("matched_prim_count", count, flush=True)
    for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        prim = stage.GetPrimAtPath("/quadruped_arm/" + name)
        mass_api = UsdPhysics.MassAPI(prim)
        print(
            "mass", name,
            "mass", mass_api.GetMassAttr().Get(),
            "diag_inertia", mass_api.GetDiagonalInertiaAttr().Get(),
            "com", mass_api.GetCenterOfMassAttr().Get(),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
