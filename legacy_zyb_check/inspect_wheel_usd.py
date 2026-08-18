from pxr import Usd

path = "/root/gpufree-data/tandem_hrl_620d798/source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3.usd"
stage = Usd.Stage.Open(path)
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if "wheel" not in name and "foot" not in name:
        continue
    attrs = {}
    for attr_name in ("axis", "physics:axis", "physics:body0", "physics:body1", "physics:jointEnabled"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            try:
                attrs[attr_name] = attr.Get()
            except Exception as exc:
                attrs[attr_name] = str(exc)
    rels = {}
    for rel_name in ("physics:body0", "physics:body1"):
        rel = prim.GetRelationship(rel_name)
        if rel and rel.IsValid():
            rels[rel_name] = [str(x) for x in rel.GetTargets()]
    print(str(prim.GetPath()), prim.GetTypeName(), attrs, rels)
