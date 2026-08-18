from __future__ import annotations

import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument("paths", nargs="+")
args = parser.parse_args()

for path in args.paths:
    print("===", path, "===", flush=True)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print("torch_load_error", type(exc).__name__, str(exc), flush=True)
        payload = None
    if isinstance(payload, dict):
        print("checkpoint_keys", list(payload.keys()), flush=True)
        for group_name in ("actor_state_dict", "model_state_dict", "state_dict"):
            state = payload.get(group_name)
            if not isinstance(state, dict):
                continue
            print("group", group_name, flush=True)
            for name, value in state.items():
                if hasattr(value, "shape"):
                    print("tensor", name, tuple(value.shape), flush=True)
        continue
    try:
        module = torch.jit.load(path, map_location="cpu")
        print("torchscript_loaded", flush=True)
        print("torchscript_code", module.code, flush=True)
        for name, child in module.named_children():
            print("child", name, "code", getattr(child, "code", "<no code>"), flush=True)
        for name, value in module.named_parameters():
            print("parameter", name, tuple(value.shape), flush=True)
    except Exception as exc:
        print("torchscript_error", type(exc).__name__, str(exc), flush=True)
