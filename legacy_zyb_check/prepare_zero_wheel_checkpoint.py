import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--dest", type=Path, required=True)
args = parser.parse_args()

payload = torch.load(args.source, map_location="cpu")
state = payload["model_state_dict"]
for key in ("actor.4.weight", "actor.4.bias"):
    if key not in state:
        raise KeyError(key)
state["actor.4.weight"][12:16].zero_()
state["actor.4.bias"][12:16].zero_()
if "std" in state and state["std"].numel() >= 16:
    state["std"][12:16].fill_(0.08)
payload["model_state_dict"] = state
payload["infos"] = dict(payload.get("infos") or {})
payload["infos"]["codex_zeroed_wheel_head"] = True
args.dest.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, args.dest)
print("saved", args.dest)
print("wheel_bias", state["actor.4.bias"][12:16].tolist())
print("wheel_std", state["std"][12:16].tolist() if "std" in state else None)
