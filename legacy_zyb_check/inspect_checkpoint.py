import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/root/gpufree-data/tandem_hrl_620d798/logs/rsl_rl/maniploco/2026-08-17_20-58-03/model_1122.pt")
args = parser.parse_args()
payload = torch.load(args.path, map_location="cpu")
print("keys", list(payload.keys()))
for group_name in ("model_state_dict", "state_dict"):
    state = payload.get(group_name)
    if not isinstance(state, dict):
        continue
    for name, value in state.items():
        if "actor" in name or "policy" in name or "std" in name:
            shape = tuple(value.shape) if hasattr(value, "shape") else None
            print(name, shape, "mean", float(value.float().mean()), "std", float(value.float().std()))
    if "actor.4.weight" in state and "actor.4.bias" in state:
        print("wheel_row_norms", state["actor.4.weight"][12:16].norm(dim=1).tolist())
        print("wheel_bias", state["actor.4.bias"][12:16].tolist())
