# ZYB-v0 real-grasp extension

This repository keeps the original `ZYB-v0` task unchanged and adds a separate
physical grasp task for the B2W + Piper arm + two-finger gripper platform.

## Added task IDs

- `ZYB-Real-Grasp-Scene-v0`: 16-D ZYB-v0 leg/wheel action contract. It is used
  for checkpoint-compatible scene and contact replay.
- `ZYB-Real-Grasp-v0`: 24-D trainable interface with 12 leg, 4 wheel, 6 arm,
  and 2 gripper actions.
- `ZYB-Real-Grasp-Play-v0`: single-environment visualization variant.

The scene contains a dynamic rigid object, source and target platforms, finger
contact sensors, and all four wheel-ground contact sensors. The gripper action
uses bilateral finger contact to retain a payload until the policy explicitly
requests release.

## Physical changes

| Item | ZYB-v0 | Real-grasp task |
| --- | ---: | ---: |
| Gripper stiffness | 17 | 85 |
| Gripper damping | 0.02 | 4.5 |
| Wheel contact material | not explicit | 1.00 / 0.85 |
| Finger contact material | not explicit | 0.90 / 0.70 (simulation) |
| Object mass | - | 0.044 kg |
| Object contact material | - | 1.36 / 1.02 |

Leg, wheel, and arm actuator parameters are checked at environment startup and
must match ZYB-v0. The finger material is a simulation value and is not marked
as a hardware calibration.

## Checks

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$PWD/source/quadruped_arm:$PYTHONPATH

python scripts/zyb_real_grasp/probe_real_grasp_scene.py \
  --task ZYB-Real-Grasp-Scene-v0 --num_envs 16 --steps 240 --headless

python scripts/zyb_real_grasp/probe_grasp_calibration.py \
  --num_envs 1 --hold_steps 180 \
  --closed_loop_alignment --retention_latch \
  --event_gated_progress --iterative_ingress \
  --output logs/real_grasp_validation/grasp_calibration --headless

python scripts/zyb_real_grasp/evaluate_checkpoint_support.py \
  --checkpoint /path/to/model_1023.pt --steps 300 --headless
```

Both probes print one JSON record. The scene probe includes tilt, base height,
rear-leg support loss, object drift, and the applied actuator contract. The
contact probe reports bilateral contact, lift, payload retention, and support.
