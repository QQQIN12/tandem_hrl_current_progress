# TANDEM-HRL lower-body debug record

This index records the parameters, remote log paths, and current interpretation.
The complete stdout/stderr for simulator runs is retained on the server under
`/root/gpufree-data/legacy_zyb_check/`; this file is an index rather than a
replacement for those logs.

## Environment and asset

- Remote host: `120.209.70.195:30346`, repository:
  `/root/gpufree-data/tandem_hrl_620d798`.
- Isaac Python: `/workspace/isaaclab/_isaac_sim/python.sh`.
- GPU: NVIDIA L40; Isaac Sim 5.1; Isaac Lab 2.3.2.
- Active TANDEM asset:
  `source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3_tandem_inertia.usd`.
- Original shared asset remains at:
  `source/quadruped_arm/quadruped_arm/robots/assets/quadruped_arm_V3.usd`.
- Wheel-foot mass: `0.918 kg`; active diagonal inertia:
  `(0.00312, 0.00585, 0.00312) kg m^2`; wheel axis is body `Y`.
- Active wheel direction signs: `(1, 1, -1, -1)` in `(FL, FR, RL, RR)`
  action order; runtime all-plus tests are explicitly labeled below.
- Active lower-body posture feedback: DLS contact correction disabled after
  corrected-order tests; nominal leg position targets remain active.
- Active arm IK: frozen with `max_joint_delta=0`; active wheel actuator damping
  in `robot_cfg.py` remains `0.5` unless a probe explicitly overrides it.
- No training, evaluation, or probe process was left running at the start of
  this record.

## Existing baseline and inertia/sign tests

1. Archived checkpoint `/root/gpufree-data/zyb_v0_model_1023.pt`, old arm IK,
   corrected inertia, implicit wheels, 24 environments, 4 episodes per command:
   log/result `eval_inertia_signplus.json`.
   - Stable rate: `1.0` for forward, reverse, both yaw directions, arc, and
     stationary.
   - Mean body velocity: forward `0.011 m/s`, reverse `-0.013 m/s`, yaw left
     `0.010 rad/s`, yaw right `-0.002 rad/s`.
   - Tracking remained unsuccessful.

2. Same checkpoint and asset with wheel stiffness `5.0`, damping `0.5`:
   `eval_inertia_signplus_stiff5.json`.
   - Forward/reverse still did not track; yaw-left stable rate fell to `0.75`.
   - Do not use nonzero wheel stiffness with the current velocity-action path.

3. Static wheel-sign comparison is retained in the earlier logs
   `multi_teacher_inertia_gate10.log` and
   `multi_teacher_inertia_gate_signplus10.log`.
   - All `+1` changed forward body velocity from about `-0.040` to `+0.026`
     m/s and reduced forward error from about `0.398` to `0.331` m/s.

## Upper-IK ablation

4. Current IK (`damping=0.05`), zero wheel command, root reset height `0.58 m`,
   90 steps: `static_drive_probe` baseline log.
   - Settled root height about `0.37 m`; maximum tilt about `0.29 rad`.

5. Same test with temporary `arm_ik_damping=100.0`:
   `static_armdamp100.log`.
   - Root height about `0.452 m`; maximum tilt `0.11077 rad`.
   - No tilt, low-height, or bad-contact termination.

6. Same test with the new safe-IK configuration (`max_joint_delta=0`), zero
   wheel command: `static_safeik_freeze.log`.
   - Root height about `0.452 m`; maximum tilt `0.11081 rad`.
   - This reproduces the frozen-arm ablation without relying on an extreme
     temporary damping value.

7. Frozen arm plus forward command (`0.35 m/s`), 45-step zero-command warmup
   followed by 45 drive steps, wheel damping `0.5`:
   `static_safeik_drive_damp05.log`.
   - Final body `vx` `0.0253 m/s`; maximum tilt `0.28524 rad`.
   - Therefore wheel/leg compensation remains an independent problem.

8. Same frozen-arm drive test with wheel damping `2.0`:
   `static_armfreeze_drive_damp2.log`.
   - Final body `vx` `0.1776 m/s`; maximum tilt `0.29037 rad`.
   - Higher velocity damping improves wheel speed/translation but increases
     the need for learned leg stabilization; it is not yet accepted as the
     final training value.

9. Completely removed `arm_ik` from the action configuration, zero wheel
   command, root reset height `0.58 m`, 90 steps: `static_noarmik.log`.
   - Result is numerically identical to the safe frozen-arm test: root height
     about `0.452 m`, maximum tilt `0.11081 rad`.
   - This isolates the disturbance to the old IK action path rather than the
     mere presence of the arm rigid bodies.

10. Lower-body multi-teacher run with safe frozen arm, 64 environments, 50
    requested iterations, action std `0.02`, learning rate `1e-5`, teacher
    blend `0.90 -> 0.50` over 50,000 environment steps:
    `train_safeik_freeze_50.log`.
    - At 19,968 steps: tilt/low-height/bad-contact all `0`, velocity errors
      `0.2360` and `0.3126`.
    - At 66,048 steps: all three safety termination rates remained `0`, but
      errors had degraded to `0.4718` and `0.5567`.
    - Saved checkpoints include `model_1025.pt`, `model_1050.pt`, and
      `model_1072.pt`; none is accepted as final until fixed-command rollout
      evaluation succeeds.

11. High leg-PD stiffness scan (`300` then `500`, damping `10`) timed out while
   the first `300` run was still in simulator startup; it was terminated by
   exact process ID. Logs: `leg_stiff_300.log`; no `500` result was produced.
   This scan is inconclusive and is not evidence that high gains are safe.

## Current code change

- `mdp/actions.py`: IK target increments are rate-limited by
  `max_joint_delta`, and reset-aware target state is maintained.
- `maniploco_env_cfg.py`: lower-body stage uses `damping=0.5` and
  `max_joint_delta=0.0` to hold the arm at reset while the chassis is trained.
- Both files passed Isaac Python byte-compilation before the next training run.

12. Early-return frozen-IK implementation, zero wheel command, root reset
    height `0.58 m`, 90 steps: `static_safeik_freeze_earlyreturn.log`.
    - The implementation now initializes the reset-aware arm target and
      returns before querying the command manager, end-effector pose, or
      Jacobian when `max_joint_delta=0`.
    - Result: mean body `vx=-0.04267 m/s`, mean `wz=0.00753 rad/s`, final
      `z=0.45166 m`, minimum `z=0.42186 m`, maximum tilt `0.11081 rad`.
    - Tilt, low-height, and bad-contact terms were all false.  The result is
      consistent with the earlier safe-freeze/no-IK ablation, while removing
      unnecessary upper-IK computation.
    - Exact remote log: `/root/gpufree-data/legacy_zyb_check/static_safeik_freeze_earlyreturn.log`.

13. User-provided prior policies were copied to the data disk and inspected:
    `/root/gpufree-data/legacy_zyb_check/policy.pt` and
    `/root/gpufree-data/legacy_zyb_check/model_2999.pt`.
    - `policy.pt` is TorchScript with actor input width `57` and output width
      `16`.  The current TANDEM `maniploco` environment produced policy
      observation shape `(1, 876)` at reset.  Actual forward call failed with
      `mat1 and mat2 shapes cannot be multiplied (1x876 and 57x512)`; see
      `probe_policy_pt.log`.
    - `model_2999.pt` is an RSL-RL checkpoint with actor input width `210` and
      output width `18`; the current environment has input width `876` and
      lower-body action width `16`.  Its dimensions therefore do not match the
      current action/observation contract even before a rollout.
    - These files are useful references, but are not directly usable current
      TANDEM policies without an explicit observation/action adapter and a
      compatible actuator configuration.

14. Source-contract comparison used for the upper-layer hypothesis:
    - The archived `quadruped_arm_LR_HRL` `maniploco` config uses ordinary
      `JointVelocityActionCfg(scale=0.1)` for four wheels and the original
      `ArmIkFromEeGoalActionCfg(damping=0.05)`.
    - The active lower-stage config uses the bounded command-conditioned wheel
      action, corrected wheel inertia asset, and `ArmIkFromEeGoalActionCfg`
      with `max_joint_delta=0.0`.
    - The TACTIC-HRL config explicitly sets inherited `arm_ik=None` and uses a
      learned `arm_policy` term.  This is evidence of an interface change, not
      proof that TACTIC itself caused the observed collapse; the lower-stage
      wheel/leg coupling still needs a separate test.
    - The original EE command samples a sphere radius `0.5--0.7 m`, pitch
      `-pi/4--0`, with roughly `1--3 s` trajectories and `0.5--2 s` holds.
      Thus an enabled old IK term can generate sustained arm motion even when
      the locomotion command is zero.

15. Archived learned leg teacher with frozen upper body and the active safe
    wheel action, fixed forward command `0.35 m/s`, one 300-step episode:
    `eval_teacher_forward_safeik.log` / `eval_teacher_forward_safeik.json`.
    - Environment used the corrected TANDEM inertia asset, `max_joint_delta=0`
      and implicit wheel damping `0.5` with zero wheel stiffness.
    - Termination-based stable rate was `1.0`, but tracking pass rate was
      `0.0`; mean body `vx=0.05265 m/s`, mean linear error `0.31738 m/s`.
    - Debug samples showed the feed-forward target near `3.2 rad/s`, while
      wheel velocities varied from roughly `-2.18` to `6.08 rad/s`; the
      controller did not realize the requested wheel speed reliably.
    - This is not evidence that the leg teacher is useless: it is evidence
      that replaying its leg actions with the current wheel actuator does not
      reproduce the old closed loop.  The wheel actuator/target semantics must
      be calibrated before further PPO conclusions.

16. Same teacher/asset/upper freeze and forward command with only implicit
    wheel damping changed from `0.5` to `2.0`:
    `eval_teacher_forward_safeik_damp2.log` / `.json`.
    - Mean body `vx` increased only from `0.05265` to `0.05936 m/s`; tracking
      remained failed (`tracking_pass_rate=0`).
    - Early tilt increased to about `0.411 rad` and root height fell to about
      `0.35--0.40 m` before the episode timeout reset.  Increasing damping
      alone is therefore not the missing lower-body controller.

17. First zero-policy-wheel-residual forward rerun was inconclusive:
    `eval_teacher_forward_zeroresid.log`.
    - The evaluator still had command/EE debug visualization enabled. IsaacSim
      scene creation took `282.24 s`; the remote `300 s` timeout expired before
      policy rollout, so no control metric is recorded from this run.
    - The evaluator was then patched to disable both command debug visuals and
      use a large render interval. This is an infrastructure correction, not a
      control result.

18. Direct teacher short probe `teacher_probe_forward_zeroresid_30.log` was
    also inconclusive as a runtime measurement: IsaacSim scene creation took
    `144.28 s` and the `180 s` timeout expired before the first printed step.
    No policy metric from this run is used.  The large variation in scene
    creation time (`2.87 s`, `144.28 s`, and `282.24 s` across otherwise
    equivalent launches) is recorded as a server/IsaacSim startup issue and is
    separate from the controller diagnosis.

19. New `StabilizedLegPositionAction` with bounded residual `0.22 rad` and
    contact-foot DLS posture feedback (`authority=0.75`, base target
    `0.46 m`), zero policy leg action, forward command `0.35 m/s` after a
    45-step zero-command warmup, 90 steps:
    `static_stabilizedleg_forward_90.log`.
    - Minimum height `0.42727 m`, maximum tilt `0.11464 rad`, and all tilt,
      low-height, and bad-contact terms were false.
    - Mean body `vx=-0.04626 m/s`; the chassis remained stable but did not yet
      track forward motion.  This is an intended intermediate result: the
      new recovery layer suppresses the previous transient collapse, while
      motion generation still has to be learned.
    - Zero-command comparison with the same action term gave minimum height
      `0.42727 m` and maximum tilt `0.11464 rad` in
      `static_stabilizedleg_zero_30.log`.

20. Same stabilized-leg action with wheel actuator damping raised to `2.0`,
    forward `0.35 m/s`, 45-step warmup plus 45 drive steps:
    `static_stabilizedleg_forward_damp2_90.log`.
    - Mean `vx=-0.01893 m/s`, minimum height `0.36700 m`, maximum tilt
      `0.28835 rad`; all termination flags stayed false.
    - The stronger wheel drive recovered some instantaneous speed
      (`vx=0.1604 m/s` at step 89) but reintroduced the transient tilt/height
      loss. Keep the current `0.5` damping for the stability-first training
      stage; do not promote `2.0` based on speed alone.

21. Stabilized-leg teacher-student smoke run:
    `train_stabilizedleg_freeze_smoke.log`, 64 environments, requested 20
    iterations, old checkpoint initialization, action std `0.02`, learning
    rate `1e-5`, teacher blend `0.95 -> 0.85` over `100000` environment steps.
    - IsaacSim scene creation took `159.07 s`; the run completed and wrote
      `model_1030.pt`, `model_1040.pt`, and `model_1042.pt` under
      `/root/gpufree-data/tandem_hrl_620d798/logs/rsl_rl/maniploco/2026-08-18_01-51-29`.
    - Safety termination rates remained zero, but later velocity errors were
      approximately `error_vel_xy=0.4298` and `error_vel_yaw=0.5690`. This is
      not an accepted checkpoint; fixed-command rollout is required.

22. Fixed forward rollout of `model_1042.pt` from that smoke run, 60 steps,
    fixed reset, with the new stabilized-leg executor:
    `probe_model1042_forward.log` / `.json`.
    - Mean body `vx=0.00513 m/s`, minimum height `0.26686 m`, maximum tilt
      `0.39040 rad`; termination flags happened not to fire.
    - This checkpoint is rejected. The zero-action static probe was stable, so
      the result indicates that inheriting the old actor under the changed
      wheel/leg executor produces destabilizing leg commands. The next fresh
      student run must not resume this actor as if its action semantics were
      unchanged.

23. Fresh neutral-teacher run without checkpoint resume:
    `train_fresh_neutral_stabilizedleg.log`, 64 environments, requested 30
    iterations, action std `0.05`, learning rate `5e-5`, teacher blend
    `1.0 -> 0.70` over `100000` steps, with no teacher checkpoint so all
    teacher candidates were neutral zero actions.
    - The run reached `model_0.pt`, `model_10.pt`, and `model_20.pt`.
    - At the latter logged stage, safety terms were zero but velocity errors
      were `error_vel_xy=0.4093` and `error_vel_yaw=0.5544`; no checkpoint is
      accepted yet.
    - The remote timeout orphaned exact training PID `208267` after the
      `model_20` save; it was explicitly terminated and verified absent. This
      is runtime cleanup, not a training result.

24. Archived `policy.pt` TorchScript adapter probe, using the current
    stabilized-leg executor, frozen arm, corrected wheel inertia, and safe
    wheel action.  The adapter built the 57-D input from current runtime state:
    base angular velocity, projected gravity, command, 12 leg + 4 wheel joint
    positions, corresponding velocities, and the last 16 actions.  This is an
    empirical compatibility probe, not a claim that this was the original
    training layout.
    - Command: `ZYB-v0`, 4 environments, 120 steps, forward command
      `(0.35, 0.0, 0.0)`, `policy.pt`, current action width 16.
    - `common_scaled` layout result:
      `/root/gpufree-data/legacy_zyb_check/policy_common_scaled_forward_v2.json`
      and local copy `policy_common_scaled_forward_v2.json`.
      Input/output were `(4,57)/(4,16)`; no tilt, low-height, or bad-contact
      termination occurred.  Mean body `vx=0.04262 m/s`, mean body
      `wz=0.03656 rad/s`, minimum height mean `0.30401 m` and minimum across
      environments `0.28318 m`; maximum tilt mean `0.26245 rad`, maximum
      across environments `0.27413 rad`; action absolute mean `4.25262`, max
      `16.49816`.  The 0.35 m/s command was therefore not tracked.
    - Scene creation for this repeat was `138.945586 s`, simulation start
      `0.750834 s`.  The large startup cost is recorded separately from the
      controller result.
    - A first `common_scaled` run before the metrics patch had the same
      qualitative result: maximum tilt mean `0.27261 rad`, minimum height mean
      `0.31448 m`, and no termination flags.

25. The second `policy.pt` input-layout probe was inconclusive, not a control
    failure.  The `legacy_guess` run was launched with the same 4-env,
    120-step forward test and output path
    `/root/gpufree-data/legacy_zyb_check/policy_legacy_guess_forward_v2.json`.
    IsaacSim had not completed scene creation by the 300 s wrapper timeout;
    no rollout metric or JSON result was produced.  The detached Python PID
    `218406` remained after the connection timeout and was killed exactly with
    `kill -9 218406`; a follow-up process check showed no training or policy
    rollout process.  The earlier non-metrics `legacy_guess` run completed with
    no terminations, maximum tilt mean `0.24380 rad`, minimum height mean
    `0.31514 m`; it did not measure body velocity and is not used to select a
    layout.

26. Interpretation of the archived policy probe: `policy.pt` is not directly
    compatible with the current 876-D observation contract, and the tested
    57-D reconstruction does not produce a useful forward controller under
    the current action executor.  It may still be retained as a reference for
    action/observation semantics, but it is not an accepted lower-body
    checkpoint.  No current PPO checkpoint is promoted by this test.

27. Static forward drive with the stabilized-leg posture layer and wheel
    damping `1.0` (45 zero-command warmup + 45 drive steps, command
    `vx=0.35 m/s`): `static_stabilizedleg_forward_damp1_90.log`.
    - Scene creation took `121.831703 s`.
    - Minimum height `0.43277 m`, maximum tilt `0.11302 rad`, and all three
      safety termination flags were false.
    - Mean body `vx=0.01483 m/s`, mean `wz=0.00310 rad/s`; at step 89 the
      instantaneous `vx=0.08180 m/s`, wheel target was `3.15618 rad/s`, and
      measured wheel speeds were approximately
      `(1.0038, 0.8816, 2.3730, 2.2419) rad/s`.
    - Damping `1.0` is a better stability/speed compromise than the tested
      `0.5` and `2.0` settings for this fixed posture probe, but it still does
      not satisfy command tracking.  This motivates a body-velocity feedback
      term in the wheel reference; damping alone is not sufficient.

28. First body-velocity feedback trial, with gains `vx=0.35` and `wz=0.20`,
    wheel damping `1.0`, same 45-step warmup and 45-step forward segment:
    `static_stabilizedleg_forward_fb_damp1_90.log`.
    - Scene creation took `2.728009 s`.
    - Mean body `vx=-0.00158 m/s`, mean `wz=-0.00515 rad/s`; the result did
      not improve command tracking.
    - Minimum height fell to `0.37418 m` and maximum tilt rose to
      `0.28694 rad`; safety flags still happened to remain false.  At step 89
      the wheel target was about `3.48 rad/s`, while measured wheel speeds
      were uneven `(2.2666, 1.6441, 3.1799, 2.1711) rad/s`.
    - This gain is too aggressive for the present leg/contact controller.  It
      is not promoted; the experiment supports adding feedback only with a
      much smaller gain and/or a support-margin gate.

29. Explicit leg action-order fix (`preserve_order=True`) with wheel feedback
    disabled and damping `1.0`, forward command `0.35 m/s`, 45-step warmup plus
    45 drive steps: `static_stabilizedleg_orderfix_forward_damp1_90.log`.
    - Scene creation took `144.056723 s`.
    - The resolved target print now matches the declared policy order
      `[FR, FL, RR, RL]`; at reset it is approximately
      `FR=(-0.204,0.818,-1.534)`, `FL=(0.204,0.818,-1.534)`,
      `RR=(-0.204,1.021,-1.533)`, `RL=(0.204,1.021,-1.533)`.
    - Forward result: mean `vx=-0.04942 m/s`, minimum height `0.37057 m`,
      maximum tilt `0.26694 rad`; no termination flag fired.
    - This is worse than the previous accidental-order result
      (`min_z=0.43277 m`, `max_tilt=0.11302 rad`).  The order fix is retained
    as the correct interface contract, but it demonstrates that the DLS
      correction/gains must be retuned for the corrected physical mapping;
      it is not evidence that the previous order was acceptable.

30. Correct leg order with the DLS posture feedback disabled, wheel feedback
    disabled, damping `1.0`, forward command `0.35 m/s`, 45-step warmup plus
    45 drive steps: `static_orderfix_forward_nofb_damp1_90.log`.
    - Scene creation took `1.218433 s`.
    - Mean body `vx=-0.03413 m/s`; minimum height `0.36560 m`, maximum tilt
      `0.28698 rad`, and no safety termination flag fired.
    - Comparing this with the zero-command no-feedback result
      (`min_z=0.43062 m`, `max_tilt=0.11017 rad`) shows that the collapse is
      caused by the wheel-drive/contact transient, not by DLS alone.  The
      current fixed default leg posture is not a sufficient walking teacher.

31. Asset geometry and wheel actuator inspection:
    `inspect_foot_geometry.log` and `wheel_probe_nogravity.log`.
    - All four wheel joints use local axis `Y`, body0 is the corresponding
      calf, body1 is the corresponding foot, and both local joint positions
      are `(0,0,0)` on the foot side and `(0,0,-0.35)` on the calf side.
    - The foot body has an actual wheel-shaped collision mesh; it is not a
      missing-collider explanation.  The corrected active copy reports mass
      `0.918 kg`, diagonal inertia `(0.00312,0.00585,0.00312) kg m^2`, and
      finite COM values near the wheel axis (`y=+0.04319 m` on FL/RL and
      `y=-0.04262 m` on FR/RR).
    - In zero-gravity actuator checks, the current velocity actuator tracks
      positive and negative targets correctly: target `+0.1` produced wheel
      speeds about `+0.1001`, target `-0.1` produced about `-0.0999`, and the
    alternating yaw patterns also tracked their signs.  Thus the remaining
      forward problem is contact/body response, not a dead wheel actuator.

32. Direct wheel-sign comparison with correct leg order, no DLS feedback,
    command `vx=0.35 m/s`, damping `1.0`: the `wheel_sign=+1` run reached
    step-89 `vx=+0.08625 m/s`, while `wheel_sign=-1` reached
    `vx=-0.07999 m/s`.  The positive sign is therefore retained for forward
    command semantics; the poor mean tracking is not explained by choosing
    the opposite global sign.

33. TACTIC-inspired lower DLS gains (`base_height_target=0.36`,
    `base_height_gain=0.8`, `orientation_gain=0.7`) did not fix the corrected
    mapping: zero-command, 90-step result
    `static_orderfix_dls_tactic_zero_90.log` had mean `vx=-0.11604 m/s`,
    minimum height `0.36792 m`, and maximum tilt `0.26708 rad`.  This rejects
    the current DLS implementation as a stability layer for this action/
    Jacobian mapping; it should be disabled until independently corrected.

34. Active rear-sign convention command probes, with correct leg order,
    nominal legs (DLS disabled), wheel damping `1.0`:
    - Forward `vx=0.35`, 45-step warmup + 45 drive steps:
      `static_orderfix_forward_rearflip_nofb_damp1_90.log` gave mean
      `vx=0.02664 m/s`, step-89 `vx=0.12888 m/s`, minimum height `0.43062 m`,
      maximum tilt `0.11017 rad`.  This is the first stable forward response
      under the corrected sign pattern, but it still under-tracks.
    - Reverse `vx=-0.25`:
      `static_active_reverse_90.log` gave mean `vx=-0.09297 m/s`, step-89
      `vx=0.01668 m/s`, minimum height `0.37813 m`, maximum tilt `0.31066 rad`.
      Reverse is not yet acceptable.
    - Yaw left/right `wz=+/-0.45`:
      `static_active_yawleft_90.log` and `static_active_yawright_90.log` had
      mean yaw rates only `0.00438` and `0.00790 rad/s`, respectively, with
      no safety term but almost no turning.  The sign convention improves
      forward support but does not yet prove command tracking for reverse or
      yaw.

35. Rear-sign convention with wheel damping `2.0`, correct leg order, nominal
    legs, DLS disabled, forward command `vx=0.35 m/s`:
    `static_active_forward_damp2_90.log`.
    - The run completed with no safety termination flags.  Mean body
      `vx=0.01980 m/s`, mean `wz=-0.00054 rad/s`, minimum height `0.42328 m`,
      and maximum tilt `0.11098 rad`.
    - At step 89 the instantaneous `vx=0.25165 m/s`; wheel targets were
      approximately `(3.1239, 3.1239, -3.1239, -3.1239) rad/s`, measured
      wheel speeds `(2.4973, 2.6583, -2.9753, -2.5539) rad/s`.
    - This is a better forward transient than damping `1.0`, but it is only a
      fixed-posture probe and does not establish reverse/yaw tracking.  The
      damping override is not yet promoted to the active robot config.

36. Rear-sign convention with wheel damping `2.0`, yaw-left command
    `wz=+0.45 rad/s`, same 45-step warmup and 45-step drive segment:
    `static_active_yawleft_damp2_90_retry.log`.
    - Mean body `vx=-0.10335 m/s`, mean `wz=-0.02947 rad/s`, minimum height
      `0.37336 m`, maximum tilt `0.29335 rad`; no termination flag fired.
    - At step 89 the measured yaw rate was only `0.00273 rad/s`, with body
      tilt `0.29093 rad` and height `0.37535 m`.  This is a failed yaw
      tracking/stability case; damping `2.0` alone cannot be promoted as the
      complete lower-body solution.
    - The detached run from the first yaw attempt was exact PID `241168`; it
      was killed and a follow-up process check found no remaining probe or
      training process.  This cleanup fact is not a control result.

37. Ground-friction override probe was inconclusive:
    `static_active_forward_friction02_90.log`, command used friction `0.2`,
    damping `1.0`, rear-sign convention, forward `vx=0.35`, 45-step warmup
    and 45-step drive.  The log contains only IsaacSim/Kit startup output;
    it contains no scene-creation line, rollout metrics, or Python traceback,
    and no matching remote process remained.  Therefore no physics
    conclusion is drawn from this run; it must be repeated with explicit exit
    status capture before changing the terrain friction model.

38. Geometric track-width probe, rear-sign convention, correct leg order,
    DLS disabled, wheel damping `1.0`, yaw-left command `wz=+0.45`,
    track-width override `0.38346 m` (measured left/right wheel-center
    separation), 45-step warmup plus 45-step drive:
    `static_active_yawleft_track383_damp1_90.log`.
    - Mean body `vx=-0.04185 m/s`, mean `wz=0.00411 rad/s`, minimum height
      `0.43062 m`, maximum tilt `0.11017 rad`; no termination flag fired.
    - At step 89, the wheel target was
      `(-0.77758, +0.77758, +0.77758, -0.77758) rad/s`, but measured wheel
      speeds were approximately `(-0.461, +0.174, -0.445, +0.154) rad/s`,
      and body yaw rate was only `0.00636 rad/s`.
    - Reducing the geometric width removes the 30% over-command in the
      differential-drive formula but does not restore yaw tracking.  The
      original `0.50 m` width is therefore not the sole cause; turn authority
      and leg/contact coupling remain unresolved.

39. Higher leg PD trial for forward motion: leg stiffness `300`, damping
    `10`, wheel damping `1.0`, rear-sign convention, DLS disabled, forward
    `vx=0.35`, 45-step warmup plus 45-step drive:
    `static_active_forward_leg300_damp10_wheel1_90.log`.
    - Scene creation took `158.518889 s`; the rollout completed with no
      safety termination flags.
    - Mean body `vx=0.00991 m/s`, mean `wz=0.00159 rad/s`, minimum height
      `0.43163 m`, maximum tilt `0.10045 rad`; step-89 `vx=0.16636 m/s`.
    - Higher leg gains slightly reduce the fixed-probe tilt compared with the
      nominal `160/5` case, but do not produce reliable command tracking.  No
      active gain change is promoted from this single probe.

40. Higher-leg-gain yaw trial (`300/10`, wheel damping `1.0`, `wz=+0.45`)
    was inconclusive: `static_active_yawleft_leg300_damp10_wheel1_90.log`
    contains only Kit startup output, no scene-creation line, no rollout
    metrics, and no traceback; a process check found no remaining probe.  It
    is recorded as a runtime failure, not as evidence that the gain pair
    fails physically.

41. High wheel damping trial, rear-sign convention, nominal legs, DLS
    disabled, `wz=+/-0.45`, wheel damping `10.0`:
    `static_active_yawleft_wheelD10_90.log` and
    `static_active_yawright_wheelD10_90.log`.
    - Positive-yaw command: mean `vx=-0.10178`, mean `wz=-0.02893`, minimum
      height `0.35743 m`, maximum tilt `0.28768 rad`.
    - Negative-yaw command: mean `vx=-0.10531`, mean `wz=0.01482`, minimum
      height `0.35934 m`, maximum tilt `0.28948 rad`.
    - Increasing wheel damping increases contact torque but causes a large
      posture transient and still does not establish yaw tracking.  It is
      rejected as a global fix.

42. Slow wheel-target ramp trial, wheel damping `2.0`, maximum wheel
    acceleration `2.0 rad/s^2`, `wz=+0.45`, same warmup/drive horizon:
    `static_active_yawleft_damp2_accel2_90.log`.
    - The 300-second wrapper timed out; the log has only Kit startup output,
      no scene setup or metrics, and the process check found no remaining
      probe.  The run is inconclusive and cannot support a transient-control
      conclusion.

43. Symmetric negative-yaw rerun with wheel damping `2.0`, rear-sign
    convention, nominal legs, DLS disabled:
    `static_active_yawright_damp2_90_retry.log`.
    - For command `wz=-0.45`, mean body `vx=-0.11700 m/s`, mean
      `wz=-0.01289 rad/s`, minimum height `0.37026 m`, maximum tilt
      `0.29805 rad`; no safety termination flag fired.
    - At step 89, body `wz=-0.00207 rad/s` and height `0.37860 m`; wheel
      targets were approximately `(+0.861,-0.861,-0.861,+0.861) rad/s`.
    - The sign of the instantaneous yaw response is consistent with the
      requested command but its magnitude is negligible and the posture
      transient is unacceptable.  This is not evidence for simply swapping
      the yaw sign in the command formula.

44. Large track-width diagnostic (`1.0 m`, wheel damping `1.0`, rear-sign
    convention, `wz=+0.45`) did not reach scene setup:
    `static_active_yawleft_track100_damp1_90.log` contains IsaacLab startup
    output but no completed environment or rollout metrics.  The remote
    Python PID `253305` remained after the wrapper timeout and was killed
    exactly; a follow-up process check found no matching probe.  This test is
    inconclusive and the width was not changed in the active configuration.

45. Direct wheel residual diagnostic, bypassing locomotion feed-forward:
    command was zero, wheel residual scale `1.0`, direct joint target action
    `(-1,+1,+1,-1)` (the TACTIC-inspired yaw pattern), wheel damping `1.0`,
    10-step zero warmup plus 30 total steps:
    `static_direct_yawpattern_damp1_30_retry.log`.
    - Scene creation completed after the temporary-lock cleanup.  At step 0,
      the wheel speeds already matched the ramped target approximately
      `(-0.331,+0.331,+0.331,-0.331) rad/s`, so the direct action path reaches
      the four wheel joints.
    - Over the short rollout, mean `vx=-0.13410 m/s`, mean `wz=-0.02018
      rad/s`, minimum height `0.43093 m`, maximum tilt `0.10971 rad`; no
      termination flag fired.  At step 15 the body had `vx=-0.19315 m/s` and
      `wz=-0.06815 rad/s`; at step 29, `vx=-0.09418 m/s`, `wz=0.01750`.
    - The result is not a valid turn-tracking controller: the nominal yaw
      pattern produces large unwanted translation and no sustained yaw.  It
      does, however, rule out a missing wheel-command write as the sole cause.

46. Alternative direct pattern `(-1,+1,-1,+1)` with the same settings, after
    clearing the verified stale temporary lock:
    `static_direct_yawpattern_alt_damp1_30_retry2.log`.
    - At step 0, the wheel speeds matched the ramped targets approximately
      `(-0.331,+0.331,-0.331,+0.331) rad/s`.
    - Mean body `vx=-0.13188 m/s`, mean `wz=0.01595 rad/s`, minimum height
      `0.43063 m`, maximum tilt `0.11098 rad`; no termination flag fired.  At
      step 15 the body had `vx=-0.20471 m/s`, `wz=0.02151 rad/s`; at step 29,
      `vx=-0.05065 m/s`, `wz=-0.00768 rad/s`.
    - This combination changes the small yaw sign but still produces large
      unwanted translation and no sustained turn.  Neither direct pattern is
      acceptable as a lower-body teacher.

47. Command-path all-plus yaw comparison, correct leg order, DLS disabled,
    wheel damping `1.0`, runtime wheel signs `(1,1,1,1)`, `wz=+0.45`,
    45-step warmup plus 45-step drive:
    `static_allplus_yawleft_nofb_damp1_90.log`.
    - Mean `vx=-0.04823 m/s`, mean `wz=0.00630 rad/s`, minimum height
      `0.43062 m`, maximum tilt `0.11017 rad`; no termination flag fired.
    - At step 89, `vx=-0.01976 m/s`, `wz=0.00300 rad/s`, compared with the
      zero-command baseline `vx=-0.01077`, `wz=0.00336`.  The all-plus
      pattern therefore also fails to add useful yaw authority.

48. High-leg-gain yaw rerun after temporary-lock cleanup: leg stiffness
    `300`, damping `10`, wheel damping `1.0`, rear-sign convention, DLS
    disabled, `wz=+0.45`, 45-step warmup plus 45-step drive:
    `static_active_yawleft_leg300_damp10_wheel1_90_retry.log`.
    - Mean `vx=-0.06304 m/s`, mean `wz=0.00181 rad/s`, minimum height
      `0.43163 m`, maximum tilt `0.11325 rad`; no termination flag fired.
    - At step 89, `vx=0.00147 m/s`, `wz=0.00264 rad/s`; stronger leg gains
      preserve height but do not restore yaw.  Leg compliance alone is not
      the sole explanation for the missing turn response.

49. Direct front-axle-only yaw pattern `(-1,+1,0,0)`, wheel damping `1.0`,
    residual scale `1.0`, 10-step warmup plus 30 steps:
    `static_direct_frontyaw_damp1_30.log`.
    - Mean `vx=-0.13065 m/s`, mean `wz=0.00202 rad/s`, minimum height
      `0.43100 m`, maximum tilt `0.10928 rad`; no termination flag fired.
    - At step 29, `vx=-0.05421`, `wz=-0.00030`; front-only differential
      torque also does not create a sustained yaw response.

50. Direct rear-axle-only yaw pattern `(0,0,+1,-1)`, same settings:
    `static_direct_rearyaw_damp1_30.log`.
    - Mean `vx=-0.14393 m/s`, mean `wz=-0.02134 rad/s`, minimum height
      `0.43041 m`, maximum tilt `0.11006 rad`; no termination flag fired.
    - At step 29, `vx=-0.10838`, `wz=-0.00845`; rear-only differential
      likewise fails.  Together with entries 45/47/48, this points away from
      a single front/rear sign typo and toward the wheel-contact/constraint
      model or an unmodeled steering/support assumption.

51. Ground-friction override probe after fixing the helper path from
    `cfg.terrain` to the actual `cfg.scene.terrain`:
    friction `0.2`, rear-sign convention, wheel damping `1.0`, `wz=+0.45`,
    45-step warmup plus 45-step drive:
    `static_active_yawleft_friction02_90_fixed.log`.
    - The log confirms the override changed `RigidBodyMaterialCfg` from
      `(1.0,1.0)` to `(0.2,0.2)`.
    - Mean `vx=-0.03416 m/s`, mean `wz=-0.00256 rad/s`, minimum height
      `0.39037 m`, maximum tilt `0.14775 rad`; no termination flag fired.
      At step 89, `vx=-0.02986`, `wz=0.00762`, height `0.39376 m`.
    - Lower friction weakens support and does not recover yaw.  It is not
      promoted.

52. Ground-friction override `2.0` with the same command and actuator
    settings: `static_active_yawleft_friction20_90.log`.
    - Mean `vx=-0.10602 m/s`, mean `wz=-0.00917 rad/s`, minimum height
      `0.37149 m`, maximum tilt `0.29155 rad`; no termination flag fired.
    - At step 89, `vx=0.00449`, `wz=-0.00359`, height `0.37716 m` and tilt
      `0.28831 rad`.  Higher friction also fails to generate useful yaw and
      aggravates the posture transient.  Keep the nominal friction `1.0`.

53. The earlier friction runs that produced only Kit startup output were
    invalid helper runs, not physics failures: the probe used `cfg.terrain`
    instead of `cfg.scene.terrain`.  The corrected probe now prints the
    before/after material values and produces complete metrics.  This code
    defect and its correction are retained in the local/remote diagnostic
    script history.

54. Runtime hygiene note: several later IsaacSim launches left the empty
    temporary files `/tmp/hub-root.lock` and `/root/.cache/ov/_cache.lock`.
    No process was using either file when checked; removing only those exact
    lock files restored complete one-step smoke runs.  One smoke process
    (`268776`) remained in scene initialization after a 60-second wrapper
    timeout and was later absent before cleanup; it produced no rollout
    metrics.  These startup/lock events are infrastructure diagnostics, not
    control evidence.

55. Geometry correction to entry 38: the `0.38346 m` value was the separation
    of the four foot-body origins (`y=+/-0.19173 m`), not the wheel mesh
    centers.  The USD collision meshes have local z-center offsets that become
    lateral offsets after their 90-degree X rotation: about `+0.04319 m` for
    FL/RL and `-0.04262 m` for FR/RR.  The actual mesh-center track is therefore
    about `0.46927 m`; the active `0.50 m` track is only about 6.6% larger.
    The earlier `0.38346 m` run remains a body-origin-width sensitivity probe,
    but its 30% over-command interpretation was incorrect and is superseded
    by this correction.  No active configuration change follows from it.

56. TACTIC source inspection (`TACTIC_HRL_env_cfg.py` and
    `tactic_actor_critic.py`) supplies an independent embodiment-level clue:
    its learned motion basis includes yaw candidates
    `(-1,+1,+1,+1)` and `(+1,-1,-1,+1)`, rather than assuming the ideal
    four-wheel differential patterns.  Its stored response chart, ordered
    `(FL,FR,RL,RR)` and output `(vx,wz)`, is approximately
    `[(0.002464,-0.015346), (0.011083,0.015423),
    (-0.008619,0.013075), (0.000000,-0.009260)]`.
    This is source-code evidence of a prior signed action-response sweep, not
    proof that the chart remains valid after the current inertia/asset edits.
    The next diagnostic is to test the positive-yaw basis directly; it must not
    be promoted without a current-asset rollout.

57. Direct test of the TACTIC positive-yaw basis `(-1,+1,+1,+1)`, residual
    scale `1.0`, wheel damping `1.0`, 10-step warmup plus 30 steps:
    `static_direct_tactic_yawplus_damp1_30_retry.log`.
    - At step 0, the four wheel speeds followed the ramped target
      approximately `(-0.333,+0.333,+0.333,+0.333) rad/s`.
    - Mean `vx=-0.16326 m/s`, mean `wz=-0.00681 rad/s`, minimum height
      `0.43019 m`, maximum tilt `0.10243 rad`; no termination flag fired.  At
      step 29, `vx=-0.16259`, `wz=0.02555`.
    - This short direct test does not reproduce the desired yaw response.  It
      is not a rejection of the TACTIC chart in its original action scaling or
      training context, but the chart cannot be transplanted directly into
      the current velocity-target probe.

58. Clean one-process wheel-response sweep on the current TANDEM asset after
    fixing the probe so direct wheel residuals are zero during the warmup.
    Settings: arm frozen, DLS disabled, nominal leg gains, wheel damping
    `1.0`, residual scale `1.0`, magnitude `1.0`, 45 zero-command warmup plus
    45 active steps, action order `(FL,FR,RL,RR)`.  Full output:
    `wheel_response_sweep_damp1_90.log`.
    - `forward_rearflip (1,1,-1,-1)`: active mean
      `vx=0.02331 m/s`, `wz=0.00219 rad/s`; final
      `vx=0.00127`, `wz=0.00902`; minimum height `0.45189 m`, maximum tilt
      `0.02181 rad`; no termination.
    - `ideal_yaw (-1,1,1,-1)`: active mean
      `vx=0.00287`, `wz=0.00244`; final `vx=-0.00437`, `wz=0.00325`;
      minimum height `0.45240 m`, maximum tilt `0.00899 rad`; no termination.
    - `tactic_yaw_plus (-1,1,1,1)`: active mean
      `vx=-0.01008`, `wz=0.00673`; final `vx=-0.02215`, `wz=-0.00155`;
      minimum height `0.44611 m`, maximum tilt `0.02791 rad`; no termination.
    - `tactic_yaw_minus (1,-1,-1,1)`: active mean
      `vx=0.01533`, `wz=0.00109`; final `vx=0.00858`, `wz=0.00069`;
      minimum height `0.45462 m`, maximum tilt `0.01874 rad`; no termination.
    - `allplus_yaw (-1,1,-1,1)`: active mean
      `vx=-0.00391`, `wz=-0.00285`; final `vx=0.00514`, `wz=0.00030`;
      minimum height `0.45213 m`, maximum tilt `0.01525 rad`; no termination.
    - `front_yaw (-1,1,0,0)`: active mean
      `vx=-0.00150`, `wz=-0.00112`; final `vx=0.00146`, `wz=0.00123`;
      minimum height `0.45296 m`, maximum tilt `0.01323 rad`; no termination.
    - `rear_yaw (0,0,1,-1)`: active mean
      `vx=-0.00126`, `wz=0.00146`; final `vx=0.00058`, `wz=0.00335`;
      minimum height `0.45371 m`, maximum tilt `0.01016 rad`; no termination.
    - Interpretation: at low action magnitude the current model has a useful
      forward response but no measurable sustained yaw response.  The clean
      warmup removes the earlier startup contamination; these are diagnostic
      responses, not training success rates.

59. Same clean one-process sweep with magnitude `3.5`, wheel damping `1.0`,
    45-step zero warmup plus 45 active steps, and no arm/DLS changes.
    Full output: `wheel_response_sweep_mag35_damp1_90.log`.
    - `ideal_yaw (-3.5,3.5,3.5,-3.5)`: active mean
      `vx=-0.00388`, `wz=-0.04972`; final `vx=-0.08404`,
      `wz=-0.07884`; minimum height `0.43526 m`, maximum tilt
      `0.07350 rad`; no termination.
    - `tactic_yaw_plus (-3.5,3.5,3.5,3.5)`: active mean
      `vx=-0.01683`, `wz=0.03406`; final `vx=-0.01838`,
      `wz=0.05761`; minimum height `0.37716 m`, maximum tilt
      `0.27800 rad`; no termination flag, but a large posture transient.
    - `tactic_yaw_minus (3.5,-3.5,-3.5,3.5)`: active mean
      `vx=-0.00438`, `wz=0.04002`; final `vx=-0.01868`,
      `wz=0.02607`; minimum height `0.44206 m`, maximum tilt
      `0.05439 rad`; no termination.
    - `allplus_yaw (-3.5,3.5,-3.5,3.5)`: active mean
      `vx=-0.01385`, `wz=0.08537`; final `vx=-0.00129`,
      `wz=0.03431`; minimum height `0.40539 m`, maximum tilt
      `0.25976 rad`; no termination flag, but unsafe posture excursion.
    - `front_yaw (-3.5,3.5,0,0)`: active mean
      `vx=0.01022`, `wz=-0.01276`; final `vx=-0.00447`,
      `wz=-0.00267`; minimum height `0.45345 m`, maximum tilt
      `0.05288 rad`; no termination.
    - `rear_yaw (0,0,3.5,-3.5)`: active mean
      `vx=-0.01771`, `wz=-0.08354`; final `vx=-0.12379`,
      `wz=0.02062`; minimum height `0.43182 m`, maximum tilt
      `0.12537 rad`; no termination.
    - Interpretation: higher wheel commands do create nonzero yaw-like
      transients, but the sign and magnitude are pattern-dependent and not a
      stable differential-drive map.  The ideal pattern produces yaw in the
      opposite sign to the nominal positive-yaw expectation; the TACTIC and
      all-plus patterns induce substantial body translation/tilt.  This is
      evidence against simply changing inertia or blindly transplanting the
      TACTIC signs.  The next discriminating test is a per-wheel response
      matrix with contact/torque diagnostics, followed by a support-aware
      teacher if turning requires leg load redistribution.

60. No-gravity actuator-isolation probe for the ideal yaw target
    `(-3.5,+3.5,+3.5,-3.5)`, `root_z=1.0 m`, wheel damping `1.0`, wheel
    acceleration limit `100`, 5-step warmup plus 35 active steps, arm IK and
    posture feedback disabled: `static_no_gravity_ideal_yaw.log`.
    - Contact force on all four feet was exactly zero.  At step 20 the target
      was approximately `(0.5591,1.2591,-0.5591,-1.2591)` and measured wheel
      velocities were `(0.5590,1.2590,-0.5590,-1.2590)`; at step 39 the
      measured values remained `(0.5591,1.2591,-0.5591,-1.2591)`.
    - Mean base velocity was essentially zero (`vx=2.1e-6 m/s`,
      `wz=1.8e-5 rad/s`) as expected without ground contact; no termination.
    - This isolates the earlier sign anomaly: the wheel actuator target and
      joint-axis signs are correct in free space.  The anomaly appears only
      when the wheels/feet interact with the ground, so changing the global
      wheel sign or only changing inertia is not justified.  Contact geometry,
      rolling constraint, foot support, and load distribution are now the
    primary suspects.

61. Per-wheel ground-response matrix on the current TANDEM asset, magnitude
    `3.5`, wheel damping `1.0`, 45-step zero warmup plus 45 active steps,
    arm frozen and DLS disabled: `wheel_response_sweep_perwheel_mag35.log`.
    The sweep also records wheel targets, actual wheel velocities, wheel
    torques, and foot-contact force norms in `(FL,FR,RL,RR)` order.
    - Baseline contact loading is strongly front/rear asymmetric: active mean
      force norms were roughly `FL 174`, `FR 176`, `RL 245`, `RR 220 N` for
      `fl_plus`, and similarly the rear feet generally carried more load.
    - `fl_plus`: mean base `vx=0.06039`, `wz=0.00837`; mean wheel velocity
      `(0.560,0.099,-0.282,-0.132)`, final height `0.46392 m`, max tilt
      `0.06535 rad`.
    - `fl_minus`: mean `vx=-0.01400`, `wz=-0.00657`; mean wheel velocity
      `(-0.391,0.151,-0.317,0.125)`, max tilt `0.03332 rad`.
    - `fr_plus`: mean `vx=0.01152`, `wz=-0.00733`; mean wheel velocity
      `(-0.273,-0.123,-0.411,0.042)`, max tilt `0.03394 rad`.
    - `fr_minus`: mean `vx=-0.10354`, `wz=-0.00745`; mean wheel velocity
      `(-0.316,-0.922,0.422,0.352)`, final height `0.40713 m`, max tilt
      `0.19607 rad`.
    - `rl_plus`: mean `vx=-0.05593`, `wz=-0.07571`; mean wheel velocity
      `(0.224,-0.202,1.922,0.432)`, final height `0.39003 m`, max tilt
      `0.23478 rad`.
    - `rl_minus`: mean `vx=0.01289`, `wz=0.01599`; mean wheel velocity
      `(0.115,-0.014,-0.998,0.091)`, max tilt `0.04898 rad`.
    - `rr_plus`: mean `vx=-0.05832`, `wz=0.07014`; mean wheel velocity
      `(-0.310,0.158,0.185,1.763)`, final height `0.39482 m`, max tilt
      `0.23358 rad`.
    - `rr_minus`: mean `vx=0.01586`, `wz=-0.02202`; mean wheel velocity
      `(0.051,0.132,0.130,-1.225)`, max tilt `0.05855 rad`.
    - Interpretation: single-wheel commands are not independent in contact;
      they redistribute the whole base and cause the passive wheel speeds to
      change.  Rear-wheel positive commands produce the largest posture loss,
      consistent with the measured rear-heavy support load.  The result does
      not identify inertia as the sole cause; it indicates the lower teacher
      must regulate foot load/posture while commanding motion.  A static
      four-wheel differential target is insufficient for turning on this
      compliant quadruped.

62. USD inertia A/B test with the same clean direct ideal-yaw probe:
    `(-3.5,+3.5,+3.5,-3.5)`, wheel damping `1.0`, 45-step warmup plus
    45 active steps, arm IK and posture feedback disabled.  The comparison
    uses the original shared `quadruped_arm_V3.usd`, while the normal run uses
    the private `quadruped_arm_V3_tandem_inertia.usd`.
    - Original shared USD: `static_originalusd_ideal_yaw_mag35.log`; mean
      `vx=-0.04389`, `wz=0.00087`, minimum height `0.42759 m`, maximum tilt
      `0.11110 rad`; final `vx=0.00437`, `wz=0.00081`.
    - Corrected private USD in the clean sweep: mean
      `vx=-0.00388`, `wz=-0.04972`, minimum height `0.43526 m`, maximum tilt
      `0.07350 rad`; final `vx=-0.08404`, `wz=-0.07884`.
    - Both variants still fail to provide a useful positive-yaw response, but
      the response changes materially with the wheel inertia.  Thus the
      inertia hypothesis is partially valid as a dynamic sensitivity, not a
      sufficient root cause.  The shared USD also retains the same rear-heavy
      contact forces (roughly `FL 167`, `FR 165`, `RL 245`, `RR 236 N` at the
      final sample) and leg sag, so geometry/support imbalance remains
      independently present.

63. Independent comparison of the local advanced `PayloadAwareSupportWBC`
    scaffold on the current remote asset.  The manually driven WBC probe used
    4 environments, 60 settle steps, 180 motion steps, wheel target magnitude
    `6 rad/s` (raw action 60), ramp 60, arm disabled, and patterns
    `(1,1,-1,-1)`, `(-1,1,1,-1)`, `(1,-1,-1,1)`.  Full outputs:
    `probe_wbc_wheel_patterns.log` and
    `probe_wbc_wheel_patterns_summary.json`.
    - Forward pattern: mean `vx=0.00097 m/s`, `wz=-0.00263`, max tilt
      `0.30122 rad`, minimum height `0.37753 m`, support count `4`.
    - First yaw pattern: mean `vx=0.00063`, `wz=0.00315`, max tilt
      `0.30157 rad`, minimum height `0.37824 m`, support count `4`.
    - Opposite yaw pattern: mean `vx=0.00034`, `wz=0.00156`, max tilt
      `0.08045 rad`, minimum height `0.45668 m`, support count `4`.
    - All three were marked geometrically stable by the probe threshold, but
      none produced meaningful translation or yaw.  The WBC scaffold cannot
      be called a fix on the current configuration: it preserves support at
      the cost of cancelling wheel motion, and two randomized starts still
      show a large transient.  It remains a useful structural reference, not
      an accepted teacher/checkpoint.

64. First 20-D SupportWBC direct probe with three patterns exceeded the
    300-second wrapper timeout during scene creation/initialization and did
    not produce rollout metrics.  The exact child process was still present
    after the wrapper returned; PID was verified against the probe command and
    terminated.  This is an infrastructure timeout, not a physics result.
    The full partial log is `probe_skill_wbc_direct.log` on the server.

65. Repeated 20-D SupportWBC probe with one environment and the forward
    pattern `(1,1,-1,-1)`, raw magnitude `60` (physical target `6 rad/s`),
    30 settle plus 60 motion steps, ramp 20:
    `probe_skill_wbc_direct_forward.log` and
    `probe_skill_wbc_direct_forward_summary.json`.
    - The real SupportWBC action contract loaded successfully: action shape
      `20`, observation shape `67`, one active action term.
    - Mean body response was `vx=-0.00485 m/s`, `wz=0.00359 rad/s`; maximum
      tilt `0.05214 rad`, minimum height `0.44907 m`, minimum support count `4`,
      maximum leg correction norm `0.13554 rad`.
    - At the final sample the target remained `(60,60,-60,-60)` in the raw
      20-D action representation, while measured wheel velocities were only
      `(0.522,-0.345,0.270,0.150) rad/s`; the executor maintained support but
      did not transmit useful rolling motion.  This confirms the advanced
      action interface is wired, but its current wheel/ground calibration is
      not yet a usable lower-body teacher.

66. Current-asset clean response sweep with wheel damping `2.0`, magnitude
    `3.5`, 45-step zero warmup plus 45 active steps, arm frozen and DLS off:
    `wheel_response_sweep_mag35_damp2.log`.
    - Forward `(3.5,3.5,-3.5,-3.5)`: active mean
      `vx=0.19078 m/s`, `wz=0.00007`; final `vx=0.20660`, `wz=-0.04031`;
      minimum height `0.42297 m`, maximum tilt `0.08527 rad`; no termination.
      Actual final wheel velocities were `(3.489,3.037,-2.058,-1.908)`.
    - Ideal yaw `(-3.5,3.5,3.5,-3.5)`: active mean
      `vx=-0.06593`, `wz=-0.10595`; final `vx=0.00137`, `wz=-0.01582`;
      minimum height `0.37289 m`, maximum tilt `0.30544 rad`; no termination,
      but posture is unsafe for a frozen-leg teacher.
    - Opposite yaw `(3.5,-3.5,-3.5,3.5)`: active mean
      `vx=-0.06596`, `wz=0.08475`; final `vx=-0.00586`, `wz=0.00365`;
      minimum height `0.36714 m`, maximum tilt `0.30815 rad`; no termination,
      also unsafe.
    - This confirms the force/torque hypothesis partially: damping `2` gives
      useful forward wheel tracking and body speed, but yaw torque transfers
      load into leg collapse.  The next controller must retain the damping-2
      wheel drive while adding a command-conditioned support/leg response; it
      should not simply raise damping globally.

67. Same magnitude `3.5` and wheel damping `2.0` with both leg actuator groups
    strengthened to stiffness `300`, damping `10`:
    `wheel_response_sweep_mag35_damp2_leg300.log`.
    - Forward: active mean `vx=0.03397`, `wz=0.00086`; minimum height
      `0.50041 m`, maximum tilt `0.04421 rad`; mean foot-contact force norms
      `(217.6,213.9,192.1,190.1) N`; no termination.
    - Ideal yaw: active mean `vx=0.02221`, `wz=-0.02926`; minimum height
      `0.51205 m`, maximum tilt `0.03331 rad`; no termination.
    - Opposite yaw: active mean `vx=0.01902`, `wz=0.03051`; minimum height
      `0.51153 m`, maximum tilt `0.04068 rad`; no termination.
    - Higher leg gains remove the large height/tilt collapse and balance the
      front/rear load better, but they also suppress most rolling motion.  This
      is a viable stability teacher/anchor, not yet a command-tracking policy.

68. The conservative lower-body teacher was isolated into a new registered
    environment `ZYB-StableLower-v0`; the existing `ZYB-v0` registration was
    left unchanged.  `stable_lower_env_cfg.py` sets arm IK delta to `0`,
    disables DLS posture feedback, wheel damping to `2.0`, both leg actuator
    groups to stiffness/damping `300/10`, and the initial command box to
    `lin_vel_x=(-0.35,0.35)`, `ang_vel_z=(-0.20,0.20)`.  The actual IsaacLab
    load succeeded with action shape `16`, observation shape `876`, and the
    action terms `leg_pos=12`, `wheel_vel=4`, `arm_ik=0`.
    Full log: `wheel_response_sweep_stable_lower.log`.
    The fixed run used one environment, seed `271828`, 45 zero-command warmup
    steps plus 45 active steps, magnitude `3.5`, wheel damping `2.0`, leg and
    calf stiffness/damping `300/10`, and patterns
    `forward_rearflip`, `ideal_yaw`, `tactic_yaw_minus`.
    - Forward `(3.5,3.5,-3.5,-3.5)`: active mean
      `vx=0.033973`, `wz=0.000864`; final `vx=-0.002417`,
      `wz=-0.001743`; minimum height `0.500408 m`, maximum tilt
      `0.044209 rad`; mean wheel velocity
      `(0.92576,0.92925,-1.08698,-0.96487)`; mean contact-force norms
      `(217.59,213.94,192.11,190.07) N`; no termination term fired.
    - Ideal yaw `(-3.5,3.5,3.5,-3.5)`: active mean
      `vx=0.022208`, `wz=-0.029257`; final `vx=0.006469`,
      `wz=-0.001008`; minimum height `0.512046 m`, maximum tilt
      `0.033314 rad`; mean wheel velocity
      `(-0.20235,0.68191,0.89792,-1.09646)`; mean contact-force norms
      `(226.57,201.59,167.16,218.36) N`; no termination term fired.
    - TACTIC opposite-yaw candidate `(3.5,-3.5,-3.5,3.5)`: active mean
      `vx=0.019016`, `wz=0.030515`; final `vx=-0.000559`,
      `wz=-0.002373`; minimum height `0.511530 m`, maximum tilt
      `0.040681 rad`; mean wheel velocity
      `(0.72527,-0.26978,-1.16259,0.99536)`; mean contact-force norms
      `(207.06,222.60,218.41,165.90) N`; no termination term fired.
    Interpretation: the new environment is a reproducible upright/support
    teacher anchor, but its fixed wheel patterns still do not track the
    requested body velocities.  It is not yet an accepted locomotion policy.

69. A configuration-only import attempt without starting Isaac Sim failed
    with `ModuleNotFoundError: No module named 'pxr'`.  Repeating the check
    through `AppLauncher` and then the real environment probe succeeded.  This
    is an IsaacSim Python initialization issue, not evidence of a controller or
    asset failure; future config checks must use the same AppLauncher path as
    the rollout scripts.

70. The first training-launch attempt used the auxiliary path
    `/root/gpufree-data/legacy_zyb_check/train.py`, but that file had not been
    copied to the server.  The exact Python result was
    `can't open file .../legacy_zyb_check/train.py: [Errno 2] No such file or
    directory`; no simulator environment or training iteration was created.
    The repository's maintained entry point
    `/root/gpufree-data/tandem_hrl_620d798/scripts/rsl_rl/train.py` was then
    used instead; it already contains the current CLI extensions.

71. The first registry check after uploading the new files showed only
    `ZYB-v0`, `ZYB-PointFoot-v0`, `ZYB-Play-v0`, and
    `ZYB-PointFoot-Play-v0`.  Root cause was a deployment filename error:
    the local mirror was named `maniploco_init.py` and had been copied as that
    filename instead of replacing the package's actual `__init__.py`.  The
    exact destination was corrected, and a fresh AppLauncher registry check
    then listed `ZYB-StableLower-v0` as the fifth ZYB environment.

72. Two-iteration PPO smoke training on `ZYB-StableLower-v0` completed using
    the maintained repository entry point.  Full log:
    `train_stable_lower_smoke.log`; run directory:
    `/root/gpufree-data/tandem_hrl_620d798/logs/rsl_rl/maniploco/2026-08-18_05-46-40_stable_lower_smoke/`.
    Settings: seed `42`, `32` environments, `2` iterations, `24` steps per
    environment, action exploration std `0.05`, leg std `0.05`, wheel std
    `0.10`, `training_stage=joint`; checkpoint files `model_0.pt` and
    `model_1.pt` were written.
    - Configuration resolved to
      `quadruped_arm.tasks.manager_based.maniploco.stable_lower_env_cfg:StableLowerEnvCfg`.
    - Actor/critic input width was `876`; action width was `16` with
      `12+4+0` leg/wheel/frozen-arm terms.
    - Iteration 0: `768` total steps, mean reward `-1.83`, velocity metrics
      `error_vel_xy=0.0050`, `error_vel_yaw=0.0072`, termination rates
      `bad_contact=0`, `tilt=0`, `low_height=0`.
    - Iteration 1: `1536` total steps, mean reward `-3.55`, velocity metrics
      `error_vel_xy=0.0435`, `error_vel_yaw=0.0428`, termination rates again
      `bad_contact=0`, `tilt=0`, `low_height=0`.
    - This validates the PPO data path, optimizer step, logging, and checkpoint
      save path.  Two iterations are only a smoke test; these numbers are not
      a learned-policy success rate and no checkpoint is promoted.

73. First two-iteration multi-teacher smoke before the proxy fix used
    `--multi_teacher`, archived teacher
    `/root/gpufree-data/zyb_v0_model_1023.pt`, seed `43`, `32` environments,
    exploration stds `0.05/0.05/0.10` for all/legs/wheels, and the same stable
    lower config.  Training completed, but every logged episode reward,
    termination, and velocity metric was `0.0`.  This was not accepted as a
    result.  Root cause was found in `MultiTeacherVecEnv`: RSL-RL's assignment
    to `episode_length_buf` created a wrapper-local attribute because the
    wrapper only implemented `__getattr__`, so the underlying environment did
    not receive randomized episode ages.  The wrapper was patched with an
    explicit forwarding property and setter.

74. Multi-teacher smoke after the `episode_length_buf` forwarding fix:
    `train_stable_lower_multiteacher_smoke_v2.log`, seed `43`, `32` envs,
    `2` iterations, teacher blend `0.90 -> 0.50` over `100000` environment
    steps, same exploration settings.  The run completed and wrote its
    checkpoints.
    - Iteration 0 (`768` steps): mean reward `-1.16`, episode length `13.00`,
      `error_vel_xy=0.0140`, `error_vel_yaw=0.0180`,
      `teacher_ensemble_match=0.0040`, termination rates
      `bad_contact=0`, `tilt=0`, `low_height=0`.
    - Iteration 1 (`1536` steps): mean reward `-2.16`, episode length `22.56`,
      `error_vel_xy=0.0413`, `error_vel_yaw=0.0371`,
      `teacher_ensemble_match=0.0115`, termination rates
      `bad_contact=0`, `tilt=0`, `low_height=0`.
    This validates the multi-teacher data path; it is not a locomotion success
    result.

75. Checkpoint evaluation used `scripts/rsl_rl/evaluate_checkpoint.py` on
    `ZYB-StableLower-v0`, `model_1.pt`, fixed command `(0.35,0,0)`, and
    deterministic reset.  With `32` environments and `60` steps the process
    remained in scene creation for `364 s`; its log stopped after the base
    environment/terrain messages and produced no CSV.  The exact remaining
    Python PID was verified and killed; no physical metric is taken from this
    run.  A reduced `1`-environment, `5`-step evaluation completed in
    `0.642 s` and wrote `eval_stable_lower_smoke_1.csv`; it is only a startup/
    checkpoint-load sanity check and too short for an episode-level success
    rate.  The discrepancy indicates an evaluation-path or multi-environment
    startup sensitivity, not a proven 32-environment physics failure.

76. A longer multi-teacher run was started with
    `ZYB-StableLower-v0`, seed `123`, `256` environments, `400` requested
    iterations, `24` steps per environment, save interval `50`, teacher blend
    `0.90 -> 0.20` over `500000` environment steps, all/leg/wheel exploration
    stds `0.10/0.08/0.12`, archived ZYB teacher checkpoint, and run name
    `stable_lower_mt_400`.  Full log: `train_stable_lower_mt_400.log`;
    run directory:
    `/root/gpufree-data/tandem_hrl_620d798/logs/rsl_rl/maniploco/2026-08-18_06-00-00_stable_lower_mt_400/`.
    Scene creation took about `3 minutes`; after entering PPO, iteration time
    was about `2.9 s` and the run reached at least iteration `78`.  Only
    `model_0.pt` and `model_50.pt` were saved before the exact training PIDs
    were stopped.
    - Around iteration `10` (`61440` steps):
      `error_vel_xy=0.2645`, `error_vel_yaw=0.1509`,
      `bad_contact=tilt=low_height=0`.
    - Around iteration `31` (`190464` steps):
      `error_vel_xy=0.2675`, `error_vel_yaw=0.2169`, and safety terminations
      were still zero.
    - Iteration `57`: mean reward `-33.97`,
      `error_vel_xy=0.3244`, `error_vel_yaw=0.2287`, tilt termination `0.0039`.
    - Iteration `78`: mean reward `-35.04`,
      `error_vel_xy=0.3253`, `error_vel_yaw=0.2612`, tilt termination `0.0046`.
    Interpretation: this schedule did not improve tracking and began to erode
    stability as teacher authority decayed.  It is a negative training result;
    `model_50.pt` is retained for later diagnosis, not promoted as the stable
    lower-body policy.

77. Intermediate actuator sweep with wheel damping `2.0`, leg/calf
    stiffness/damping `220/8`, magnitude `3.5`, 45-step zero warmup plus
    45 active steps, arm frozen, DLS off: `wheel_response_sweep_leg220_damp8.log`.
    - Forward: active mean `vx=0.039786`, `wz=-0.005763`, minimum height
      `0.469793 m`, maximum tilt `0.068837 rad`; no termination.
    - Ideal yaw pattern `(-3.5,3.5,3.5,-3.5)`: active mean
      `vx=0.040639`, `wz=-0.040973`, minimum height `0.492983 m`,
      maximum tilt `0.061325 rad`; no termination.
    - Opposite yaw pattern `(3.5,-3.5,-3.5,3.5)`: active mean
      `vx=0.049258`, `wz=0.030504`, minimum height `0.491812 m`,
      maximum tilt `0.072720 rad`; no termination.
    This gives modestly more body motion than `300/10` while remaining
    upright in this short fixed probe, but it still does not track a useful
    yaw command by itself.

78. Command-conditioned yaw baseline with high leg gains `300/10`, wheel
    damping `2.0`, command `(vx,wz)=(0,0.2)`, 45-step warmup plus 45 active
    steps, no breakaway and default `wz_sign=+1`:
    `static_cmd_yaw020_stable_leg300_break0.log`.
    Final wheel target was approximately
    `(-0.453,+0.453,+0.453,-0.453)`; mean body `vx=-0.005245`,
    `wz=-0.001795`, minimum height `0.503708 m`, maximum tilt
    `0.033403 rad`; no termination.  The target is below the observed
    contact breakaway response and the sign is opposite the desired positive
    body yaw.

79. Opt-in turn breakaway test with the same settings and
    `turn_breakaway_wz=0.8`: `static_cmd_yaw020_stable_leg300_break08.log`.
    Final target became approximately
    `(-1.813,+1.813,+1.813,-1.813)`, but mean body
    `vx=-0.001806`, `wz=-0.007436`, minimum height `0.503708 m`, maximum
    tilt `0.033403 rad`; no termination.  Increasing wheel target above the
    low-speed command threshold did not create useful yaw, so a simple
    breakaway bias is not promoted.

80. Yaw sign-isolation test with the same high-gain settings and
    `wz_sign=-1`: `static_cmd_yaw020_stable_leg300_wzminus.log`.
    Final target was approximately
    `(+0.453,-0.453,-0.453,+0.453)`; mean body
    `vx=-0.004877`, `wz=+0.001370`, minimum height `0.503708 m`, maximum
    tilt `0.033403 rad`; no termination.  The sign convention is corrected
    relative to the earlier pattern, but the low command still has negligible
    magnitude.  Therefore the final fix requires coordinated leg mobility,
    not only a sign or breakaway parameter.

81. A separate `ZYB-MobilityLower-v0` environment was added and registered;
    the registry now contains both `ZYB-StableLower-v0` and
    `ZYB-MobilityLower-v0`.  It uses wheel damping `2.0`, leg/calf
    stiffness/damping `220/8`, frozen arm IK, DLS disabled, `wz_sign=-1`,
    wheel residual scale `0.25`, command ranges
    `lin_vel_x=(-0.35,0.35)`, `ang_vel_z=(-0.35,0.35)`.  Two-iteration
    multi-teacher smoke (`train_mobility_lower_smoke.log`, seed `44`,
    `32` envs, teacher blend `0.90 -> 0.60`) completed after a variable
    `115.25 s` scene creation.
    - Iteration 0: mean reward `-2.66`,
      `error_vel_xy=0.0052`, `error_vel_yaw=0.0034`,
      `bad_contact=tilt=low_height=0`.
    - Iteration 1: mean reward `-4.02`,
      `error_vel_xy=0.0514`, `error_vel_yaw=0.0466`,
      `bad_contact=tilt=low_height=0`.
    This validates the mobility configuration and sign correction at the PPO
    interface, but it is not yet evidence of learned mobility.

82. A mobility lower-body long run was then started with
    `ZYB-MobilityLower-v0`, seed `144`, `256` environments, `300` requested
    iterations, teacher blend `0.95 -> 0.55` over `1000000` environment steps,
    and exploration stds `0.10/0.08/0.12`.  Log:
    `train_mobility_lower_mt_300.log`.  After `4:57` wall-clock time the log
    still stopped at the base-environment/terrain initialization line, with no
    scene-creation completion and no PPO iteration.  The exact training PID
    was killed; no physical or learning metric is assigned to this run.
    This is a repeat of the server's intermittent large-environment startup
    sensitivity, not evidence against the mobility controller.

83. A reduced `ZYB-MobilityLower-v0` multi-teacher run was launched to avoid
    the 256-environment startup stall.  Exact launch settings were: Isaac
    Python `/workspace/isaaclab/_isaac_sim/python.sh`, `64` environments,
    `200` requested iterations, seed `145`, `24` steps per environment,
    teacher checkpoint `/root/gpufree-data/zyb_v0_model_1023.pt`, teacher
    blend `0.95 -> 0.65` over `300000` environment steps, exploration standard
    deviations `0.10` overall, `0.08` legs, `0.12` wheels, and
    `training_stage=joint`.  The log is
    `/root/gpufree-data/legacy_zyb_check/train_mobility_lower_mt_200.log`.
    Environment initialization took about `3:38`; the run then reached at
    least `7` iterations (`10752` total timesteps, `3.70 s` iteration time).
    At that point the measured velocity errors were `error_vel_xy=0.0786`
    and `error_vel_yaw=0.1452`; termination rates were
    `tilt=0.0625`, `bad_contact=0`, and `low_height=0`; teacher ensemble match
    reward was `0.0497`.  This is not a stable-policy result: the nonzero tilt
    termination during early exploration is evidence that the current action
    distribution can still topple the base.  The process was still running at
    capture time, so later metrics must be appended rather than inferred.

84. The 64-environment run in entry 83 was stopped after the exact Isaac
    Python process was identified and sent `SIGINT`; no training processes
    remained afterward.  The complete stdout log was copied locally as
    `train_mobility_lower_mt_200.log`.  The rollout sequence shows a clear
    degradation rather than monotonic learning.  Selected exact points
    (`total_timesteps: error_vel_xy, error_vel_yaw, tilt termination,
    teacher-match reward) were:
    - `1536: 0.0134, 0.0156, 0.0000, 0.0043`;
    - `6144: 0.1382, 0.0784, 0.0169, 0.0179`;
    - `10752: 0.0786, 0.1452, 0.0625, 0.0497`;
    - `19968: 0.2913, 0.2943, 0.0977, 0.0904`;
    - `26112: 0.2475, 0.2951, 0.0938, 0.1107`;
    - `33792: 0.2883, 0.2603, 0.1061, 0.0840`;
    - `39936: 0.3813, 0.3086, 0.1374, 0.1181`;
    - `47616: 0.3435, 0.3207, 0.1562, 0.0969`;
    - final captured `52224: 0.2517, 0.3037, 0.1406, 0.1055`.
    `bad_contact` and `low_height` stayed `0.0000`; this does not mean the
    robot was stable because the explicit `tilt` termination was nonzero.
    The run reached 34 iterations, with approximately `3.6--5.3 s` per
    iteration after startup.  The run is rejected as a candidate checkpoint:
    the teacher-match reward increased but command errors and tilt rate also
    increased, so the teacher imitation term is not enforcing the desired
    physical behavior in this configuration.

85. A dedicated teacher-interface probe was added and run on the current
    `MobilityLowerEnvCfg`, with one environment, `90` simulation steps, `30`
    zero-command warmup steps, frozen arm IK, and the archived teacher
    `/root/gpufree-data/zyb_v0_model_1023.pt`.  The probe uses the current
    wheel feed-forward controller with zero wheel residual and tests three
    leg-action modes: `teacher_raw`, `teacher_clipped` (raw leg action clipped
    to the current `0.22 m` processed-position residual, equivalent raw limits
    `0.55` for hips and `0.4888889` for thighs/calves), and `ensemble_shield`
    (the initial ZYB/conservative/neutral mixture with alpha `0.95`).  Full
    logs are `teacher_interface_probe_forward.log`,
    `teacher_interface_probe_yawpos.log`, and
    `teacher_interface_probe_yawneg.log`.
    - Forward command `(vx,wz)=(0.35,0.0)`: raw teacher maximum leg output
      `4.2418`, mean fraction beyond the feasible raw limit `0.8861`, mean
      active `vx=0.0170`, mean `wz=0.0192`, minimum height `0.4383 m`, maximum
      tilt `0.08794 rad`; clipped mode mean `vx=-0.0241`, mean `wz=-0.0295`,
      minimum height `0.4321 m`, maximum tilt `0.2667 rad`; shield mode mean
      `vx=0.0045`, mean `wz=-0.0016`, minimum height `0.4413 m`, maximum tilt
      `0.1678 rad`.  None terminated in this 60-step active window.
    - Positive yaw command `(0.0,+0.35)`: raw teacher mean `wz=-0.0169`,
      minimum height `0.3311 m`, maximum tilt `0.5513 rad`, and tilt termination
      at step `79`; clipped mean `wz=-0.0639`, maximum tilt `0.3086 rad`;
      shield mean `wz=-0.0501`, maximum tilt `0.2085 rad`.  No mode tracked the
      requested `+0.35 rad/s`.
    - Negative yaw command `(0.0,-0.35)`: raw teacher mean `wz=-0.0144`,
      minimum height `0.4253 m`, maximum tilt `0.1914 rad`; clipped mean
      `wz=-0.0818`, minimum height `0.4544 m`, maximum tilt `0.3225 rad`;
      shield mean `wz=-0.0726`, minimum height `0.4537 m`, maximum tilt
      `0.2176 rad`.  The output is not a useful signed command response.
    The teacher's raw action scale is therefore incompatible with the current
    bounded leg action interface, and even a componentwise clip does not
    restore command tracking.  The archived teacher is rejected as a direct
    physical teacher for the current asset; it may only be retained as a
    representation/reference after an action-semantic adapter is learned or
    re-calibrated.

86. A scalar-gain ablation extended the teacher probe with a neutral leg
    action and global ZYB leg gains `0.05`, `0.10`, `0.20`, and `0.30` (no
    componentwise clipping).  Forward command `(0.35,0.0)` results were:
    - neutral: mean `vx=0.04011`, mean `wz=-0.00462`, min height
      `0.47056 m`, max tilt `0.06784 rad`;
    - gain `0.05`: mean `vx=0.04196`, mean `wz=-0.00392`, min height
      `0.47515 m`, max tilt `0.05722 rad`;
    - gain `0.10`: mean `vx=0.02780`, mean `wz=-0.00092`, min height
      `0.47297 m`, max tilt `0.04560 rad`;
    - gain `0.20`: mean `vx=0.00809`, mean `wz=-0.00358`, min height
      `0.46484 m`, max tilt `0.05842 rad`;
    - gain `0.30`: mean `vx=-0.00056`, mean `wz=0.01747`, min height
      `0.45942 m`, max tilt `0.05593 rad`.
    The full output is `teacher_interface_probe_forward_gains.log`; the raw,
    clipped, and shield modes were also repeated in that file.  The best
    global gain is statistically indistinguishable from the neutral leg
    reference and still has very poor command tracking, so scalar scaling is
    not promoted.
    A corresponding positive-yaw scalar-gain probe was attempted, but the
    Isaac process remained at terrain initialization for `4:33` with no
    `mode_summary`; its exact process tree was killed and the partial log was
    retained as `teacher_interface_probe_yawpos_gains.log`.  This is recorded
    as startup sensitivity, not as a physics result.

87. The neutral-shield training run from entry 86 was stopped after reaching
    approximately `58` iterations (`89088` timesteps).  It produced
    `model_25.pt` and `model_50.pt`; the latter was copied locally as
    `mobility_lower_neutral_100_model_50.pt`.  Training-side logs through the
    stop showed `tilt=bad_contact=low_height=0`, but the actual policy was
    still executed through `MultiTeacherVecEnv` with a neutral shield whose
    blend had only decayed from `0.95` toward `0.70`.
    A separate fixed-command evaluation of `model_50.pt` used
    `ZYB-MobilityLower-v0`, one environment, deterministic reset, command
    `(vx,vy,wz)=(0.35,0,0)`, and `300` simulation steps.  Full files are
    `eval_mobility_neutral_model50_forward.csv`,
    `eval_mobility_neutral_model50_forward_trace.csv`, and the corresponding
    log.  The trace recorded `2` tilt termination events in `300` steps,
    `0` bad-contact events, and `0` low-height events; mean physical base
    `vx=-0.010777 m/s`, mean `wz=0.104437 rad/s`, mean forward tracking error
    `0.383628 m/s`, and mean yaw error `0.613518 rad/s`.  The CSV's aggregated
    termination field reported `tilt_mean=0.6167`, which is inconsistent with
    the trace event count because the termination buffer persists across reset;
    the per-step trace is the authoritative count.  This checkpoint is not
    promoted: the neutral shield stabilized the training rollout but did not
    produce a standalone deployable lower policy.

88. The intrinsic safety-envelope code was uploaded and registered as
    `ZYB-StandaloneLower-v0` without changing `ZYB-MobilityLower-v0`.  The
    same `model_50.pt` was evaluated on the standalone task with one
    deterministic environment, fixed command `(0.35,0,0)`, and `300` steps.
    The per-step trace recorded `0` tilt, bad-contact, or low-height events;
    this is an improvement over the two tilt events in entry 87.  However,
    the physical trace still had mean `base_vx=-0.010630 m/s`, mean
    `base_wz=0.038232 rad/s`, mean forward error `0.360630 m/s`, and mean yaw
    error `0.196349 rad/s`.  The safety gate therefore bounds the failure but
    does not create motion, and this checkpoint remains unpromoted.  The
    standalone CSV/trace/log are `eval_standalone_model50_forward.*`.

89. A standalone lower-body PPO run was launched on `ZYB-StandaloneLower-v0`
    with no archived teacher checkpoint and no external teacher wrapper:
    seed `147`, `64` environments, `120` requested iterations, `24` steps per
    environment, `teacher_blend_start=0.95`, `teacher_blend_end=0.0`,
    blend horizon `160000` environment steps, exploration standard deviations
    `0.06` overall, `0.05` legs, `0.08` wheels, learning rate `5e-5`, and
    `training_stage=joint`.  The run directory is
    `/root/gpufree-data/tandem_hrl_620d798/logs/rsl_rl/maniploco/2026-08-18_07-19-07_standalone_lower_neutral_120/`;
    the remote stdout is `train_standalone_lower_neutral_120.log`.  It was
    stopped after `167424` environment steps; `model_100.pt` was retained
    locally as `standalone_lower_neutral_120_model_100.pt`.
    While the safety blend was decaying, command errors remained high and no
    termination signal appeared.  After the blend reached zero, tilt-related
    metrics became nonzero (`tilt` about `0.0065` at `164352` steps and
    `0.0156` at `165888`/`167424`), while the yaw error remained roughly
    `0.27--0.71 rad/s`.  The run does not establish a deployable lower policy;
    it shows that the intrinsic gate prevents some immediate failures but
    cannot replace a correctly calibrated motion executor.

90. The retained `standalone_lower_neutral_120/model_100.pt` was evaluated
    without a teacher wrapper on a fixed four-command grid using
    `ZYB-StandaloneLower-v0`, `4` environments, deterministic reset, and
    `300` steps.  The grid order was `(vx,wz)=(-0.35,-0.35),
    (-0.35,+0.35), (+0.35,-0.35), (+0.35,+0.35)`.  Per-environment trace
    means were respectively:
    `(-0.03535,-0.01227)`, `(-0.03711,+0.00310)`,
    `(+0.01380,-0.00481)`, and `(-0.00352,+0.01457)` in `(vx,wz)`;
    all four had zero tilt, bad-contact, low-height, and failure events.
    Thus the checkpoint is safe on this short grid but tracks neither the
    signed translation nor the signed turn command; it is not promoted.

91. Source inspection of TACTIC-HRL found that its physical action layout is
    `17` dimensions (`12` leg actions, `4` wheel actions, `1` gripper), but
    its environment still uses ordinary `JointVelocityActionCfg(scale=0.1)`
    for wheels.  Its learned motion chart uses the old signed basis
    `(1,1,-1,1)` for forward and the stored four-wheel response matrix
    `[(0.002464,-0.015346),(0.011083,0.015423),
    (-0.008619,0.013075),(0,-0.009260)]`.  The current-asset clean sweep
    instead found the rear-flip pattern `(1,1,-1,-1)` to be the only tested
    forward-like pattern, while the old TACTIC-like patterns produced weak or
    posture-coupled responses.  Therefore TACTIC is a reusable architectural
    scaffold, not a validated current-asset teacher.  Before any TACTIC
    training, its wheel executor, basis, response matrix, and leg-support
    interface must be recalibrated and tested on this asset.

92. A one-iteration TACTIC smoke test was run on the server with
    `TACTIC-HRL-Unified-v0`, current private asset configuration, headless
    IsaacSim 5.1, one environment, seed `42`, and reduced exploration
    (`action_std=0.02`, leg `0.01`, wheel `0.03`).  The task parsed with
    `Missing terms=[]`, created the scene, and completed `24` timesteps in
    `7.65 s`; no bad-contact, tilt, low-height, or mission-success termination
    was reported.  The resolved actor input was `1003`, critic input `1532`,
    and actor output `63` (`17` physical plus `46` hierarchy), confirming that
    the TACTIC code path is executable on the current server/asset.  This is
    only an interface and startup result, not evidence of locomotion learning;
    the run directory/log is under `logs/rsl_rl/TACTIC_HRL/` and the copied
    stdout is `train_tactic_smoke.log`.

93. The current-asset reward-semantic correction was uploaded only to the
    lower-body candidate configs and their shared reward implementation:
    `wheel_forward_use` now uses signs `(1,1,-1,-1)`;
    `wheel_turn_support` uses the same signs, measured track width `0.4693 m`,
    and explicit `wz_sign=-1`; the safe wheel command uses the same track
    width in `StableLower` and `MobilityLower`.  The legacy `ZYB-v0` config was
    not changed.  A post-change `ZYB-StableLower-v0` one-iteration smoke test
    with one environment, seed `42`, action/leg/wheel exploration
    `0.02/0.01/0.03`, and learning rate `1e-4` parsed with
    `Missing terms=[]`, resolved to `876` observations and `16` actions, and
    completed `24` timesteps with zero bad-contact, tilt, and low-height
    termination.  Because the run had one environment and one short rollout,
    it produced no meaningful episode tracking metric; it validates the code
    path and reward parameter wiring only.  The stdout is
    `train_stable_lower_rewardfix_smoke.log`.

94. A no-teacher from-scratch PPO run was launched after the reward fix on
    `ZYB-StableLower-v0`: seed `148`, `64` environments, `200` requested
    iterations, `24` steps per environment, action/leg/wheel exploration
    standard deviations `0.05/0.05/0.03`, learning rate `1e-4`, and save
    interval `25`.  The run reached and saved `model_100.pt` at
    `155136` environment timesteps, then was stopped before the remaining
    iterations because the command errors had plateaued.  Around iteration
    `100`, training reported `error_vel_xy=0.3598`,
    `error_vel_yaw=0.2728`, `collision` episode reward about `-5.80`, and
    zero `bad_contact`, `tilt`, and `low_height` termination.  This means the
    policy learned/maintained an upright rollout but not useful command
    tracking; the checkpoint is a safe-hold candidate only, not a stable
    mobility policy.  The remote log is
    `train_stable_lower_scratch_200.log`; the local checkpoint is
    `stable_lower_scratch_200_model_100.pt`.

95. The scratch `model_100.pt` was evaluated independently without a teacher
    wrapper on `ZYB-StableLower-v0`, deterministic reset, four environments,
    `300` steps, and command grid
    `(-0.35,-0.20),(-0.35,+0.20),(+0.35,-0.20),(+0.35,+0.20)` in `(vx,wz)`.
    Per-environment mean `(vx,wz)` was respectively
    `(0.009,-0.029)`, `(0.020,-0.014)`, `(-0.008,-0.019)`, and
    `(-0.007,-0.012) m/s,rad/s`; corresponding mean absolute errors were
    `(0.361,0.192)`, `(0.373,0.232)`, `(0.358,0.267)`, and `(0.357,0.218)`.
    All four had zero tilt, bad-contact, low-height, and failure events.
    The per-step trace is authoritative; this checkpoint is rejected for
    mobility and retained only as a safe-hold diagnostic.

96. The first attempt to evaluate that checkpoint failed before IsaacSim
    initialization because the helper script was run outside the repository
    `scripts/rsl_rl` path and could not import `cli_args`.  Adding
    `PYTHONPATH=/root/gpufree-data/tandem_hrl_620d798/scripts/rsl_rl` allowed
    the exact same evaluation to complete.  This is an infrastructure/path
    error, not a physics or policy result; the failed log is
    `eval_stable_lower_scratch_model100_grid.log` and the successful outputs
    use the `_v2` suffix.

97. An intermediate physical response test was requested with wheel damping
    `2.0`, hip/thigh stiffness/damping `180/6`, calf stiffness/damping
    `180/6`, direct wheel magnitude `3.5`, `45` zero-command warmup steps and
    `45` active steps, testing `forward_rearflip`, `ideal_yaw`, and
    `tactic_yaw_minus`.  Two attempts were made.  Both IsaacSim processes
    reached only `[app ready]` (one after about `5:34`, the retry after about
    `2:17` at capture) and never printed `Time taken for scene creation`,
    `sweep_config`, or a response line.  No other Isaac process held the
    checked cache locks; the exact process trees were terminated and the
    locks were removed only after confirming no simulator remained.  These
    attempts are server/initialization failures and have no physical result;
    logs are `wheel_response_sweep_leg180_damp6.log` and
    `wheel_response_sweep_leg180_damp6_retry.log`.

98. The previously validated safe-hold checkpoint was tested as a multi-teacher
    anchor rather than being used as a claimed mobility policy.  The checkpoint
    was `logs/rsl_rl/maniploco/2026-08-18_07-51-25_stable_lower_scratch_200/model_100.pt`,
    with task `ZYB-MobilityLower-v0`, `32` environments, seed `149`, `2`
    iterations, `24` steps per environment, teacher blend `0.95 -> 0.90`,
    teacher blend horizon `10000` steps, action/leg/wheel exploration
    `0.02/0.02/0.02`, and learning rate `5e-5`.  IsaacSim 5.1 on the L40
    loaded the task with `Missing terms=[]`, enabled multi-teacher mode, and
    completed `1536` timesteps in `7.28 s`; no teacher-shape/load error,
    traceback, bad-contact, tilt, or low-height termination appeared.  The
    output checkpoint directory is
    `logs/rsl_rl/maniploco/2026-08-18_08-16-33_mobility_safe_teacher_model100_smoke`.
    This validates that the safe lower checkpoint is usable as a teacher
    interface for MobilityLower; it does not yet establish command tracking.

99. A short safe-anchor mobility run used that checkpoint as the teacher with
    `ZYB-MobilityLower-v0`, `64` environments, seed `149`, `100` requested
    iterations, save interval `25`, action/leg/wheel exploration
    `0.04/0.04/0.03`, learning rate `5e-5`, and teacher blend `0.95 -> 0.35`
    over `200000` steps.  The run reached at least `47616` timesteps and
    saved `model_25.pt` at
    `logs/rsl_rl/maniploco/2026-08-18_08-18-27_mobility_safe_teacher_model100/`;
    it was then stopped after the trend was judged unsafe to continue.  The
    training metrics fluctuated rather than converging: late samples included
    `error_vel_xy=0.2333..0.3448`, `error_vel_yaw=0.3060..0.3744`, with
    `bad_contact=0`, `low_height=0`, but nonzero tilt termination ranging from
    `0.0156` to `0.0312`.  The run improved over a pure safe-hold policy in
    that it produced some commanded motion, but it did not show monotonic
    learning and was not promoted as a stable policy.  Full stdout is
    `train_mobility_safe_teacher_model100.log`.

100. The first independent evaluation command for `model_25.pt` did not enter
     IsaacSim because the remote two-shell quoting stripped the negative
     `--base_command_grid` argument; argparse reported
     `argument --base_command_grid: expected one argument`.  The corrected
     `--base_command_grid=-0.35,0.35,2,-0.20,0.20,2` form completed normally.
     This is an invocation/quoting failure, not a policy or physics result;
     both logs are retained as `eval_mobility_safe_teacher_model25_grid.log`
     and `eval_mobility_safe_teacher_model25_grid_v2.log`.

101. Independent deterministic evaluation of the safe-anchor `model_25.pt`
     used `ZYB-MobilityLower-v0`, seed `42`, four environments, `300` steps,
     and the command order `(-0.35,-0.20),(-0.35,+0.20),(+0.35,-0.20),
     (+0.35,+0.20)` in `(vx,wz)`.  The per-environment means were
     respectively `(-0.0236,+0.0316)`, `(-0.0278,-0.0122)`,
     `(+0.0777,+0.0220)`, and `(+0.1001,+0.0464)` m/s,rad/s.  The command
     errors were respectively `(0.3267,0.3482)`, `(0.3227,0.3001)`,
     `(0.2723,0.3144)`, and `(0.2499,0.2960)`; all four had zero tilt,
     bad-contact, low-height, and failure events in the per-step trace.
     Therefore the checkpoint is physically safe for this short grid but has
     weak forward authority and incorrect/very weak yaw tracking, so it is
     retained as a diagnostic anchor and rejected as the final mobility lower
     policy.  The authoritative artifacts are
     `eval_mobility_safe_teacher_model25_grid_v2.csv`, its `_per_env.csv`,
     and its `_trace.csv`.

102. A current-asset wheel-response basis sweep used the conservative
     `StableLower` configuration, wheel damping `6.0`, leg stiffness/damping
     `300/10`, direct wheel magnitude `3.5`, `35` zero-command warmup steps
     and `35` active steps.  The single-wheel responses (mean active body
     `(vx,wz)`) were:
     `FL+ (0.0706,0.0062)`, `FL- (-0.0536,0.0039)`,
     `FR+ (0.0775,-0.0048)`, `FR- (-0.0345,-0.0014)`,
     `RL+ (0.0252,-0.0473)`, `RL- (-0.0158,0.0189)`,
     `RR+ (0.0330,0.0450)`, and `RR- (-0.0192,-0.0199)`.
     The combined patterns were
     `forward_rearflip (+,+,-,-): (0.0501,-0.0021)`,
     `forward_tactic (+,+,-,+): (0.1500,0.0300)`,
     `ideal_yaw (-,+,+,-): (0.0498,-0.0475)`,
     `tactic_yaw_plus (-,+,+,+): (0.1384,0.0191)`,
     `tactic_yaw_minus (+,-,-,+): (0.0535,0.0477)`, and
     `allplus_yaw (-,+,-,+): (-0.0196,0.0670)`.
     All tested patterns had zero termination flags; the maximum active tilt
     was `0.0808 rad`.  This is direct evidence that the current asset's
     positive joint-velocity direction is forward for all four wheels; the
     earlier rear sign flip `(1,1,-1,-1)` was not supported by this sweep.
     Full logs are `wheel_response_stablelower_damp6.log` and
     `wheel_response_stablelower_damp6_basis.log`.

103. Lower-only wheel-sign correction was uploaded and compiled: the
     `StableLower` and `MobilityLower` configs now use wheel signs
     `(1,1,1,1)` and yaw reward sign `+1`; the legacy `ZYB-v0` base config was
     left unchanged.  With `MobilityLower`, old wheel damping `2.0`, zero
     policy actions, and a four-command grid
     `(-0.35,-0.20),(-0.35,+0.20),(+0.35,-0.20),(+0.35,+0.20)`, the direct
     command feed-forward produced mean vx
     `(-0.2682,-0.2677,+0.2880,+0.2887)` m/s and mean wz
     `(+0.0010,-0.0011,-0.0023,+0.0018)`; all termination flags were zero.
     The forward sign error was therefore a real controller-interface issue,
     not a PPO conclusion.  Artifacts use the `eval_mobility_signfix_feedforward_grid_v1_*`
     names.

104. A configurable `turn_speed_gain` was added to the safe wheel action and
     set to `8.0` in the two lower configs.  With signs corrected but wheel
     damping still `2.0`, pure yaw commands `wz=-/+0.20` produced only
     `wz=-0.0059/+0.0110` (zero actions, no termination), so the result was
     not promoted.  Repeating with Mobility wheel damping `6.0` produced
     `wz=-0.0409/+0.0403` for the same commands, with zero tilt, bad-contact,
     and low-height events.  This establishes a calibrated but low-authority
     turning channel.  Artifacts use
     `eval_mobility_signfix_yawonly_gain8_v1_*` and
     `eval_mobility_signfix_yawonly_gain8_damp6_v2_*`.

105. With signs `(1,1,1,1)`, `turn_speed_gain=8`, Mobility wheel damping
     `6.0`, and zero policy actions, the original mixed grid produced mean
     `(vx,wz)` `(-0.2025,-0.0786)`, `(-0.2051,+0.0718)`,
     `(+0.1697,-0.0455)`, and `(+0.1694,+0.0403)` m/s,rad/s.  All four had
     zero safety termination events.  Directionality is correct, but command
     magnitude is below the requested `0.35/0.20` box because the wheel target
     saturates and the current contact response is weak.

106. A fresh calibrated safe-anchor PPO run used `ZYB-MobilityLower-v0`,
     signs `(1,1,1,1)`, `turn_speed_gain=8`, Mobility wheel damping `6.0`,
     command box `vx +/-0.25`, `wz +/-0.10`, `64` environments, seed `150`,
     `100` requested iterations, action/leg/wheel exploration
     `0.025/0.025/0.02`, learning rate `5e-5`, and safe teacher checkpoint
     `stable_lower_scratch_200/model_100.pt` with blend `0.98 -> 0.55` over
     `150000` steps.  At about `20k` steps, errors were `xy=0.1898` and
     `yaw=0.1009` with tilt `0.0156`; near `47,616` steps they fluctuated at
     `xy=0.1707`, `yaw=0.1147` with tilt `0.0312`.  `model_25.pt` was saved,
     but the run was stopped before further checkpoints because stability was
     degrading.  The full log is `train_calibrated_mobility_teacher_model100.log`.

107. Deterministic fixed-grid evaluation of that calibrated `model_25.pt`
     used the training command box `(-0.25,-0.10),(-0.25,+0.10),
     (+0.25,-0.10),(+0.25,+0.10)`.  Mean `(vx,wz)` was
     `(-0.2882,+0.0466)`, `(-0.3928,+0.1209)`, `(+0.0204,+0.0269)`, and
     `(-0.0707,+0.1183)`; vx/yaw mean absolute errors were respectively
     `(0.1636,0.3817)`, `(0.2046,0.3495)`, `(0.2326,0.4343)`, and
     `(0.3208,0.3724)`.  Each row had a nonzero tilt rate of `0.0133`
     (with no bad-contact or low-height flag).  The checkpoint is rejected:
     free leg residual learning can override the calibrated wheel behavior
     and even reverse forward direction.  Artifacts use the
     `eval_calibrated_mobility_model25_box_v1_*` names.

108. Implemented a literature-inspired wheel/leg turning coordinator in
     `stabilized_leg_action.py` and registered three diagnostic tasks:
     `ZYB-TurnCoordLower-v0` (nominal leg stiffness),
     `ZYB-TurnCoordSoftLower-v0` (bounded 0.75 minimum stiffness factor), and
     `ZYB-TurnCoordOppositeLower-v0` (opposite hip-offset sign).  The
     coordinator uses a smoothed normalized `vx*wz` signal, applies a bounded
     left/right hip target offset, and optionally writes joint stiffness and
     damping back to PhysX.  The safety gate restores the stiffness factor
     toward one as height or tilt margin deteriorates.  The three files
     compiled with Isaac Python; nominal and soft two-iteration smokes both
     initialized and completed without runtime errors.  This is an
     experimental controller, not a promoted checkpoint.

109. Zero-policy fixed-grid evaluation of `ZYB-TurnCoordLower-v0` used seed
     `42`, four environments, 300 steps, and commands in order
     `(-0.25,-0.10),(-0.25,+0.10),(+0.25,-0.10),(+0.25,+0.10)`.
     Per-environment mean `(vx,wz)` was
     `(-0.1838,-0.0280),(-0.1965,+0.0255),(+0.1981,-0.0066),
     (+0.1643,+0.0084)` m/s,rad/s.  Mean absolute tracking errors were
     respectively `(0.0793,0.0769),(0.0652,0.0766),(0.0565,0.0941),
     (0.0899,0.0961)`; all rows had zero tilt, bad-contact, and low-height
     termination rates.  Forward/backward tracking improved, but yaw
     authority was lower than the calibrated wheel-only feed-forward.  The
     artifact is `eval_turn_coord_nominal_zero_per_env.csv`.

110. Repeating the same zero-policy grid with
     `ZYB-TurnCoordOppositeLower-v0` produced mean `(vx,wz)`
     `(-0.1934,-0.0260),(-0.1852,+0.0275),(+0.1649,-0.0124),
     (+0.1989,+0.0043)` and errors
     `(0.0675,0.0778),(0.0768,0.0750),(0.0894,0.0920),
     (0.0557,0.0959)`.  Safety termination rates were zero in all rows.
     Reversing the hip-offset sign did not restore yaw authority, so the
     present hip-offset mechanism is not promoted as the final turn
     coordinator.  The artifact is `eval_turn_coord_opposite_zero_per_env.csv`.

111. Repeating the grid with `ZYB-TurnCoordSoftLower-v0` and bounded dynamic
     leg stiffness/damping produced mean `(vx,wz)`
     `(-0.2157,-0.0473),(-0.2129,+0.0655),(+0.1939,-0.0635),
     (+0.1988,+0.0180)`.  The command response remained safe in this
     deterministic zero-policy test: all tilt, bad-contact, and low-height
     rates were zero.  The yaw response improved in three signed cases but
     remained asymmetric and did not establish robust tracking; the soft
     version is retained only as an ablation.  Artifact:
     `eval_turn_coord_soft_zero_per_env.csv`.

112. External literature review for the turning design: ETH ANYmal uses
     non-steerable, torque-controlled wheels with online wheel/base trajectory
     optimization, ZMP/whole-body balance, and a hierarchical whole-body
     torque controller; the wheel and torso/support motion are planned
     together rather than produced by a raw leg stiffness change.  ETH's
     whole-body MPC explicitly optimizes wheel/joint velocities and ground
     reaction forces with rolling constraints.  Centauro/Pegasus-style robots
     with steering DoF use unicycle/Ackermann-like wheel constraints and
     hierarchical IK/MPC.  A 2026 Go2-W active-roll paper uses differential
     wheel torque for yaw and knee actuators as active suspension to generate
     anti-roll moment; its nominal stance is held stiff and the MPC minimizes
     lateral load transfer.  Sources: arXiv `1909.07193`, arXiv `2010.06322`,
     ETH Research Collection item `289cfa0a-b7b9-4e0d-87ff-88d74bd2bed9`,
     and GitHub `meisman-ucb/go2w-roll-control-mpc`.

113. The first four-environment, 300-step zero-policy evaluation of
     `ZYB-TurnKneeCoordLower-v0` was launched with seed `42` and the mixed
     command grid, but remained inside `env.reset()` for over six minutes:
     `post_load_evaluation_seed` was never printed and no CSV was produced.
     The evaluator process was safely terminated after checking its state
     (`futex_wait`, about 1.85 GB RSS, no Python exception in the log).  This
     is recorded as an evaluation-initialization hang, not as a controller
     failure.  A subsequent one-environment, ten-step diagnostic completed
     and produced `eval_turn_knee_short.csv` and
     `eval_turn_knee_short_per_env.csv`; it had no safety termination, but
     ten steps are too short to judge steady-state tracking.

114. A reduced four-environment evaluation of `ZYB-TurnKneeCoordLower-v0`
     used seed `42`, 100 steps, zero policy actions, and commands in order
     `(-0.25,-0.10),(-0.25,+0.10),(+0.25,-0.10),(+0.25,+0.10)`.  Mean
     `(vx,wz)` was `(-0.2235,-0.0201),(-0.2238,+0.0198),
     (+0.1626,-0.0470),(+0.1573,+0.0484)` m/s,rad/s.  Mean absolute
     `(vx,wz)` errors were `(0.0660,0.1001),(0.0668,0.1060),
     (0.0884,0.0934),(0.0929,0.0964)`; tilt, bad-contact, low-height, and
     done rates were zero.  The calf/knee offset changes the transient
     distribution but does not yet create reliable yaw authority.  Artifact:
     `eval_turn_knee_zero_100_per_env.csv`.

115. The opposite-sign knee/calf ablation, `ZYB-TurnKneeCoordOppositeLower-v0`,
     used the same seed, grid, 100-step budget, and zero-policy setup.  Mean
     `(vx,wz)` was `(-0.1989,-0.0667),(-0.2103,+0.0668),
     (+0.1885,-0.0310),(+0.1933,+0.0299)`.  Mean absolute errors were
     `(0.0805,0.1042),(0.0658,0.0807),(0.1051,0.0873),
     (0.0973,0.0852)`; all safety termination rates were zero.  The sign
     changes yaw asymmetry but still does not meet the requested +/-0.10
     rad/s tracking box.  Neither knee-sign variant is promoted as a final
     controller.  Artifact: `eval_turn_knee_opposite_100_per_env.csv`.

116. Added wheel-gain and feedback-only diagnostic tasks without changing the
     legacy `ZYB-v0` configuration: `ZYB-TurnGain12Lower-v0`,
     `ZYB-TurnGain16Lower-v0`, `ZYB-TurnGain12FeedbackLower-v0`,
     `ZYB-TurnLoadBalanceLower-v0`, and their ablations.  The new config and
     registration files compiled successfully with Isaac Python.  These are
     explicitly diagnostic tasks; no learned checkpoint was promoted.

117. `ZYB-TurnGain12Lower-v0`, zero policy, seed `42`, four environments,
     100 steps, and the standard mixed grid produced mean `(vx,wz)`
     `(-0.1828,-0.0887),(-0.1895,+0.0872),(+0.1291,-0.0876),
     (+0.1366,+0.0770)`.  Mean yaw errors were
     `0.0689,0.0625,0.0662,0.0643`; mean vx errors were
     `0.0906,0.0795,0.1209,0.1134`.  All tilt, bad-contact, low-height,
     and done rates were zero.  The apparent improvement over gain `8` came
     from a twelve-fold turn-speed calibration, so this is not a physical
     final controller.  Artifact: `eval_turn_gain12_zero_100_per_env.csv`.

118. `ZYB-TurnGain16Lower-v0` used the same setup and produced mean
     `(vx,wz)` `(-0.1682,-0.1242),(-0.1576,+0.1230),
     (+0.0891,-0.1378),(+0.1012,+0.1164)`.  The yaw command was overdriven
     and forward speed degraded; safety rates were still zero in this short
     test.  Gain `16` is rejected.  Artifact:
     `eval_turn_gain16_zero_100_per_env.csv`.

119. A 300-step zero-policy run of gain `12` without yaw feedback showed
     long-horizon yaw means `(-0.0483,+0.0476,-0.0280,+0.0217)` for the four
     grid rows, with yaw errors `(0.0726,0.0703,0.0900,0.0921)` and zero
     tilt/bad-contact/low-height rates.  The short 100-step gain-12 result
     therefore overstated steady-state authority.  The trace also showed
     command-dependent rear-side load transfer.  Artifacts:
     `eval_turn_gain12_zero_300_per_env.csv` and
     `eval_turn_gain12_zero_300_trace.csv`.

120. `ZYB-TurnGain12FeedbackLower-v0` added `vx_feedback_gain=0.15` and
     `wz_feedback_gain=0.50` while keeping the gain-12 wheel calibration.
     At 100 steps it produced mean `(vx,wz)`
     `(-0.1865,-0.0907),(-0.1912,+0.0897),(+0.1412,-0.0859),
     (+0.1451,+0.0860)` and yaw errors `(0.0441,0.0371,0.0469,0.0432)`;
     all safety rates were zero.  At 300 steps yaw errors became
     `(0.0436,0.0416,0.0726,0.0733)` with zero tilt, bad-contact, and
     low-height rates.  This is the best current diagnostic feed-forward
     candidate, but the gain-12 factor remains a simulation compensation,
     not a physically justified final mapping.  Artifacts:
     `eval_turn_gain12_feedback_zero_100_per_env.csv`,
     `eval_turn_gain12_feedback_zero_300_per_env.csv`, and
     `eval_turn_gain12_feedback_zero_300_trace.csv`.

121. A static zero-command calf-offset sweep used four environments with
     left/right offsets `(0,0),(+0.05,-0.05),(-0.05,+0.05),(+0.05,+0.05)`
     applied to both front and rear calves.  The opposite left/right trials
     confirmed that positive left and negative right calf targets shift
     load toward the right, and the reverse shifts load toward the left.
     All rows had zero tilt and low-height rates.  Artifact:
     `eval_static_calf_load_balance_per_env.csv`.

122. Added an optional contact-force load-balancing term that low-pass filters
     `(left_force-right_force)/(left_force+right_force)` and applies a bounded
     calf correction; it is disabled by default.  The strong gain version
     reduced some left/right load imbalance but caused a large yaw error in
     one row (`0.163 rad/s`), so it was not promoted.  The gentle ablation
     preserved the gain-12 feedback yaw errors near `0.043-0.048 rad/s` in
     the 100-step test, but did not improve speed or establish long-horizon
     superiority.  Both remain diagnostic only.  Artifacts:
     `eval_turn_load_balance_zero_100_per_env.csv` and
     `eval_turn_load_balance_gentle_zero_100_per_env.csv`.

123. The remote robot config reports wheel actuator limits of
     `effort_limit_sim=23.5` and `velocity_limit_sim=30.0`; the task-level
     safety clamp was `5.0 rad/s`.  With wheel radius `0.11 m`, `5 rad/s`
     means about `0.55 m/s` wheel-rim speed, not a `5 rad/s` body yaw rate.
     Nevertheless, gain `12` makes the commanded differential speed about
     twelve times the ideal kinematic value and is retained only as a
     compensation diagnostic.  A `6 rad/s` safety-envelope test was started
     but remained in `env.reset()` for over four minutes without a CSV; the
     process was terminated and the configuration was not used for selection.

124. Added physically scaled `turn_speed_gain=1` diagnostics with yaw
     feedback gains `4` and `8`, keeping the wheel clamp at `5 rad/s`.
     Gain `4` produced mean yaw rates `(-0.0143,+0.0133,-0.0184,+0.0187)`
     and yaw errors `(0.0857,0.0867,0.0818,0.0825)`; gain `8` produced
     `(-0.0282,+0.0269,-0.0296,+0.0284)` and errors
     `(0.0718,0.0731,0.0709,0.0741)`.  Both were safe in the 100-step
     test but lack sufficient yaw authority.  This shows that the current
     asset/contact model cannot meet the requested yaw box with a purely
     ideal differential-speed map; a torque/GRF/whole-body controller or
     verified friction/contact calibration is required.  Artifacts:
     `eval_turn_kinematic_feedback4_zero_100_per_env.csv` and
     `eval_turn_kinematic_feedback8_zero_100_per_env.csv`.

125. Implemented a separate `SafeDifferentialWheelTorqueAction` diagnostic
     using physically scaled wheel-speed references, wheel-speed error torque,
     bounded yaw-rate error torque, an 8 Nm torque envelope, and a 100 Nm/s
     torque slew limit.  The new task `ZYB-WheelTorqueLower-v0` compiled and
     completed a two-iteration PPO smoke run without action-manager errors;
     this was an interface validation, not a performance result.  A high-yaw
     torque ablation was also registered separately.

126. Zero-policy 100-step evaluation of `ZYB-WheelTorqueLower-v0` produced
     mean `(vx,wz)` `(-0.2017,-0.0048),(-0.2016,+0.0057),
     (+0.1799,-0.0089),(+0.1750,+0.0084)`.  Yaw errors were
     `(0.0952,0.0943,0.0914,0.0921)` with zero tilt, bad-contact, and
     low-height rates.  The high-yaw torque gain `24` ablation produced
     `(-0.2085,-0.0116),(-0.2047,+0.0097),(+0.1760,-0.0175),
     (+0.1714,+0.0188)` and also failed to provide useful yaw authority.
     Artifacts: `eval_wheel_torque_zero_100_per_env.csv` and
     `eval_wheel_torque_highyaw_zero_100_per_env.csv`.  The torque prototype
     is not promoted; its commanded effort path needs direct target/torque
     trace verification before tuning further.

127. Unit interpretation correction: with `r=0.11 m`, `b=0.4693 m`,
     `vx=0.25 m/s`, and `wz=0.10 rad/s`, physical differential kinematics
     (`turn_speed_gain=1`) gives approximately
     `omega_left=2.06 rad/s` and `omega_right=2.49 rad/s`.  The diagnostic
     `turn_speed_gain=12` instead gives approximately
     `omega_left=-0.29 rad/s` and `omega_right=4.83 rad/s`, which is an
     aggressive compensation and not a reasonable final command mapping.
     The simulator's wheel actuator limit `30 rad/s` is only a hard limit,
     not a recommended operating target.  Gain-12 results are therefore
     explicitly rejected as the final teacher despite their better short
     yaw metrics.

128. Added wheel/base execution tracing to the evaluator.  Each trace now
     records base gravity projection and tilt, four wheel joint velocities,
     simulator velocity/effort targets when available, applied torque, and
     the command-conditioned teacher's feed-forward/residual/target values.
     The torque teacher additionally exposes its physically scaled wheel
     speed reference.  The evaluator and torque-action source both compiled
     with the remote Isaac Python environment.  This instrumentation is
     diagnostic only and does not change the controller policy.

129. `ZYB-WheelTorqueLower-v0` zero-action trace used the torque smoke
     `model_1.pt`, four command-grid environments, 60 steps, and
     `step_dt=0.033333 s`.  The physical wheel references were
     `(-2.059,-2.486)`, `(-2.486,-2.059)`, `(2.486,2.059)`, and
     `(2.059,2.486) rad/s` for the four `(vx,wz)` rows; no wheel reference
     was 5 rad/s.  The largest measured wheel speed in the trace was about
     `6.06 rad/s` in one positive-velocity row, despite the reference being
     at most `2.49 rad/s`; this is an execution overshoot, not a command
     target.  Per-environment maximum tilt was about `0.046-0.065 rad`, with
     no recorded tilt/low-height/bad-contact termination.  Yaw tracking error
     remained about `0.083-0.092 rad/s`, so this prototype is neither a
     selected yaw teacher nor evidence that the torque path is already safe.
     Artifacts: `eval_wheel_torque_trace_steps.csv`,
     `eval_wheel_torque_trace_per_env.csv`, and
     `eval_wheel_torque_trace_summary.csv`.

130. `ZYB-MobilityLower-v0` zero-action trace used the existing gain-8
     wheel teacher checkpoint `model_25.pt`, the same four-row command grid,
     and 60 steps.  Its feed-forward targets reached about `3.98 rad/s`
     (not 5) for the high-side wheel; the `max_wheel_speed=5.0` setting is a
     clamp, not a nominal target.  The physical gain-1 map would be only
     `2.06/2.49 rad/s` for the same command box.  However, because the
     implicit velocity actuator can apply up to its `23.5 Nm` effort limit,
     measured wheel speed overshoot reached roughly `4.21 rad/s` and tilt
     reached `0.149-0.176 rad` in the four rows.  This confirms that the
     unresolved risk is actuator/contact execution and leg-body coupling,
     not a need to raise the wheel-speed target.  Artifacts:
     `eval_mobility_wheel_trace_steps.csv` and
     `eval_mobility_wheel_trace_per_env.csv`.

131. Added the independent `ZYB-PhysicalSafeLower-v0` task.  It uses all
      positive wheel signs, measured track width `0.4693 m`, physical
      `turn_speed_gain=1`, command limits `vx=+-0.25 m/s` and
      `wz=+-0.10 rad/s`, `max_wheel_speed=3.0 rad/s`, and a `4.0 rad/s`
      actuator velocity guard.  Wheel effort is limited to `8.0 Nm`, wheel
      damping is `6.0`, the wheel target slew limit is `4.0 rad/s^2`, and
      the learned wheel residual is limited to `0.05`; the leg residual is
      frozen for this identification stage.  The old diagnostic tasks were
      left unchanged.

132. The new physical-safe task passed a 100-step four-quadrant zero-action
     matrix using the existing compatible checkpoint.  Maximum tilt was
     `0.030-0.034 rad`, no tilt/bad-contact/low-height termination occurred,
     and actual wheel-speed peaks were `2.64-3.45 rad/s`.  The feed-forward
     and rate-limited targets stayed within `2.06-2.48 rad/s`; the remaining
     yaw tracking error was `0.098-0.100 rad/s`, so this is a stable teacher
     anchor but not yet a complete turning teacher.  A 300-step run then
     reached the expected time-out only at the evaluation horizon; all
     other failure terms stayed zero, maximum tilt remained
     `0.030-0.034 rad`, and actual wheel-speed peaks stayed below about
     `3.85 rad/s`.  Artifacts:
     `eval_physical_safe_lower_per_env.csv`,
     `eval_physical_safe_lower_steps.csv`, and
     `eval_physical_safe_lower_long_per_env.csv`.

133. Added `ZYB-PhysicalSafeLearningLower-v0`, inheriting the safe teacher
      envelope but allowing only a `0.08 rad` bounded leg residual with a
      tilt/height safety gate; wheel residual was reduced to `0.03`.  The
      source and task registration compiled remotely.  This is the first
      learning candidate; the `ZYB-PhysicalSafeLower-v0` task remains the
      frozen-reference diagnostic.

134. A five-iteration PPO interface smoke on the learning candidate with 32
      environments completed and saved `model_0.pt` through `model_4.pt`.
      It had zero bad-contact, tilt, and low-height terminations during the
      logged iterations.  A 100-step evaluation of `model_4.pt` also had no
      safety terminations; maximum tilt was `0.036-0.058 rad` and absolute
      wheel-speed peaks were about `2.67-4.10 rad/s`.  Its yaw errors remained
      approximately `0.100-0.116 rad/s`, so this smoke did not yet improve
      turning.

135. A 50-iteration run with 64 environments was started to test learning
     under the same safety envelope.  The reward fell monotonically from
     `-2.22` at iteration 0 to `-18.92` at iteration 15 while the logged
     safety termination rates stayed zero.  This is learning/tracking
     degradation hidden by the safety gate, not a success criterion.  The
     run was stopped after iteration 15; checkpoints through `model_10.pt`
     were retained.  The independent evaluation of `model_10.pt` was
     terminated after more than two minutes without producing a CSV, so it
     is explicitly not selected.  No previous safe-reference artifacts were
     modified.

136. Before the freeze-logic correction, evaluating the old Mobility
      `model_25.pt` on `ZYB-PhysicalSafeLower-v0` exposed the bug: despite
      `max_policy_residual=0`, the unrestricted leg policy moved the legs,
      producing wheel-speed peaks up to `9.13 rad/s` and tilt up to
      `0.361 rad`.  The safety termination flags did not fire, so relying on
      termination alone would have missed this unsafe transient.

137. Corrected `StabilizedLegPositionAction.apply_actions`: a non-positive
      `max_policy_residual` now explicitly sets the target to the default leg
      posture; it no longer skips the clamp.  The source compiled remotely.
      Repeating the old-policy test after the fix gave tilt maxima
      `0.030-0.034 rad`, wheel-speed peaks about `2.37-3.35 rad/s`, and zero
      tilt/bad-contact/low-height terminations.  This fix is now part of the
      remote active source.

138. The old policy evaluated under the `0.08 rad` learning envelope remained
      safe for 60 steps (tilt maxima `0.050-0.058 rad`, no safety terminations)
      but yaw errors were still `0.152-0.189 rad/s`.  The safe envelope is
      therefore effective as a shield, but it does not create missing yaw
      authority by itself.  Artifact:
      `eval_physical_safe_learning_old_policy_per_env.csv`.

139. Enabled the repository's `MultiTeacherVecEnv` for a five-iteration
      smoke using the old checkpoint as the ZYB teacher.  The ZYB,
      half-amplitude conservative, and neutral candidates were published;
      the teacher-match reward rose from `0.0022` to `0.0640`, and all safety
      termination rates stayed zero.  A hold-out 100-step evaluation of
      `model_4.pt` without the training-time wrapper also stayed safe
      (`0.048-0.058 rad` tilt maxima), but yaw errors remained about
      `0.173-0.263 rad/s`.  This validates the shield interface, not a final
      locomotion result.  Artifact:
      `eval_physical_safe_multiteacher_model4_per_env.csv`.

140. Added a bounded `turn_speed_gain=3` diagnostic task.  Its theoretical
      command-box reference is approximately `1.63/2.91 rad/s`, below the
      3-rad/s reference cap.  The first remote evaluation exceeded two
      minutes without producing any CSV, so it was terminated and is not
      selected; no performance claim is made for gain3.

141. Per the user's decision to exercise the 5 rad/s regime, added the
     separate registered task `ZYB-PhysicalSafeWheel5Lower-v0`.  It inherits
     the physical-safe support posture and 8 Nm wheel effort envelope, sets
     `turn_speed_gain=12.8`, `max_wheel_speed=5.0`, zero wheel residual, and
     keeps a `4 rad/s^2` reference slew limit.  With the command corner
     `(vx,wz)=(0.25,0,0.10)`, the feed-forward reference reached
     approximately `4.98 rad/s` on the fast wheels and `-0.46 rad/s` on the
     slow wheels.  The 60-step one-environment diagnostic completed without
     tilt, low-height, bad-contact, or failure termination; actual wheel-speed
     maximum was `4.13 rad/s`, maximum base tilt `0.10055 rad`, mean base
     `(vx,wz)=(0.08823,0.05097)`, and mean tracking errors were
     `(0.16177,0.05239)`.  Artifacts are
     `eval_physical_safe_wheel5_diag.csv`,
     `eval_physical_safe_wheel5_diag_per_env.csv`, and
     `eval_physical_safe_wheel5_diag_steps.csv`.

142. The same 5-rad/s task was evaluated for 100 steps over the four command
     grid rows `(-0.25,-0.10)`, `(-0.25,+0.10)`, `(+0.25,-0.10)`, and
     `(+0.25,+0.10)`, with zero policy residual.  Per-environment mean body
     responses `(vx,wz)` were approximately
     `(-0.161,-0.037)`, `(-0.103,+0.023)`, `(+0.124,-0.025)`, and
     `(+0.129,+0.016)`.  Maximum tilts were `0.133`, `0.086`, `0.154`, and
     `0.144 rad`; maximum absolute measured wheel speeds were about
     `4.963`, `4.823`, `4.826`, and `4.230 rad/s`, respectively.  The
     reference maximum was about `4.98 rad/s` in the high-speed corners and
     all tilt, low-height, bad-contact, and failure termination rates were
     zero.  This supports using the 5-rad/s task as a simulation diagnostic,
     but not claiming long-horizon or real-robot safety yet.  Artifacts are
     `eval_physical_safe_wheel5_grid.csv`,
     `eval_physical_safe_wheel5_grid_per_env.csv`, and
     `eval_physical_safe_wheel5_grid_steps.csv`.

143. Added `ZYB-PhysicalSafeWheel5LearningLower-v0` and
     `ZYB-PhysicalSafeWheel5TeacherLearningLower-v0`, preserving the 5-rad/s
     command envelope while allowing only a bounded leg residual (`0.08 rad`)
     and wheel residual (`0.03`).  A five-iteration multi-teacher PPO smoke
     used seed `151`, 32 environments, the archived ZYB teacher, and partial
     initialization from the compatible Mobility `model_25.pt`.  It completed
     and saved `model_4.pt`; teacher-match reward increased from `0.0036` to
     `0.0396`, but mean reward fell from `-0.55` to `-4.37`,
     `error_vel_xy` rose from `0.0069` to `0.0419`, and
     `error_vel_yaw` rose from `0.0061` to `0.0265`.  Safety termination rates
     remained zero.  This validates the 5-rad/s learning interface only; no
     checkpoint was promoted.  Run directory:
     `logs/rsl_rl/maniploco/2026-08-18_11-38-41_physical_safe_wheel5_multiteacher_smoke`.

144. Independent four-quadrant evaluation of that `model_4.pt` on the 5-rad/s
     learning task, before measured-speed braking, produced maximum absolute
     wheel speeds `5.577`, `5.102`, `5.751`, and `4.431 rad/s`, maximum tilts
     `0.086`, `0.097`, `0.098`, and `0.117 rad`, and yaw errors
     `0.164`, `0.111`, `0.153`, and `0.121 rad/s`; all explicit safety
     termination rates were zero.  This is a negative checkpoint result: the
     learned residual can exceed the reference envelope and does not yet
     track the command reliably.  Artifacts are
     `eval_physical_safe_wheel5_model4_grid.csv`,
     `eval_physical_safe_wheel5_model4_grid_per_env.csv`, and
     `eval_physical_safe_wheel5_model4_grid_steps.csv`.

145. Added optional measured-speed braking to `SafeDifferentialWheelVelocityAction`.
     The 5-rad/s task now detects actual wheel speed beyond `5.0 rad/s` and
     temporarily commands zero effective wheel speed; the evaluator records
     actual velocity, brake target, and brake-active flags.  With a `0.5 rad/s`
     brake margin, the same model's maximum wheel speeds changed only slightly
     to `5.538`, `5.161`, `5.739`, and `4.806 rad/s`.  Setting the brake target
     to zero (`5.0 rad/s` margin) reduced them to `5.309`, `5.102`, `5.514`,
     and `4.431 rad/s`, while maximum tilts were `0.076`, `0.098`, `0.081`,
     and `0.117 rad`.  The brake improves the overspeed transient but is not a
     mathematical hard limiter; the model remains unpromoted.  Artifacts are
     `eval_physical_safe_wheel5_model4_brake_grid.*` and
     `eval_physical_safe_wheel5_model4_brake0_grid.*`.

146. Completed the pending gain-3 reverse-yaw sign test with
     `ZYB-PhysicalSafeGain3ReverseYawLower-v0`, one environment, 60 steps,
     zero policy residual, and command `(0.25,0,0.10)`.  The target wheels
     ended at approximately `(2.91,1.63,2.91,1.63) rad/s`; mean body response
     was `vx=0.1641 m/s`, `wz=-0.00972 rad/s`, maximum tilt `0.0308 rad`,
     and mean yaw error `0.1097 rad/s`, with no safety termination.  Reversing
     only `wz_sign` therefore does not recover positive yaw authority and the
     gain-3 ablation is not promoted.  Artifacts are
     `eval_physical_safe_gain3_reverse_diag.csv`,
     `eval_physical_safe_gain3_reverse_diag_per_env.csv`, and
     `eval_physical_safe_gain3_reverse_diag_steps.csv`.

147. Ran the 30-iteration wheel-only 5-rad/s multi-teacher PPO branch
     `ZYB-PhysicalSafeWheel5WheelOnlyTeacherLearningLower-v0`.  The branch
     explicitly freezes both leg and arm policy residuals and leaves only a
     small wheel residual.  It completed with checkpoints `model_0.pt`,
     `model_10.pt`, `model_20.pt`, and `model_29.pt`, but reward stayed
     negative and degraded from approximately `-16` to `-18.12`; this is a
     training-interface smoke result, not a converged lower-body policy.
     The run used seed `152`, 32 environments, the archived ZYB teacher, and
     the compatible Mobility `model_25.pt` initialization.  Run directory:
     `logs/rsl_rl/maniploco/2026-08-18_11-53-30_physical_safe_wheel5_wheelonly_mt_30`.

148. Evaluated wheel-only `model_29.pt` over the four 5-rad/s command corners
     for 100 steps.  Mean `(vx,wz)` responses were approximately
     `(-0.161,-0.036)`, `(-0.082,+0.017)`, `(+0.095,-0.029)`, and
     `(+0.141,+0.029)`; maximum tilts were `0.136`, `0.081`, `0.149`, and
     `0.141 rad`; maximum absolute wheel speeds were `5.183`, `5.159`,
     `5.334`, and `4.934 rad/s`.  Yaw errors were approximately `0.066`,
     `0.083`, `0.080`, and `0.072 rad/s`, while safety termination terms
     remained zero.  The short test looked acceptable for stability, but it
     was not sufficient for promotion.

149. The same wheel-only `model_29.pt` was then evaluated for 300 steps.
     It completed by time-out only, with no tilt, low-height, bad-contact, or
     failure termination, but maximum absolute wheel speed reached `6.017`
     rad/s and mean yaw errors remained about `0.091` to `0.102 rad/s` across
     the four corners.  This long-horizon result rejects `model_29.pt` as a
     frozen lower-body candidate: short-term stability does not establish
     command tracking or speed-envelope compliance.
     Artifacts are `eval_physical_safe_wheel5_wheelonly_model29_grid.*` and
     `eval_physical_safe_wheel5_wheelonly_model29_long.*`.

150. Evaluated the 5-rad/s measured-feedback variant
     `ZYB-PhysicalSafeWheel5FeedbackLower-v0`.  At 100 steps the four command
     corners had mean `(vx,wz)` approximately
     `(-0.162,-0.036)`, `(-0.103,+0.023)`, `(+0.125,-0.024)`, and
     `(+0.127,+0.028)`, with maximum tilts `0.132`, `0.086`, `0.154`, and
     `0.142 rad`; all safety termination terms were zero.  At 300 steps the
     body means became approximately `(-0.130,-0.008)`, `(-0.116,+0.005)`,
     `(+0.163,-0.008)`, and `(+0.169,+0.008)`, while maximum absolute wheel
     speed reached `6.052` and `5.987 rad/s` in two corners.  Measured
     velocity feedback therefore did not solve the long-horizon yaw drift or
     provide a hard speed limit.  Artifacts are
     `eval_physical_safe_wheel5_feedback_grid.*` and
     `eval_physical_safe_wheel5_feedback_long.*`.

151. Completed the short four-corner test for
     `ZYB-PhysicalSafeWheel5CoordLower-v0`, which adds a bounded, smoothed
     hip-offset coordination term during turning while keeping the arm and
     policy residuals frozen.  Mean `(vx,wz)` responses were approximately
     `(-0.116,-0.037)`, `(-0.080,+0.015)`, `(+0.121,-0.029)`, and
     `(+0.139,+0.035)`.  Maximum tilts were `0.124`, `0.083`, `0.140`, and
     `0.144 rad`; the learned/physical wheel speeds stayed below about
     `4.63 rad/s` in this run and all explicit safety termination terms were
     zero.  The added leg coordination changed the transient and front/rear
     load distribution, but the yaw tracking errors remained about `0.065` to
     `0.085 rad/s`, so this short test does not establish a solution to the
     yaw-mapping problem.  It is retained as an experimental branch and is
     not promoted as the frozen lower-body policy.  Artifacts are
     `eval_physical_safe_wheel5_coord_grid.csv`,
     `eval_physical_safe_wheel5_coord_grid_per_env.csv`, and
     `eval_physical_safe_wheel5_coord_grid_steps.csv`.
