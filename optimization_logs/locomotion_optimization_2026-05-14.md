# Hexapod locomotion optimization log

Start: 2026-05-14T02:00:32Z
Goal: forward-only straight, smooth tripod gait with low torso roll/pitch/yaw, balanced leg participation, low drag and saturation.

## Baseline context
- Working forward checkpoint candidate: logs/rl_games/hexapod_flat_direct/2026-05-13_20-28-33/nn/hexapod_flat_direct.pth (recent reward around 20.79).
- Dropped backward/mixed command experiment; current command range restored to forward-only (0.15, 0.25) m/s.
- TensorBoard is running on port 6006; no training/playback process active at start.

## Iteration notes

### 2026-05-14T02:03Z - Baseline forward trace from 2026-05-13_20-28-33
Command: fixed forward 0.20 m/s for ~15 s. Summary: `optimization_logs/baseline_forward/summary_forward0p20.md`.

Measured behavior:
- Forward displacement 1.519 m, forward speed 0.101 m/s: moving but only about half commanded speed.
- Lateral drift / forward 0.049: acceptable.
- Yaw delta 0.437 rad: too much heading drift for straight-line goal.
- Mean flat orientation 0.0028 and forward tilt 0.0013: torso pitch/roll proxy is good.
- Tripod-like fraction 0.865 and 23 phase switches: gait is genuinely tripod-ish.
- Foot participation imbalance: FR duty 0.856, BR 0.701, ML 0.771; MR duty 0.252, BL 0.361. Right/front and left-middle are over-contacting; middle-right and back-left under-contact.
- FR drag is worst: mean 0.127 m/s, max 0.706 m/s.
- Mean coxa range 0.747 rad but coxa action saturation rate 0.506: policy often commands coxa bounds, so more action saturation pressure or smaller coxa action scale may help.

Decision: do not overhaul. Keep forward gait and tune for lower coxa saturation, less FR drag, less yaw drift, and more balanced foot duty.

### 2026-05-14T02:05Z - Patch A: conservative cleanup
Changes:
- Fixed reward sign bug: `episode_world_lateral_velocity_abs` is now penalized via `world_lateral_velocity_reward_scale=-1.0` instead of being rewarded positively.
- Increased yaw tracking reward scale 0.35 -> 0.55.
- Increased action saturation penalty -0.35 -> -0.45.
- Slightly increased stuck-foot, mean-drag, and max-drag penalties.
- Added `contact_time_imbalance` penalty to reduce persistent overuse/underuse of individual feet without imposing a hard gait script.

Reasoning: baseline gait is already tripod-like, so focus on straightness, participation balance, and saturation rather than forcing new kinematics.

### 2026-05-14T02:08Z - Patch B: coxa saturation and planted-foot pressure
Patch A comparison vs baseline:
- Yaw improved 0.437 -> 0.374 rad and lateral drift ratio improved 0.049 -> 0.038.
- Forward speed worsened 0.101 -> 0.093 m/s.
- Tripod-like fraction worsened 0.865 -> 0.816.
- MR duty improved 0.252 -> 0.411 and BL duty improved 0.361 -> 0.435, but FR remained over-contacted at 0.875 and drag stayed high.
- Coxa saturation remained high; MR coxa action saturation is 1.0 in both baseline and Patch A despite not being at physical joint limits.

Changes:
- Added `coxa_action_saturation` reward term, scale -0.12, to penalize coxa command saturation directly.
- Tightened max foot contact time 0.65 -> 0.55 s and stuck-foot scale -0.10 -> -0.12.

Reasoning: target the observed stuck/saturated-leg failure mode without imposing a hard scripted gait.

### 2026-05-14T02:11Z - Patch B trace result
Summary: `optimization_logs/patch_b_forward/summary_forward0p20.md`.

Compared to Patch A:
- Lateral drift improved 0.038 -> 0.019.
- Forward speed recovered slightly 0.093 -> 0.097 m/s but remains below baseline 0.101.
- Tripod-like fraction recovered 0.816 -> 0.855, close to baseline 0.865.
- Coxa saturation improved 0.499 -> 0.448.
- But FR duty worsened 0.875 -> 0.888, ML worsened 0.736 -> 0.799, BL worsened 0.435 -> 0.329, MR worsened 0.411 -> 0.352.

Decision: Patch B is mixed. Keep the coxa saturation concept, but undo the harsher planted-foot contact tightening.

### 2026-05-14T02:12Z - Patch C setup
Changes relative to Patch B:
- Reverted max foot contact time 0.55 -> 0.65 s.
- Reverted stuck-foot scale -0.12 -> -0.10.
- Kept coxa-specific action saturation but softened scale -0.12 -> -0.08.

Goal: retain lower coxa saturation without worsening foot duty balance.

### 2026-05-14T02:16Z - Patch C trace result and Patch D setup
Patch C summary: `optimization_logs/patch_c_forward/summary_forward0p20.md`.

Compared with the original baseline:
- Forward speed is about the same: 0.099 m/s vs 0.101 m/s baseline.
- Lateral drift is better: 0.037 vs 0.049 baseline.
- Tripod-like fraction remains close: 0.856 vs 0.865 baseline.
- Coxa saturation is improved: 0.482 vs 0.506 baseline, but still high.
- Yaw drift is worse: 0.531 rad vs 0.437 baseline. FR foot remains over-contacted at 0.873 duty with high drag; MR remains under-contacted at 0.273 duty.

Patch D changes:
- Set yaw command sampling to exactly zero: `yaw_velocity_range=(0.0, 0.0)`. Current goal is straight forward locomotion, so training on random turning was unnecessary and likely encouraged heading wander.
- Increased max stance-foot drag penalty slightly: -0.008 -> -0.012.
- Increased contact-time imbalance penalty slightly: -0.025 -> -0.04.

Reasoning: do not rewrite the gait. The robot is already tripod-ish. Remove yaw-command ambiguity, then apply mild pressure against the FR/MR contact imbalance and single-foot dragging.

### 2026-05-14T02:20Z - Patch D trace result and Patch E setup
Patch D summary: `optimization_logs/patch_d_forward/summary_forward0p20.md`.

Patch D is not a keeper despite a high training scalar peak (~20.58):
- Forward speed improved to 0.114 m/s and tripod-like fraction improved to 0.905.
- But lateral drift became unacceptable: 0.173 lateral/forward.
- Yaw drift worsened to 0.681 rad.
- MR remained under-contacted at 0.200 duty; FR remained over-contacted at 0.852 duty.

Patch E changes:
- Keep straight-yaw command sampling: `yaw_velocity_range=(0.0, 0.0)`.
- Restore the extra Patch D contact/drag penalties to Patch C values: max drag -0.012 -> -0.008, contact imbalance -0.04 -> -0.025.
- Reduce body-frame velocity tracking weight 1.5 -> 1.3 so speed does not dominate straightness.
- Increase yaw-rate tracking weight 0.55 -> 1.0.
- Increase world lateral velocity penalty -1.0 -> -3.0.

Reasoning: Patch D proved the system can increase speed/tripod score but at the cost of direction. Patch E prioritizes straightness without changing the observation space or forcing a hard gait script.

### 2026-05-14T02:25Z - Patch E trace result and Patch F setup
Patch E summary: `optimization_logs/patch_e_forward/summary_forward0p20.md`.

Patch E is the current best by trace:
- Forward speed: 0.106 m/s, slightly above baseline 0.101.
- Lateral drift / forward: 0.028, best so far.
- Yaw delta: 0.413 rad, slightly better than baseline 0.437 and much better than Patch D 0.681.
- Tripod-like fraction: 0.932, best so far.
- Main remaining defect: foot duty imbalance persists. FR duty 0.856 is too high; MR duty 0.191 is too low. This matches the repeated visual observation that right-side participation is still uneven.

Patch F changes:
- Added episode-level foot duty tracking from measured foot contact masks.
- Added penalties for duty variance, too-low foot duty, and too-high foot duty.
- This is deliberately not hard-coded to MR/FR. It pushes all six feet toward participation without prescribing exact phase timing.

Reasoning: the gait is close enough that the next change should target the recurring participation imbalance, not rewrite the locomotion objective.

### 2026-05-14T02:29Z - Patch F trace result
Patch F summary: `optimization_logs/patch_f_forward/summary_forward0p20.md`.

Patch F is rejected:
- Yaw improved to 0.306 rad, but the foot participation objective failed.
- MR duty worsened 0.191 -> 0.173 and FR remained overused at 0.861.
- Lateral drift worsened from Patch E 0.028 -> 0.037.
- Forward speed dropped from Patch E 0.106 -> 0.103.

Decision: disable the episode-foot-duty penalty terms. Current best remains Patch E: straightness-weighted reward with yaw command fixed to zero, yaw-rate tracking scale 1.0, body velocity scale 1.3, and world lateral velocity penalty -3.0.

### 2026-05-14T02:30Z - Patch E render
Rendered current best Patch E checkpoint with fixed forward command 0.20 m/s.
Video copied to `/workspace/IsaacLabEureka/outputs/hexapod_patch_e_best_forward_0p20_2026-05-14.mp4`.

### 2026-05-14T02:43Z - Long Patch E continuation result
Long continuation summary: `optimization_logs/patch_e_long_forward/summary_forward0p20.md`.

The long continuation is rejected despite higher training scalar (~22.94 peak):
- Forward speed dropped badly: 0.106 -> 0.077 m/s.
- Lateral drift worsened: 0.028 -> 0.163.
- Tripod-like fraction worsened: 0.932 -> 0.813.
- It did improve MR duty 0.191 -> 0.375 and reduce FR duty 0.856 -> 0.825, but at the cost of direction and gait quality.

Decision: best current policy remains Patch E from run `2026-05-14_02-20-47`, not the long continuation. Useful lesson: scalar alone is misleading; the duty-improved checkpoint sacrificed straight travel and clean tripod timing.

### 2026-05-14T$(date -u +%H:%MZ) - Full Dr. Eureka launch
Started full Eureka run for `Isaac-Velocity-Flat-Hexapod-Direct-v0` using current Patch E environment setup.
Config: 4 parallel runs, 4 Eureka iterations, 1500 PPO iterations per candidate, rl_games, model gpt-5.4.

### 2026-05-14T02:57Z - Full Dr. Eureka run active
Started full Eureka run for `Isaac-Velocity-Flat-Hexapod-Direct-v0` from the Patch E setup.
Config: 4 parallel candidates, 4 Eureka iterations, 1500 PPO iterations per candidate, rl_games, model gpt-5.4.
Observed run dirs:
- /workspace/IsaacLabEureka/logs/rl_runs/rl_games_eureka/hexapod_flat_direct/2026-05-14_02-53-27_Run-0
- /workspace/IsaacLabEureka/logs/rl_runs/rl_games_eureka/hexapod_flat_direct/2026-05-14_02-53-27_Run-1
- /workspace/IsaacLabEureka/logs/rl_runs/rl_games_eureka/hexapod_flat_direct/2026-05-14_02-53-31_Run-2
- /workspace/IsaacLabEureka/logs/rl_runs/rl_games_eureka/hexapod_flat_direct/2026-05-14_02-53-53_Run-3
Early check: all four workers alive; candidates reached roughly epochs 84-96/1500. Rewards are mixed, which is expected at this stage.
