# Hexapod Project — Handoff Brief

You are taking over an in-progress robot RL project from a prior Claude Code
session. The user is building a custom 6-legged hexapod with 3 DOF per leg,
intended for reinforcement-learning-driven locomotion in NVIDIA Isaac Lab.

## Where everything lives

- **GitHub repo:** https://github.com/awah-mike/Hexapod_test
- **User:** awah-mike (email: a@gramcorporation.com)
- **You're running on:** NVIDIA Brev "Isaac Launchable" (L40S 46 GB VRAM, AWS
  Ashburn, ~$3.64/hr). Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11 preinstalled
  under `/workspace/isaaclab/`.
- **Isaac Sim Python launcher:** `/workspace/isaaclab/_isaac_sim/python.sh`
  (use this, not the system `python3`, for any script that imports Isaac modules).
- **Isaac Lab wrapper:** `/workspace/isaaclab/isaaclab.sh -p <script.py>` works too.

## First thing you should do

```bash
cd /workspace
git clone https://github.com/awah-mike/Hexapod_test.git
cd Hexapod_test
cat README.md
cat scripts/fix_urdf.py | head -80
```

The README has the design conventions documented explicitly (joint axes, limits,
masses, naming, body frame). The `fix_urdf.py` header comment explains why each
choice was made. Between those two files plus the URDF itself, you can recover
~95% of the prior session's context.

## Project status (task list)

| # | Task | Status | Notes |
|---|---|---|---|
| 15 | URDF validation + joint-zero pose | **DONE** | `hexapod.urdf` is the cleaned product |
| 16 | GitHub repo setup | **DONE** | Files committed to `awah-mike/Hexapod_test` |
| 17 | URDF → USD conversion + Isaac Lab `ArticulationCfg` | **IN PROGRESS — YOU START HERE** |
| 18 | DirectRL env + gym task registration | pending |
| 19 | PPO baseline training (hand-tuned reward) | pending |
| 20 | (Optional) Eureka reward search | pending |

## Your immediate task: complete task 17

**Two sub-steps:**

### 17a. Generate `hexapod.usd` from `hexapod.urdf`

A script is already written at `scripts/convert_urdf_to_usd.py`. Run it:

```bash
cd /workspace/Hexapod_test
/workspace/isaaclab/_isaac_sim/python.sh scripts/convert_urdf_to_usd.py
```

Expected output: ~20 sec of Isaac Sim boot logs, then `USD written to .../hexapod.usd`.
If the importer API import fails, the script has a fallback path
(`isaacsim.asset.importer.urdf` vs `omni.importer.urdf` — both are tried).

After it runs, inspect:
```bash
ls -la assets/robots/hexapod/hexapod.usd
```
File size should be in the 0.5–5 MB range.

Commit the USD back:
```bash
git config user.email "a@gramcorporation.com"
git config user.name "awah-mike"
git add assets/robots/hexapod/hexapod.usd
git commit -m "Add generated hexapod.usd"
git push
```
(Pushing requires GitHub auth. Easiest: use a Personal Access Token from
https://github.com/settings/tokens with `repo` scope. First push will prompt
for username/password — paste the PAT as the password.)

### 17b. Write the Isaac Lab `ArticulationCfg`

Create `source/hexapod_lab/hexapod_lab/assets/hexapod.py` (or similar — set up
the package structure as you go). It needs to:

1. Define `HEXAPOD_CFG: ArticulationCfg` referencing the USD path.
2. Set `init_state.joint_pos` to a **standing pose** (not q=0, since q=0 is legs
   flat horizontal and the robot would collapse). You'll need to compute
   sensible angles to put the feet on the ground. Suggested starting point:
   - Coxa: 0 (legs spread laterally)
   - Femur: ~0.4–0.6 rad (right side) / -0.4–-0.6 rad (left side)
     [signs are asymmetric because the femur axes are mirrored — see URDF]
   - Tibia: ~1.0–1.5 rad (right side) / -1.0–-1.5 rad (left side)
   Test these values by spawning the robot in Isaac Sim and adjusting until the
   feet are on the ground.
3. Define actuators — `IdealPDActuatorCfg` is fine for now with placeholder
   gains (e.g., `stiffness=40.0`, `damping=2.0`, `effort_limit=10.0`,
   `velocity_limit=10.0`). The user hasn't provided motor specs yet. Make these
   numbers easy to edit later.
4. Configure rigid body + articulation root properties (disable self-collisions
   between adjacent links via `enabled_self_collisions=False`).

**Reference:** look at `/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/anymal.py`
for the canonical pattern. Mimic its structure.

## Critical conventions baked into the URDF (already in README, restated for emphasis)

**Body frame:** +X = right side of body, -X = left, +Y = forward, +Z = up.

**Coxa axes:** `(0, 0, +1)` on right, `(0, 0, -1)` on left. Positive command =
leg swings forward on both sides.

**Femur and tibia axes:** user-specified per leg pair via angles θ measured CCW
about +Z from -Y. Right axes = `(sin θ, -cos θ, 0)`, left mirrored across YZ.
Specific values:
- FR/FL: θ = ±28.583°
- MR/ML: θ = ±1.417°
- BR/BL: θ = ∓26.417°

**Joint limits (asymmetric per side due to mirrored axes):**
- Coxa: ±45°
- Femur: right `[0, +70°]`, left `[-70°, 0]`
- Tibia: right `[-140°, 0]`, left `[0, +140°]`

**Masses:** coxa=0.2 kg, femur=1.0 kg, tibia=0.7 kg, base=11.26 kg.

**Inertias:** isotropic diagonal placeholders (~1e-3 kg·m²). NOT real values.
The CoM positions in `<inertial><origin>` are from CAD and are real. Inertia
magnitudes are estimates. Don't trust them for sim-to-real claims.

**Motor specs:** placeholder `effort=10 N·m`, `velocity=10 rad/s` in the URDF.
User hasn't picked actual motors yet. Revisit before hardware.

## Connection to prior work (optional context)

The user previously ran a successful RL reward-search experiment on Anymal-C
(NOT this hexapod) using IsaacLabEureka, on a different Brev instance. Two
modifications were made to IsaacLabEureka:
- `source/isaaclab_eureka/isaaclab_eureka/config/tasks.py` — added Anymal-C task
  entry with explicit API docs and a compound success_metric
- `source/isaaclab_eureka/isaaclab_eureka/config/prompt_templates.py` — added
  a `PROGRAM_MD` constant (Karpathy/AlphaEvolve-style runbook) prepended to
  the initial LLM prompt

The `PROGRAM_MD` is reusable verbatim for the hexapod Eureka run (task 20). The
Anymal `tasks.py` entry is a template — adapt it for the hexapod's joint names,
shapes, and `_get_observations` method when task 20 comes up.

**Model recommendation:** `gpt-5.4-mini` worked well for reward search.
**Avoid:** `gpt-4o` — it hallucinated APIs and crashed all 12 candidates in
testing.

## Gotchas / things to watch

- **Don't paste multi-line code via bash heredocs** (`cat > file <<EOF`). Whitespace
  and quoting break it. Use the VS Code editor (Ctrl+P → file → Ctrl+A → Delete →
  paste → Ctrl+S) instead.
- **Don't run commands in the terminal where training is happening.** Open new
  terminals via Ctrl+Shift+\` (backtick).
- **Mesh paths in `hexapod.urdf` are `../meshes/X.stl`** (relative to URDF
  location). Isaac Sim's importer resolves this from the URDF's directory.
  Don't move the URDF without moving the meshes or updating paths.
- **The user's design zero pose = legs flat horizontal.** This is NOT the spawn
  pose. The runtime spawn pose comes from `ArticulationCfg.InitialStateCfg.joint_pos`.
  At q=0 the robot would collapse onto its body.
- **The hexapod's coxa joint origins are non-radial.** Middle legs are at
  Y ≈ -0.05 (slightly back of center), not Y = 0. Front/back legs are not
  symmetric distances along Y. Don't assume radial geometry.
- **Joint limits are asymmetric per side** (right gets `[0, +X]`, left gets
  `[-X, 0]`). When writing reward functions or policy actions, remember
  positive command on right ≈ negative command on left for the same physical
  motion.
- **Brev meter runs continuously.** When you finish a work session, stop the
  instance in the Brev web UI. Don't leave it running overnight.

## Other things the user has expressed preferences on

- Prefers tight, structured responses with concrete file paths and commands.
- Likes to verify visually before moving forward (e.g., always wants to inspect
  HTML viz before committing).
- Will provide specific design values when asked (joint angles, motor specs,
  etc.). Don't guess if a design intent isn't clear — ask.
- Comfortable with terminal commands but not deeply expert in Isaac Lab / ROS
  internals.

## You can verify the URDF visually at q=0

`scripts/visualize_urdf.py` generates an interactive plotly HTML showing the
robot's kinematic tree, joint frames, and rotation axes:

```bash
python3 scripts/visualize_urdf.py assets/robots/hexapod/urdf/hexapod.urdf hexapod_zero_pose.html
```

Open the HTML in any browser. Drag to rotate, scroll to zoom. Yellow arrows =
joint rotation axes; small RGB axes at each joint = local XYZ frame.

This isn't critical for task 17 but useful if you want to sanity-check the URDF
before spending compute on USD conversion.

## When task 17 is done

Update the task tracker, push the USD + ArticulationCfg + any new package
scaffolding to GitHub, and tell the user. Then move to task 18 (writing the
DirectRL env class).

End of handoff.
