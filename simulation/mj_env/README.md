# SO101 tabletop MuJoCo environment

A Gymnasium environment for evaluating a LeRobot **π0.5** policy on SO101
grasping: one arm, eight tabletop objects whose graspable feature is in a
different place on each, and observations shaped the way the PI05 processor
expects them (see [`doc/pi05_vla_so101_notes.md`](../doc/pi05_vla_so101_notes.md)).

```python
import gymnasium as gym
import mj_env

env = gym.make("SO101Tabletop-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0, options={"target_object": "cube_red"})

frame = env.unwrapped.policy_observation()   # obs + the "task" prompt string
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Frames and units

The robot base is at the origin, **+x points forward** across the table, +z is
up, and the **tabletop surface is z = 0**, flush with the base. The arm stands
at the rear edge of a 0.90 × 0.84 m table.

`observation.state` and `action` are the six joint positions in **native
MuJoCo units** — radians for the five arm joints and for the calibrated gripper
joint, in `so101.ARM_JOINT_NAMES` order followed by `gripper`. Actions are
absolute position targets for the MJCF's position actuators; control runs at
20 Hz over a 500 Hz simulation (`frame_skip = 25`).

LeRobot's SO101 follower speaks normalized motor units instead, so a checkpoint
trained on real SO101 data needs `mj_env.LeRobotSO101Adapter`:

```python
adapter = mj_env.LeRobotSO101Adapter(env.unwrapped)
state = adapter.to_lerobot(obs["observation.state"])   # -100..100, gripper 0..100
env.step(adapter.from_lerobot(policy_action))
```

The adapter anchors normalization on the MJCF joint limits, because simulation
has no encoder calibration. Before trusting an evaluation, check those limits
against the calibration file of the arm the checkpoint was trained on.

## Observations

| key | shape | notes |
| --- | --- | --- |
| `observation.state` | `(6,)` float32 | joint positions, radians |
| `observation.image` | `(H, W, 3)` uint8 | `global_camera`, top-down view with the arm at bottom-centre and the work area above |
| `observation.wrist_image` | `(H, W, 3)` uint8 | `wrist_camera`, side-mounted and looking obliquely across the jaw opening plane so both fingers remain visible |
| `task` | `str` | added by `policy_observation()` only |

The main GLFW view is an independent free inspection camera and is not a policy
observation. The former `overhead_camera` name is retained as a compatibility
alias for `global_camera`; both refer to the same top-down view. Offscreen rendering is capped at
1280 × 960 by the scene's `offwidth`/`offheight`.

With `render_mode="human"`, a lightweight GLFW viewer shows the global
observation in the top-left corner and the wrist observation in the top-right
corner, plus the current policy prompt centred along the bottom edge. It draws
directly with MuJoCo's OpenGL API, leaving only the main scene, the two policy
inputs, and the prompt. Pass `viewer_observation_overlays=False` to hide the
camera images; the prompt remains visible. Pass `show_viewer_ui=True` to use
MuJoCo's stock passive viewer with both control panels instead; that mode does
not show these custom overlays because its image-overlay path is unreliable on
Windows/MuJoCo 3.11.

## Objects

All eight are free bodies, re-sampled in position and yaw at every reset.
Sizes were chosen against the jaw opening measured off the finger meshes:
2 mm at the closed limit, 94 mm at the open one.

| body | shape | grip width | why it is here |
| --- | --- | --- | --- |
| `rod_red`, `rod_yellow` | 168 mm capsule, ⌀18 mm | 18 mm | graspable anywhere along the shaft, but only across it |
| `sphere_blue`, `sphere_green` | ⌀40 mm ball | 40 mm | no flat faces; needs a wide approach and a firm close |
| `cube_red`, `cube_blue` | 35 mm cube | 35 mm | the easy parallel-face baseline |
| `dumbbell_purple`, `dumbbell_orange` | ⌀44 mm discs on a ⌀16 mm handle | 16 mm | **only the handle fits the jaw** — aiming at the centroid fails |

Colours repeat across shapes on purpose (`cube_red` / `rod_red`), so the prompt
has to disambiguate by shape and not just by colour.

## Reset layout

Objects are rejection-sampled into the polar band **r ∈ [0.17, 0.28] m,
|θ| ≤ 60°** in front of the base, with a keep-out around the base and a
minimum gap between objects. Overlap is tested between 2-D *capsule*
footprints, not enclosing circles — a circle around a 168 mm rod would claim
most of the table and make the layout unpackable. The sampled poses are then
dropped a few millimetres and settled under physics, so nothing starts
interpenetrating and no rest heights are hard-coded.

That band is not arbitrary. **The SO101 has five joints, so it cannot hold an
arbitrary six-DoF pose**; a vertical approach plus an arbitrary jaw roll is
over-constrained. What rescues the top-down grasp is that the jaw is
symmetric, so a roll and that roll plus half a turn are the same grasp —
folding the required roll into [-π/2, π/2] keeps `wrist_roll` inside its
−2.74…2.84 limit. With that reduction, a swept IK check solves an exact
vertical grasp for 100 % of poses out to r = 0.26, 90–100 % at r = 0.28, and
almost none by r = 0.32.

## Success and reward

`info["success"]` requires the target to be **≥ 60 mm off the table while both
finger pads carry load against it**. Contact-based, not distance-based: an
object knocked across the table or flung into the air must not count as a
pick. The grasp is sampled across every physics substep of a control step,
because a held object drops one pad contact for a step here and there while
the wrist accelerates.

Reward shapes reach → hold → lift: `-‖tcp − target‖`, `+1` while grasped,
up to `+5` for lift height (only while grasped), `+10` on success.

## Layout of the package

| file | role |
| --- | --- |
| `assets/scene.xml` | the scene, as plain viewer-openable MJCF |
| `scene.py` | compiles it, and applies what MJCF cannot express |
| `objects.py` | object catalogue + dimensions measured from the model |
| `layout.py` | reset-time placement sampling |
| `gripper.py` | jaw calibration measured off the finger meshes |
| `env.py` | the Gymnasium environment |
| `adapters.py` | radians ↔ LeRobot normalized units |

Two things live in `scene.py` rather than the XML, because they belong to
bodies owned by the included robot file and MJCF cannot amend an included
body: the **wrist camera**, and higher friction on the two finger pads. Both
are applied with `mujoco.MjSpec` after parsing, which keeps
[`so101/assets/so101_new_calib.xml`](../so101/assets/so101_new_calib.xml) a
pure hardware description.

`gripper.py` is worth knowing about before wiring up a policy: the MJCF's
`gripperframe` site sits at the **fingertip**, but the pads that actually hold
an object meet 0.5–4 cm behind and to one side of it. Anything that aims
inverse kinematics at the site drives the fingertips past the object and
closes the jaw on nothing.

## Scripts

```bash
python -m mj_env.scripts.smoke_test          # spaces, determinism, layout, cameras, adapter
python -m mj_env.scripts.render_global_camera # write one global-camera preview PNG
python -m mj_env.scripts.render_wrist_camera # write one wrist-camera preview PNG
python -m mj_env.scripts.scripted_pick --all # solvability baseline, all eight objects
python -m mj_env.scripts.inspect_pick --object dumbbell_purple --film
python -m mj_env.scripts.view_scene --seed 3 # viewer with both observation overlays
```

To tune the wrist camera without opening the interactive viewer, render a
single image and override any candidate camera parameters on the command line:

```bash
python -m mj_env.scripts.render_wrist_camera \
  --position 0 -0.06 -0.015 \
  --target 0 0 -0.15 \
  --up 0 0 -1 \
  --fovy 80 \
  --jaw-open-deg 45 \
  --output .tmp/wrist_camera.png
```

Without these overrides it reads `WRIST_CAMERA_POS`, `WRIST_CAMERA_TARGET`,
`WRIST_CAMERA_UP`, and `WRIST_CAMERA_FOVY` directly from `scene.py`.

The global camera has the same workflow:

```bash
python -m mj_env.scripts.render_global_camera \
  --position 0.15 0 0.72 \
  --target 0.15 0 0 \
  --up 1 0 0 \
  --fovy 52 \
  --width 400 \
  --height 400 \
  --output .tmp/global_camera.png
```

Without overrides it reads `GLOBAL_CAMERA_POS`, `GLOBAL_CAMERA_TARGET`,
`GLOBAL_CAMERA_UP`, and `GLOBAL_CAMERA_FOVY` from `scene.py`.  For a true
top-down view, keep the position and target X/Y pairs equal.  Change both X
values together to move the arm vertically in the image, both Y values
together to balance the left/right margins, and use height/FOVY to zoom.

`scripted_pick` is the one to run before blaming a policy. It is an IK-driven
expert that knows each object's grasp axis, and it currently lifts **40/40**
(all eight objects × five layout seeds). If it stops passing, the environment
regressed, not the policy.

`inspect_pick` prints, per phase, the TCP pose, how far the target has been
shoved, and exactly which geoms are touching it; `--film` writes a
phase × camera contact sheet and `--view` replays the pick in the interactive
viewer. That contact list is what diagnoses a failed grasp — it distinguishes
closing on the object, on the wrong part of it, and on nothing.

Headless machines need an offscreen GL backend, e.g. `MUJOCO_GL=egl`.
