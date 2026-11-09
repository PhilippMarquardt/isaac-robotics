# 3D asset → robot-usable interactive USD in Isaac Lab

This is the operating manual for the pipeline. It assumes the reader is an LLM
agent (you) that has been handed a single 3D asset and is expected to produce an
interactive USD that Isaac Lab can load and a robot can plausibly interact with.

The single hardest part of this job is **not** the USD plumbing. It is the
semantic step: looking at a piece of geometry and deciding, by analogy to the
real-world object it represents, *what parts of it would move, light up, or
respond if a human or robot used it*. The rest of this document tells you how
to act on that analysis.

---

## 0. Read this before downloading anything

You are the LLM. You are responsible for:

1. **Building a complete real-world inventory of the object FIRST,
   independent of the asset.** Before opening any USD: write down every
   moving / emitting / interactable part the *real* version of this object
   has. A real cash register has: cash drawer (slides out), 10–20 keys
   (each presses down), a mechanical lever / crank (rotates), a totalizer
   display (lights up), a key lock (rotates). A real boombox has: 5–10
   buttons, 2 dials (volume, tuning), a cassette door (hinges open), an
   extending antenna, an LED indicator. A real desk lamp has: bulb (emits),
   switch (toggles), bendable arm joints (poseable). Write the inventory
   *before* looking at the asset. This is your fidelity target.
2. **Choosing or accepting a 3D asset.** It can be from Poly Haven, Fab,
   Sketchfab, the user's local files, a Unity export, a Blender export, etc.
3. **Inspecting it as a real object.** Treat the mesh hierarchy and prim names
   as your only window into what the artist provided. Names like `bulb`,
   `switch`, `drawer_03`, `lid`, `handle`, `lever`, `tap`, `knob`, `wheel`,
   `door`, `latch`, `chain`, `pedal`, `button`, `trigger` are strong hints.
   So is any prim that is a *separate Xform/Mesh* from the main body —
   artists split things into separate Xforms exactly when those pieces
   move independently in real life.
4. **Diffing your real-world inventory against the asset.** Every item in
   step 1's inventory falls into one of three buckets:
   - **Asset has its own Xform.** Use it directly as an `animated_part`
     or `light`. Cleanest case.
   - **Asset has the geometry but it's baked into the body mesh** (no
     separate Xform). You must **extract** those faces into a new Xform.
     Three signals to find them:
       1. **`GeomSubset` prims** — if the asset author already tagged the
          faces (e.g. a `_glass` subset for the display window), promote
          the subset to a top-level Mesh prim and use that.
       2. **Material assignment** — keys often share a different material
          slot than the body. Group faces by `material:binding`.
       3. **Spatial bounding box** — when the asset has neither signal,
          identify the part's location from real-world knowledge of the
          object (e.g. "the keys sit on a horizontal plane near the top
          of the body, centered along the front lip"), compute the
          bbox in world space, and extract every triangle whose centroid
          falls inside that bbox.
     The extracted faces leave the original body mesh (use `--cut`) and
     become a new `Mesh` under a new `Xform`. Now they're animatable.
   - **Asset is missing the geometry entirely** (e.g. the real object has
     an LED that the artist never modelled). Only then do you synthesize:
     add a small primitive (cube / sphere / sphere_light) at the right
     world position. Synthesis is a **last resort** — never use it when
     the geometry is actually present but baked in, because the result is
     a floating duplicate that looks ridiculous and confuses the policy.
5. **Producing a spec** that captures every interactive from step 1 — whether
   sourced from the asset or synthesized — then **letting the converter
   and demo runner do the rest**.

**Why this matters: an RL policy trained on an under-specified scene fails
sim-to-real.** If the real cash register has a side lever the cashier pulls
to open the drawer, and your sim has only the drawer animating in isolation,
the policy never learns to engage the lever first. When deployed it will
treat the drawer as openable on demand and fail every interaction. You
**must** match real-world fidelity, even when the asset is monolithic.
Synthesize whatever the asset omits. Under-specification is the single
biggest cause of sim-to-real failure for this pipeline.

Do not skip steps 1, 3, or 4. Skipping 1 means you only see what the
asset gave you (artist bias). Skipping 3 means you animate the wrong thing.
Skipping 4 means you ship an incomplete asset and the policy fails.

---

## 1. The pipeline at a glance

```
[3D asset source]            <- Poly Haven / Fab / Blender / Unity / wherever
        │
        ▼
[download asset]             <- one file (.glb / .fbx / .obj / .usd*) +
                                accompanying textures
        │
        ▼
[convert_to_usd.py]          <- only if source is not already USD
        │
        ▼
[inspect_usd.py]             <- dump prim tree as JSON for analysis
        │
        ▼
[YOU read JSON, identify     <- THIS is the semantic step.
 interactives by analogy        Choose name + kind + motion for each part.
 to real object]
        │
        ▼
[write spec/<asset>.json]    <- single source of truth for the asset
        │
        ▼
[make_interactive.py]        <- wraps source by reference, adds light prims,
                                stores interactive metadata
        │
        ▼
[demo_interactive.py]        <- spawns the wrapper, animates every interactive
                                on a schedule, toggles linked lights
        │
        ▼
(later) physics upgrade      <- add UsdPhysics.PrismaticJoint / HingeJoint
                                via Isaac Sim's property panel or the Omniverse
                                Blender Connector; then the demo runner can
                                detect press / open via joint position instead
                                of running scripted animation
```

Each step is one script. Each script is the **smallest** thing that needs to be
re-run when something changes.

---

## 2. Step-by-step

### 2.1 Pick and download an asset

Priority for new assets is **Poly Haven** (CC0, no auth, native USD files,
already split into per-part Xforms):

```bash
curl -fsL "https://api.polyhaven.com/assets?type=models" | \
  python -c "import json,sys; d=json.load(sys.stdin); [print(k,'|',v.get('categories',[]),'|',v.get('tags',[])[:8]) for k,v in d.items()]"
```

Reject any listing whose `tags` include **`rigged`** or `animated`. Skinned /
armatured assets only show their bind pose when loaded statically in Isaac
Lab; their joints don't drive anything until you re-target them. Stay with
non-rigged assets until you specifically want articulation.

Download the **1k** USD variant plus all `include` textures:

```bash
curl -fsL "https://api.polyhaven.com/files/<asset_slug>" | python -c "
import json, sys
d = json.load(sys.stdin)
u = d['usd']['1k']['usd']
print('USDC URL:', u['url'])
print('--- textures ---')
for k, v in u['include'].items(): print(v['url'])
"
# then curl -fL -A 'Mozilla/5.0' each URL into
# scene_pipeline/assets/source/<asset>/...
```

If the asset is not already USD (FBX, glTF, OBJ), convert it via Isaac Sim's
asset converter:

```
isaaclab.bat -p scripts/convert_to_usd.py \
    --src assets/source/<asset>/scene.glb \
    --dst assets/usd/<asset>.usd
```

That produces a USD with materials baked in, default prim set, Z-up if
possible. Some asset formats default to Y-up; you'll address that in step
2.3 if so.

### 2.2 Inspect the prim tree

Run:

```
isaaclab.bat -p scripts/inspect_usd.py \
    --usd assets/source/<asset>/<asset>_1k.usdc \
    --out_json outputs/<asset>_tree.json
```

`outputs/<asset>_tree.json` contains, for every prim:

- `path`, `type`, `name`
- world-space `bbox_min` / `bbox_max` / `center` / `size`
- flags: `is_mesh`, `is_material`, `is_light`, `is_xform`

Read the JSON in full. You're looking for:

- The **default prim** (the asset's root). All sub-paths inside the wrapper
  USD will live under `/Prop/Visual` and the default prim's *name* gets
  stripped on reference — keep that in mind when writing spec paths.
- The **upAxis** (`Z` is Isaac-native; `Y` requires +90° rotation on X).
- The **metersPerUnit** (must be `1.0`; otherwise scale during spawn).
- Separate **Xforms** at the second level under the default prim. Those are
  candidate interactives.
- **Materials** named like `*_light`, `*_glass`, `*_emit`, `*_screen` — those
  are typical bulb/screen surfaces, useful for emissive toggling.
- **Mesh names** containing `switch`, `button`, `bulb`, `drawer`, `door`,
  `lid`, `cap`, `lever`, `knob`, `wheel`, `crank`, `tap`, `valve`, `latch`,
  `chain`, `trigger`, `pedal`, `handle`, `dial`, `key`.

### 2.2.1 LLM-driven classification

For non-trivial assets, run `classify_with_llm.py` instead of (or in addition
to) the deterministic classifier. It takes the connected-component tree and
the object's real-world name and asks an OpenAI model to label every part:
real-world identity, joint type, interaction effect.

```
python scripts/classify_with_llm.py \
    --tree outputs/<asset>_tree.json \
    --object_kind "vintage hand-crank cash register" \
    --out_json outputs/<asset>_physics_plan.json \
    --out_md   outputs/<asset>_physics_plan.md
```

Requires `OPENAI_API_KEY` in `scene_pipeline/.env`. The LLM sees per-part
size and world-position only — no geometry images — but that's enough to
disambiguate parts whose identity depends on real-world context (e.g. a
thin bar near the top-right corner of a vintage register IS the hand crank,
even when geometry-only heuristics would tag it as decorative trim).

### 2.3 Identify interactives by real-world analogy — **the semantic step**

Open the JSON next to a mental image of the real object. For each candidate
prim, ask:

1. **What is this part in the real world?** A bulb? A drawer face? A power
   switch?
2. **What does a human do with it?** Press it, slide it, rotate it, lift it,
   pull it?
3. **What state(s) can it have?** Binary (on/off, open/closed)? Continuous
   (drawer position 0–1)? Latched (rocker switch — stays where you flipped
   it)?
4. **Does it cause something else to change?** E.g. flipping the switch
   should toggle the bulb's emission. Pulling a door reveals interior light
   if there is one. Pressing a button starts a kettle's heating.

Translate each answer into an entry in the spec under `interactives`:

| kind             | what it represents                | motion (typical)        | required spec fields                                                                |
|------------------|-----------------------------------|-------------------------|-------------------------------------------------------------------------------------|
| `light`          | a bulb / lamp head / screen       | n/a                     | `head_prim`, `on_intensity`, `on_color`, optional `emissive_material`               |
| `animated_part`  | drawer / door / switch / lever    | `translate_y` etc.      | `prim`, `motion`, `amplitude`, optional `linked_light`                              |

Choose `motion` to match the **real direction the part moves**:

- A side rocker switch tipping left ↔ right: `rotate_y` (or `rotate_x`,
  whichever world axis is perpendicular to the switch's long axis).
- A drawer pulled forward: `translate_y` with **negative** amplitude (toward
  the viewer in Z-up conventions where +Y is back-into-the-cabinet).
- A door swinging open about a vertical hinge: `rotate_z`.
- A push button: `translate_z` (negative for depress) with ~3–5 mm amplitude.
- A pedal: `rotate_x` about a low pivot.

If a part has *no* obvious analog motion in the real world (legs of a
chair, base of a lamp, body of a cabinet), **do not** add it as an
`animated_part`. Static geometry should stay static.

If a part has motion *but no link* to a light (a freestanding drawer), leave
`linked_light` unset. The runner will still animate the part on a
schedule but won't touch any light.

### 2.4 Write the spec

`spec/<asset>.json` is the contract. It has two top-level interactive arrays
that the converter and runner treat as equivalent:

- **`interactives`** — parts sourced from existing Xforms in the asset.
- **`synthetic_parts`** — parts you invent (cubes / spheres / cylinders /
  small UsdLuxLights) at a chosen world position because the asset omitted
  the geometry. Synthesised parts use the same `kind`/`motion`/`amplitude`
  schema; they additionally specify the proxy shape and where to put it.

Example for a lamp with one bulb and one switch (asset has them):

```json
{
  "asset": "../assets/source/<asset>/<asset>_1k.usdc",
  "out":   "../assets/usd/<asset>_v2.usd",
  "interactives": [
    {
      "kind": "light",
      "name": "main_bulb",
      "head_prim": "/asset_root/asset_root_bulb",
      "emissive_material": "/asset_root/_materials/asset_root_glass",
      "on_intensity": 5000,
      "on_color": [1.0, 0.95, 0.85]
    },
    {
      "kind": "animated_part",
      "name": "power_switch",
      "prim": "/asset_root/asset_root_switch",
      "motion": "rotate_y",
      "amplitude": 22.0,
      "linked_light": "main_bulb"
    }
  ]
}
```

Example for a monolithic asset (cash register) where keys and lever are
baked into the body. The drawer comes from the asset; everything else is
synthesised:

```json
{
  "asset": "../assets/source/cash_register/CashRegister_01_1k.usdc",
  "out":   "../assets/usd/cash_register_v2.usd",
  "interactives": [
    {
      "kind": "animated_part",
      "name": "cash_drawer",
      "prim": "/CashRegister_01/CashRegister_011_drawer",
      "motion": "translate_y",
      "amplitude": -200.0
    }
  ],
  "synthetic_parts": [
    {
      "kind": "animated_part",
      "name": "lever_side",
      "shape": "cylinder",
      "radius_mm": 8,
      "height_mm": 90,
      "world_pos": [0.31, -0.10, 0.32],
      "world_rot_deg": [90, 0, 0],
      "color": [0.6, 0.55, 0.5],
      "motion": "rotate_x",
      "amplitude": 35.0,
      "linked_light": null,
      "triggers": "cash_drawer"
    },
    {
      "kind": "animated_part",
      "name": "key_total",
      "shape": "cube",
      "size_mm": [18, 18, 12],
      "world_pos": [0.05, -0.10, 0.48],
      "color": [0.85, 0.82, 0.78],
      "motion": "translate_z",
      "amplitude": -6.0
    },
    {
      "kind": "light",
      "name": "display_led",
      "shape": "sphere_light",
      "radius_mm": 8,
      "world_pos": [0.0, -0.15, 0.55],
      "on_intensity": 200,
      "on_color": [1.0, 0.4, 0.2]
    }
  ]
}
```

Three rules:

- All `*_prim` paths in `interactives` are **paths inside the source USD**,
  exactly as printed by `inspect_usd.py`. The converter strips the
  default-prim prefix when writing wrapper metadata, so you don't need to.
- Synthetic parts use **world coordinates** in meters / millimeters. Pick
  them by reading the inspect JSON's bbox for the body (so you know the
  asset's overall extent) and placing your proxy at the *real-world
  location* of the missing part. If the cash register body is at
  `ctr=(0.075, -0.045, 0.313)` size `(0.6, 0.44, 0.62)`, the side lever
  on a vintage register lives at roughly `(+0.30, body_front_face,
  0.5*body_height)`.
- Amplitudes: **degrees** for `rotate_*` motions, **millimeters** for
  `translate_*` motions. Negative values are allowed and meaningful
  (e.g. `-250` for a drawer that pulls forward 25 cm).

#### 2.4-b When to synthesize (decision rules)

For every item in your real-world inventory:

| asset state                                          | action                                       |
|------------------------------------------------------|----------------------------------------------|
| Has dedicated Xform                                  | Use it as `interactives` entry               |
| Has geometry but baked into body mesh (no Xform)     | Synthesize proxy at real-world position      |
| Asset omits the geometry entirely                    | Synthesize proxy + accept visual mismatch    |
| Real object is fluid / particles / non-rigid         | Out of scope for this pipeline (skip)        |

When synthesizing:

- Use the **smallest credible shape**: a 12 mm sphere for an LED, a
  12-15 mm cube for a key cap, a 8 mm × 90 mm cylinder for a side lever.
  Goal is a believable interactive surface for the policy, not artistic
  fidelity.
- Match the color of the asset's diff texture roughly. Pure white is
  fine for tactile buttons; warm yellow/red for warning lights; metallic
  grey for levers.
- Place by reading the body's bbox + your real-world knowledge of where
  the part sits on this kind of object.
- `triggers: "<other_part_name>"` (optional) means "when this synthetic
  part fires its press event, also flip the linked part's state". So
  the side lever pressing triggers the drawer popping out. This is the
  composition primitive that lets you encode "lever → drawer" causality
  the same way `linked_light` encodes "switch → bulb".

#### 2.4-c Fidelity checklist before saving the spec

Before you save `<asset>.json`, count:

- Number of real-world interactives in step 0's inventory: **N_real**.
- Number of entries in `interactives` + `synthetic_parts`: **N_spec**.

If **N_spec < N_real**, you are shipping an under-specified asset and the
RL policy trained on it will fail sim-to-real. Go back and synthesize the
missing items. Mark them in `synthetic_parts` so the next reviewer (human
or LLM) knows which were invented.

### 2.5 Build the wrapper USD

```
isaaclab.bat -p scripts/make_interactive.py --spec spec/<asset>.json
```

This produces `assets/usd/<asset>_v2.usd` — a tiny USD that:

- references the source by relative path under `/Prop/Visual`
- adds a `UsdLuxSphereLight` (intensity 0) at every `light` interactive's
  computed head-bbox center
- stores all interactive descriptors as JSON in custom data under
  `interactive_v2:interactives_json` on the `/Prop` root prim

The wrapper is purely **visual + metadata**. No physics opinions. The runtime
owns animation and physics.

### 2.6 Run the generic demo

```
isaaclab.bat -p scripts/demo_interactive.py \
    --usd assets/usd/<asset>_v2.usd \
    --pos 0,0,0 \
    --cycles 6 --period_s 5 \
    --out_dir outputs/<asset>_demo
```

The runner:

- spawns each `--usd` under `/World/Asset_<i>` at its `--pos`
- creates a `ToggleableLight` for each `light` interactive
- captures the rest transform of each `animated_part` and animates it as a
  latched two-state toggle every `period_s` seconds: `+amplitude` ↔
  `-amplitude` for rotations, `0` ↔ `amplitude*sign` for translations
- when a part has a `linked_light`, toggles that light at the midpoint of the
  transition (when the switch is visually mid-flip)
- saves a PNG per cycle in `--out_dir`

You can pass **multiple** `--usd` and `--pos` pairs to compose a scene out
of several interactive assets — same runner, no extra code.

---

## 3. Decision rules for the LLM (cheat sheet)

When you have the JSON tree in front of you and you have to decide what is
interactive:

1. **One Xform per real-world moving part.** If the artist gave you a
   separate Xform, it almost always moves in the real object.
2. **Material names betray emissive surfaces.** `*_light`, `*_glass`,
   `*_screen`, `*_emit`, `*_neon`, `*_LED` → that material is a candidate
   `emissive_material` and the parent geom is a candidate light head.
3. **Long thin meshes near a base or grip area are switches/levers.** The
   long axis tells you the rotation axis is perpendicular to it.
4. **Repeated Xforms with `_01`/`_02`/`_03` suffixes are drawers, shelves,
   buttons in a row.** Convert all of them. Vary amplitudes / phases if you
   want a wave effect.
5. **If the asset is rigged (`SkelRoot`, `Skeleton` prims present), STOP.**
   Skinned assets do not render correctly statically. Either pick a
   non-rigged variant, bake the skeleton at bind pose in Blender, or
   accept that you'll only see the bind pose mesh until the joints are
   re-targeted to Isaac Sim's articulation system.
6. **If a Y-up indicator is present**, plan to load the wrapper with the
   viewer's `--y_up` flag. The converter records the source's up axis in
   metadata but does not rotate; the viewer applies +90° X rotation on
   load.
7. **When in doubt, omit.** A wrong interactive (a chair leg "switch")
   looks worse than no interactivity. Static is fine.

---

## 4. From scripted animation to real physics

The scripted-animation runner is intentionally simple: it edits
`xformOp:translate` and `xformOp:rotateXYZ` every frame. That is enough to
**visually** prove the interactive behavior. It is **not** enough for RL —
a policy will exploit the lack of physics (teleport through the drawer
face, "press" the switch through the cabinet wall).

To upgrade an asset to physics-correct interactivity:

### 4.1 Static colliders + dynamic rigid bodies

Open the source USD in Isaac Sim:

- For the **lamp body** / **cabinet body** / any non-moving structural
  mesh: select the prim → Property → Add → Physics → `Rigid Body` with
  `Kinematic Enabled = true`, plus `Collision API` with approximation
  `Convex Decomposition` (for complex shapes) or `Convex Hull` (for simple
  ones).
- For each **moving part** (switch, drawer, door, lid): select → add
  `Rigid Body` (dynamic, mass ~10–500 g typically), `Collision API` with
  `Convex Hull`.

### 4.2 Joints

For each moving part, add the right joint **between the static body and
the dynamic body**:

| Real motion          | USD joint                  | Notes                                                   |
|----------------------|----------------------------|---------------------------------------------------------|
| Drawer slide         | `PhysicsPrismaticJoint`    | Axis = direction of pull. Limits = 0 to drawer travel.  |
| Door swing           | `PhysicsRevoluteJoint`     | Axis = hinge axis. Limits = 0° to 90–110°.              |
| Push button          | `PhysicsPrismaticJoint`    | 2–5 mm travel. Add spring drive to return to rest.      |
| Rocker / toggle      | `PhysicsRevoluteJoint`     | ±15–30°. Two stable positions can be done with two     |
|                      |                            | weak springs that fight each other.                     |
| Knob / dial          | `PhysicsRevoluteJoint`     | Full 360° travel or limited.                            |

For each joint, set:
- `body0` = static body
- `body1` = moving body
- `localPos0` = the moving part's world rest position (so the joint's frame
  matches reality)
- Drive API for springy returns: `type=force`, `stiffness=500–5000`,
  `damping=5–50`, `targetPosition=0` (rest)

Save the USD. **At runtime the joint is enforced by PhysX**, so a robot
finger that touches the drawer face actually pulls the drawer open via
contact, and the drawer stops at the joint limits.

### 4.3 Updating the spec for a physics-driven asset

Add a flag and a joint path to the interactive entry:

```json
{
  "kind": "animated_part",
  "name": "drawer_top",
  "prim": "/cabinet/drawer_01",
  "motion": "physics_joint",
  "joint_prim": "/cabinet/drawer_01_joint",
  "joint_travel": 0.25,
  "linked_light": null
}
```

When `motion == "physics_joint"`, the runner skips the scripted animation
and instead reads `physics:state:linear:position` from the joint each step.
A press/open event fires when the joint crosses `0.5 * joint_travel`. This
is the path used by `runtime/interactions.py PressButton(joint_prim_path=...)`.

### 4.4 Blender route

If you prefer to author colliders + bodies + (with the **NVIDIA Omniverse
Blender Connector**) joints inside Blender, do that and export to USD.
Stock Blender's USD exporter writes `RigidBodyAPI` + `CollisionAPI` but
**not** joints — joints require the connector. Either way, the resulting
USD plugs into the same converter; you just write a spec with
`motion: physics_joint` pointing at the joint your DCC authored.

---

## 5. What still has to be human-authored

The pipeline does not (yet) do these for you. If your task needs them, you
write them by hand and the runner will use them:

- **Reward functions** for RL — no inference about "the right reward" can
  come from geometry alone.
- **Robot placement** — the runner does not place a Franka / G1 / arm. Add
  the robot in your own demo script.
- **Articulation between robot and prop** — the runtime doesn't wire the
  robot hand to the switch. The robot just collides like any other body
  (in physics-driven mode) or is irrelevant (in scripted mode).

If the user asks for any of these, they are extensions on top of this
pipeline, not bugs in it.

---

## 6. When something doesn't render

The most common failure modes and what to check first:

- **Scene is empty:** the wrapper USD's reference path is broken. Re-check
  the relative path stored in `prepend references` (dump with
  `scripts/dump_usd_text.py`).
- **Object appears as a single white blob:** the asset is rigged
  (`SkelRoot`). Bake or pick a different asset.
- **Render is gray / black:** dome light intensity too low (default 80–400
  for indoor scenes), or capture camera was never re-aimed via
  `set_world_poses_from_view` (its offset rotation defaults to identity,
  which faces an arbitrary axis).
- **PhysX `no bodies defined at body0 and body1`:** the joint's body
  relationships didn't resolve through the reference. Don't author joints
  inside the wrapper USD — apply them at runtime via
  `runtime/physics_attach.py`, or author them in the source USD before
  referencing.
- **`--headless --enable_cameras` crashes 530 ms in `rtx.scenedb.plugin.dll`:**
  known issue on RTX 5070 Ti + Isaac Sim 5.1 on this machine. Use **windowed**
  mode (omit `--headless`) and let the Kit window open during capture. The
  `view_scene_isaaclab.py` and demo scripts already default to that.

---

## 7. TL;DR for the LLM

1. **Inventory the real object FIRST.** List every interactive, before
   touching any USD. Under-specifying here is the #1 cause of sim-to-real
   failure.
2. Download a non-rigged asset. Prefer Poly Haven for clean per-part Xforms
   and CC0 license.
3. Run `inspect_usd.py`. Read the JSON.
4. Diff your inventory against the prim tree. Every inventory item gets a
   spec entry: either as `interactives` (sourced from an asset Xform) or as
   `synthetic_parts` (a proxy you invent at the right world position).
   **No item is allowed to be silently dropped.**
5. Build with `make_interactive.py`, render with `demo_interactive.py`.
6. If the user wants RL-grade interaction, upgrade to PhysX joints (4.1–4.3).

The inventory + diff steps (1, 4) are yours. The rest is mechanical.
