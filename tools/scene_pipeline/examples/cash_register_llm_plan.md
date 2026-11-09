# LLM physics plan — `vintage hand-crank cash register`

Body: `/CashRegister_01/CashRegister_01_body` size [0.6004907041788101, 0.4360676109790802, 0.6213557496666908]

## Category counts

| category | count |
|---|---|
| `screw_or_fastener` | 7 |
| `bezel_or_moulding` | 6 |
| `action_button` | 3 |
| `keypad_key` | 3 |
| `display_panel` | 2 |
| `switch_toggle` | 2 |
| `decorative_detail` | 2 |
| `drawer` | 1 |
| `hand_crank_lever` | 1 |
| `latch` | 1 |
| `knob_dial` | 1 |

## Per-part assignments

### `cluster_0089`
- members (82): `/CashRegister_01/part_0029`, `/CashRegister_01/part_0030`, `/CashRegister_01/part_0031`, `/CashRegister_01/part_0032`, … +78 more
- real_world_part: front keypad and keyframe assembly
- category: `display_panel`
- size: [0.1828, 0.1352, 0.22069]  center: [0.00836, -0.13735, 0.36207]
- physics: `{"joint_type": "fixed", "mass_kg": 0.35}`
- notes: Fixed front assembly; nearest face is -Y and the large slab-like front cluster reads as the static keypad/display surround rather than a single moving part.

### `cluster_0119`
- members (5): `/CashRegister_01/CashRegister_011_drawer`, `/CashRegister_01/part_0141`, `/CashRegister_01/part_0142`, `/CashRegister_01/part_0165`, … +1 more
- real_world_part: cash drawer
- category: `drawer`
- size: [0.48407, 0.32729, 0.12651]  center: [0.01917, -0.28707, 0.06543]
- physics: `{"joint_type": "prismatic", "axis": "-Y", "lower_m": 0.0, "upper_m": 0.18, "mass_kg": 1.2, "spring_k_N_per_m": 120.0, "damping": 18.0}`
- interaction: `{"primary_action": "pull drawer open", "trigger_threshold_frac": 0.2, "triggers": ["/CashRegister_01/CashRegister_011_drawer"]}`
- notes: Drawer axis follows the front-face normal because this large front slab sits on the -Y face.

### `cluster_0121`
- members (4): `/CashRegister_01/part_0108`, `/CashRegister_01/part_0109`, `/CashRegister_01/part_0167`, `/CashRegister_01/part_0168`
- real_world_part: side hand crank housing and handle group
- category: `hand_crank_lever`
- size: [0.02268, 0.03551, 0.05135]  center: [-0.11199, -0.0918, 0.44693]
- physics: `{"joint_type": "revolute", "axis": "+Z", "lower_deg": -180.0, "upper_deg": 180.0, "mass_kg": 0.12, "spring_k_Nm_per_rad": 0.0, "damping": 0.08}`
- interaction: `{"primary_action": "turn hand crank", "trigger_threshold_frac": 0.15, "triggers": ["/CashRegister_01/part_0108", "/CashRegister_01/part_0109", "/CashRegister_01/part_0167", "/CashRegister_01/part_0168"]}`
- notes: Treated as the side crank assembly; axis chosen vertical from principal_axis_label +Z, consistent with a rotary side crank group.

### `cluster_0106`
- members (3): `/CashRegister_01/part_0004`, `/CashRegister_01/part_0139`, `/CashRegister_01/part_0146`
- real_world_part: top amount display hood
- category: `display_panel`
- size: [0.05636, 0.08058, 0.02636]  center: [0.16275, -0.04528, 0.52534]
- physics: `{"joint_type": "fixed", "mass_kg": 0.08}`
- notes: Fixed top display feature; it sits near the +Z face and its slab form does not suggest a push or hinge motion.

### `cluster_0097`
- members (3): `/CashRegister_01/part_0065`, `/CashRegister_01/part_0133`, `/CashRegister_01/part_0135`
- real_world_part: front action button
- category: `action_button`
- size: [0.03788, 0.02677, 0.03495]  center: [0.13264, -0.17042, 0.29549]
- physics: `{"joint_type": "prismatic", "axis": "+Y", "lower_m": 0.0, "upper_m": 0.005, "mass_kg": 0.005, "spring_k_N_per_m": 800.0, "damping": 1.0}`
- interaction: `{"primary_action": "press button", "trigger_threshold_frac": 0.6, "triggers": ["/CashRegister_01/part_0065", "/CashRegister_01/part_0133", "/CashRegister_01/part_0135"]}`
- notes: Front-face button uses inward +Y motion because the cluster sits on the -Y face and is a compact cube-like protrusion.

### `cluster_0096`
- members (2): `/CashRegister_01/part_0130`, `/CashRegister_01/part_0134`
- real_world_part: front bezel strip
- category: `bezel_or_moulding`
- size: [0.02935, 0.00575, 0.01534]  center: [0.12837, -0.13676, 0.34136]
- physics: static
- notes: Static trim; the thin strip at the -Y front face is better read as bezel detail, with principal axis +X only indicating horizontal placement.

### `cluster_0111`
- members (2): `/CashRegister_01/part_0150`, `/CashRegister_01/part_0151`
- real_world_part: front switch lever
- category: `switch_toggle`
- size: [0.00427, 0.03029, 0.04063]  center: [0.14587, -0.11477, 0.39368]
- physics: `{"joint_type": "revolute", "axis": "+X", "lower_deg": -20.0, "upper_deg": 20.0, "mass_kg": 0.01, "spring_k_Nm_per_rad": 0.12, "damping": 0.02}`
- interaction: `{"primary_action": "toggle switch", "trigger_threshold_frac": 0.5, "triggers": ["/CashRegister_01/part_0150", "/CashRegister_01/part_0151"]}`
- notes: Toggle axis chosen perpendicular to the lever's principal_axis_label +Y, giving a plausible +X rocking pin.

### `cluster_0001`
- members (1): `/CashRegister_01/part_0001`
- real_world_part: drawer front screw or rivet
- category: `screw_or_fastener`
- size: [0.01049, 0.00385, 0.01049]  center: [0.13985, -0.19633, 0.18759]
- physics: static
- notes: Tiny 10 mm front detail is treated as a static fastener; its small size dominates over the +Z orientation cue.

### `cluster_0002`
- members (1): `/CashRegister_01/part_0002`
- real_world_part: drawer front screw or rivet
- category: `screw_or_fastener`
- size: [0.01056, 0.00385, 0.01056]  center: [-0.10002, -0.19566, 0.18759]
- physics: static
- notes: Tiny 11 mm front detail is treated as a static fastener; its small size dominates over the +Z orientation cue.

### `cluster_0004`
- members (1): `/CashRegister_01/part_0006`
- real_world_part: rear side trim rod
- category: `bezel_or_moulding`
- size: [0.00368, 0.03579, 0.00368]  center: [0.21604, 0.0935, 0.17256]
- physics: static
- notes: Static rear trim; the thin strip's principal axis +Y only describes its vertical/back placement and not a usable joint.

### `cluster_0005`
- members (1): `/CashRegister_01/part_0007`
- real_world_part: rear top post
- category: `decorative_detail`
- size: [0.00368, 0.00368, 0.03579]  center: [0.21491, 0.16081, 0.46675]
- physics: static
- notes: Static decorative post at the back; principal axis +Z indicates an upright rod rather than an obvious control.

### `cluster_0006`
- members (1): `/CashRegister_01/part_0008`
- real_world_part: rear top post
- category: `decorative_detail`
- size: [0.00368, 0.00368, 0.03579]  center: [0.21491, 0.16081, 0.57199]
- physics: static
- notes: Static decorative post at the back; principal axis +Z indicates an upright rod rather than an obvious control.

### `cluster_0007`
- members (1): `/CashRegister_01/part_0013`
- real_world_part: side screw cap
- category: `screw_or_fastener`
- size: [0.00747, 0.00745, 0.00781]  center: [0.25094, -0.01116, 0.30455]
- physics: static
- notes: Tiny 8 mm cube on the +X side is best treated as a static fastener despite principal axis +X.

### `cluster_0008`
- members (1): `/CashRegister_01/part_0017`
- real_world_part: drawer front handle bar
- category: `latch`
- size: [0.00368, 0.03579, 0.00368]  center: [0.21604, -0.13071, 0.17256]
- physics: `{"joint_type": "fixed", "mass_kg": 0.03}`
- interaction: `{"primary_action": "grasp to pull drawer", "trigger_threshold_frac": 0.2, "triggers": ["cluster_0119"]}`
- notes: Fixed handle on the front; principal axis +Y indicates its depth/placement but the associated motion belongs to the drawer on the -Y face.

### `cluster_0091`
- members (1): `/CashRegister_01/part_0129`
- real_world_part: keypad key
- category: `keypad_key`
- size: [0.00849, 0.00508, 0.00768]  center: [0.11794, -0.12424, 0.36631]
- physics: `{"joint_type": "prismatic", "axis": "+Y", "lower_m": 0.0, "upper_m": 0.005, "mass_kg": 0.005, "spring_k_N_per_m": 800.0, "damping": 1.0}`
- interaction: `{"primary_action": "press key", "trigger_threshold_frac": 0.6, "triggers": ["/CashRegister_01/part_0130", "/CashRegister_01/part_0134"]}`
- notes: Front-mounted small key/button uses inward +Y travel because it sits on the -Y face.

### `cluster_0093`
- members (1): `/CashRegister_01/part_0131`
- real_world_part: keypad key
- category: `keypad_key`
- size: [0.00849, 0.00508, 0.00768]  center: [0.11794, -0.14862, 0.32408]
- physics: `{"joint_type": "prismatic", "axis": "+Y", "lower_m": 0.0, "upper_m": 0.005, "mass_kg": 0.005, "spring_k_N_per_m": 800.0, "damping": 1.0}`
- interaction: `{"primary_action": "press key", "trigger_threshold_frac": 0.6, "triggers": ["/CashRegister_01/part_0130", "/CashRegister_01/part_0134"]}`
- notes: Front-mounted small key/button uses inward +Y travel because it sits on the -Y face.

### `cluster_0094`
- members (1): `/CashRegister_01/part_0132`
- real_world_part: keypad key
- category: `keypad_key`
- size: [0.00849, 0.00508, 0.00768]  center: [0.11794, -0.16081, 0.30297]
- physics: `{"joint_type": "prismatic", "axis": "+Y", "lower_m": 0.0, "upper_m": 0.005, "mass_kg": 0.005, "spring_k_N_per_m": 800.0, "damping": 1.0}`
- interaction: `{"primary_action": "press key", "trigger_threshold_frac": 0.6, "triggers": ["/CashRegister_01/part_0130", "/CashRegister_01/part_0134"]}`
- notes: Front-mounted small key/button uses inward +Y travel because it sits on the -Y face.

### `cluster_0100`
- members (1): `/CashRegister_01/part_0140`
- real_world_part: top receipt or display trim bar
- category: `bezel_or_moulding`
- size: [0.04548, 0.00301, 0.02226]  center: [0.16795, 0.02669, 0.57438]
- physics: static
- notes: Static top trim; the thin strip lies near the +Z face with principal axis +X, but no hinge or push cue is present.

### `cluster_0103`
- members (1): `/CashRegister_01/part_0143`
- real_world_part: right side knob
- category: `knob_dial`
- size: [0.00542, 0.01497, 0.01497]  center: [0.3726, -0.02695, 0.21745]
- physics: `{"joint_type": "revolute", "axis": "+X", "lower_deg": -180.0, "upper_deg": 180.0, "mass_kg": 0.02, "spring_k_Nm_per_rad": 0.0, "damping": 0.03}`
- interaction: `{"primary_action": "turn knob", "trigger_threshold_frac": 0.1, "triggers": ["/CashRegister_01/part_0103"]}`
- notes: Side knob rotation follows the side-face normal +X because the part sits directly on the +X face and protrudes outward there.

### `cluster_0104`
- members (1): `/CashRegister_01/part_0144`
- real_world_part: bottom rear base rod
- category: `bezel_or_moulding`
- size: [0.06454, 0.06454, 0.01272]  center: [-0.19059, 0.13774, 0.00888]
- physics: static
- notes: Static base trim; the long strip runs along principal axis +X at the -Z underside, which suggests a structural foot detail.

### `cluster_0105`
- members (1): `/CashRegister_01/part_0145`
- real_world_part: bottom rear base rod
- category: `bezel_or_moulding`
- size: [0.06454, 0.06454, 0.01272]  center: [0.22607, 0.13774, 0.00888]
- physics: static
- notes: Static base trim; the long strip runs along principal axis +X at the -Z underside, which suggests a structural foot detail.

### `cluster_0107`
- members (1): `/CashRegister_01/part_0147`
- real_world_part: top key button
- category: `action_button`
- size: [0.01064, 0.0104, 0.00488]  center: [-0.10002, -0.01015, 0.53609]
- physics: `{"joint_type": "prismatic", "axis": "-Z", "lower_m": 0.0, "upper_m": 0.005, "mass_kg": 0.005, "spring_k_N_per_m": 800.0, "damping": 1.0}`
- interaction: `{"primary_action": "press button", "trigger_threshold_frac": 0.6, "triggers": ["/CashRegister_01/part_0107"]}`
- notes: Top-face small button uses downward -Z travel because it sits on the +Z face.

### `cluster_0108`
- members (1): `/CashRegister_01/part_0148`
- real_world_part: top key button
- category: `action_button`
- size: [0.01064, 0.0104, 0.00488]  center: [0.01806, -0.01015, 0.53609]
- physics: `{"joint_type": "prismatic", "axis": "-Z", "lower_m": 0.0, "upper_m": 0.005, "mass_kg": 0.005, "spring_k_N_per_m": 800.0, "damping": 1.0}`
- interaction: `{"primary_action": "press button", "trigger_threshold_frac": 0.6, "triggers": ["/CashRegister_01/part_0108"]}`
- notes: Top-face small button uses downward -Z travel because it sits on the +Z face.

### `cluster_0109`
- members (1): `/CashRegister_01/part_0149`
- real_world_part: front selector lever
- category: `switch_toggle`
- size: [0.0031, 0.02855, 0.02067]  center: [0.1025, -0.10856, 0.41799]
- physics: `{"joint_type": "revolute", "axis": "+X", "lower_deg": -25.0, "upper_deg": 25.0, "mass_kg": 0.01, "spring_k_Nm_per_rad": 0.15, "damping": 0.03}`
- interaction: `{"primary_action": "toggle selector", "trigger_threshold_frac": 0.5, "triggers": ["/CashRegister_01/part_0109"]}`
- notes: Toggle axis is perpendicular to the lever's principal_axis_label +Y, so +X is used as the rocking direction.

### `cluster_0112`
- members (1): `/CashRegister_01/part_0154`
- real_world_part: rear screw cap
- category: `screw_or_fastener`
- size: [0.01303, 0.00471, 0.01303]  center: [-0.08099, 0.16599, 0.47908]
- physics: static
- notes: Small back-face detail is treated as a static fastener; the +X principal axis does not outweigh its tiny slab size.

### `cluster_0113`
- members (1): `/CashRegister_01/part_0155`
- real_world_part: rear screw cap
- category: `screw_or_fastener`
- size: [0.01284, 0.00471, 0.01284]  center: [0.12581, 0.16599, 0.47908]
- physics: static
- notes: Small back-face detail is treated as a static fastener; the +X principal axis does not outweigh its tiny slab size.

### `cluster_0114`
- members (1): `/CashRegister_01/part_0156`
- real_world_part: rear screw cap
- category: `screw_or_fastener`
- size: [0.01299, 0.00471, 0.01299]  center: [-0.08099, 0.16599, 0.19452]
- physics: static
- notes: Small back-face detail is treated as a static fastener; the +X principal axis does not outweigh its tiny slab size.

### `cluster_0115`
- members (1): `/CashRegister_01/part_0157`
- real_world_part: rear screw cap
- category: `screw_or_fastener`
- size: [0.01293, 0.00471, 0.01293]  center: [0.12581, 0.16599, 0.19452]
- physics: static
- notes: Small back-face detail is treated as a static fastener; the +X principal axis does not outweigh its tiny slab size.

### `cluster_0122`
- members (1): `/CashRegister_01/part_0170`
- real_world_part: rear side plate
- category: `bezel_or_moulding`
- size: [0.00472, 0.01303, 0.01303]  center: [-0.18213, 0.13154, 0.19251]
- physics: static
- notes: Static small back-side plate; its placement on the +Y rear face and principal axis +Z suggest trim rather than a moving mechanism.
