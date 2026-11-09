"""Patch cosmos-predict2.5 so LoRA/EMA checkpoints actually load into a PEFT + SAC-wrapped DiT.

The loader did not strip the `_checkpoint_wrapped_module.` (selective activation checkpointing)
and PEFT `base_model.model.` / `.base_layer.` prefixes, so every LoRA target was silently
unmatched and the DiT started from random weights. This inserts a key remap (plus a small
probe log) right before `state_dict = _reg_state_dict` in
`text2world_model_rectified_flow.py`. Edit `P` to point at your cosmos-predict2.5 checkout.
"""
# (note: launcher --config arg must point at video2world/config.py, not the experiment file)
P = "/root/cosmos-predict2.5/cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py"

PATCH = '''        # [v2 patch] Remap ckpt keys to PEFT+SAC-wrapped model keys + probe.
        from cosmos_predict2._src.imaginaire.utils import log
        def _remap(sd, target_module, tag):
            model_keys = list(target_module.state_dict().keys())
            rmap = {}
            for mk in model_keys:
                stripped = (mk.replace("base_model.model.", "")
                              .replace("_checkpoint_wrapped_module.", "")
                              .replace(".base_layer.", "."))
                rmap[stripped] = mk
            with open("/root/remap_probe.txt", "a") as fh:
                fh.write("=== " + tag + " probe ===\\n")
                for mk in model_keys[:8]:
                    fh.write("  model: " + mk + "\\n")
                for sk in list(sd.keys())[:8]:
                    fh.write("  ckpt:  " + sk + "\\n")
                fh.write("  rmap_size=" + str(len(rmap)) + " sd_size=" + str(len(sd)) + " model_keys=" + str(len(model_keys)) + "\\n")
            out = {}
            mapped = 0
            for k, v in sd.items():
                if k in rmap:
                    out[rmap[k]] = v; mapped += 1
                else:
                    out[k] = v
            log.critical("[v2 remap " + tag + "] mapped " + str(mapped) + "/" + str(len(sd)))
            return out
        _reg_state_dict = _remap(_reg_state_dict, self.net, "reg")
        if self.config.ema.enabled:
            _ema_state_dict = _remap(_ema_state_dict, self.net_ema, "ema")

'''

NEEDLE = "        state_dict = _reg_state_dict\n"

s = open(P).read()
# Strip any prior broken patch first by restoring from .v2bak if present
import os
bak = P + ".v2bak"
if os.path.exists(bak):
    s = open(bak).read()
    open(P, "w").write(s)
assert NEEDLE in s, "needle missing"
s2 = s.replace(NEEDLE, PATCH + NEEDLE, 1)
assert s2 != s
open(P, "w").write(s2)
print("OK applied")
