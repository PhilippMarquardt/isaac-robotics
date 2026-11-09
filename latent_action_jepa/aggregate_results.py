"""Walk every metrics.json under /workspace/latent_action_jepa/runs and /root/jepa_runs/
and produce a single consolidated table for STAGE2_FINAL_TABLE.md.
"""
import json
import os
import sys


def find_metrics(root):
    found = []
    for dirpath, _, files in os.walk(root):
        if "metrics.json" in files:
            found.append(os.path.join(dirpath, "metrics.json"))
    return found


def main():
    roots = [
        "/workspace/latent_action_jepa/runs",
        "/root/jepa_runs",
    ]
    metrics_files = []
    for r in roots:
        if os.path.exists(r):
            metrics_files.extend(find_metrics(r))
    print(f"# Stage 2 — consolidated metrics across {len(metrics_files)} runs\n")
    for path in sorted(metrics_files):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"## {path}\n*ERROR reading: {e}*\n")
            continue
        rel = path.replace("/workspace/latent_action_jepa/", "").replace("/root/", "ROOT/")
        print(f"## {rel}\n")

        def fmt_metrics(m, prefix=""):
            if not isinstance(m, dict):
                return
            top1 = m.get("top1")
            if isinstance(top1, (int, float)):
                key = prefix.rstrip(" /")
                print(f"  - {key}: top1 {m.get('top1',0):.3f}  top5 {m.get('top5',0):.3f}  mr {m.get('mean_rank',0):.2f}")
            else:
                for k, v in m.items():
                    fmt_metrics(v, prefix + k + " / ")

        if "results" in d:
            for split, sec in d["results"].items():
                print(f"### {split}")
                fmt_metrics(sec)
                print()
        else:
            for key in ["pre_train", "pre_val", "post_train", "post_val", "train", "val"]:
                if key in d:
                    print(f"### {key}")
                    fmt_metrics(d[key])
                    print()
        if "args" in d:
            args = d["args"]
            interesting = ["prior_arch", "prior_depth", "prior_dropout", "flow_depth", "ks",
                           "K_pool", "steps", "bs", "lr", "wd", "sample_steps",
                           "lambda_nce", "lambda_z_distill"]
            shown = {k: args[k] for k in interesting if k in args}
            if shown:
                print(f"  cfg: {shown}\n")


if __name__ == "__main__":
    main()
