"""
Summarize Bridge-2 multi-seed results.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/summarize_bridge2_multiseed.py \
      --input result/v14/bridge2_occlusion_clean/multiseed/ \
      --output result/v14/bridge2_occlusion_clean/multiseed/summary.json
"""
import argparse, json, os, sys, math
from typing import Dict, List

import numpy as np


def load_eval(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seeds = [0, 1, 2]
    models = ["mask_structure", "bbox_only"]

    all_data = {}
    for model in models:
        for seed in seeds:
            eval_path = os.path.join(args.input, f"seed{seed}", model, "eval", "eval.json")
            if os.path.exists(eval_path):
                all_data[(model, seed)] = load_eval(eval_path)
            else:
                print(f"  Missing: {eval_path}")

    if len(all_data) < len(seeds) * len(models):
        print(f"  Found {len(all_data)}/{len(seeds)*len(models)} eval files. Some missing.")
        if len(all_data) < 2:
            sys.exit(1)

    # Metrics to aggregate.
    metric_keys = [
        "normal_iou", "z_zero_iou", "z_shuffle_iou",
        "bbox_gap_zero", "bbox_gap_shuffle",
        "normal_dice_warp", "z_zero_dice_warp",
        "mask_gap_warp", "mask_warp_gap",
        "action_probe_acc", "swap_acc_ns", "swap_acc_cs_ns",
        "actor_probe_acc", "category_probe_acc",
        "z_var", "overall_nmi", "per_slot_nmi_avg",
        "oracle_identity_dice_h", "oracle_gt_warp_dice_h",
        "z_shuffle_order_ok",
    ]

    summary = {}
    for model in models:
        model_data = {k: [] for k in metric_keys}
        for seed in seeds:
            key = (model, seed)
            if key not in all_data:
                continue
            d = all_data[key]
            for mk in metric_keys:
                val = d.get(mk, None)
                if val is not None:
                    model_data[mk].append(float(val) if not isinstance(val, bool) else val)

        stats = {}
        for mk, vals in model_data.items():
            if not vals:
                continue
            if isinstance(vals[0], bool):
                stats[mk] = sum(vals)
            else:
                stats[mk] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "values": vals,
                }
        summary[model] = stats

    # GO checks.
    # Extract values using either mask_gap_warp or mask_warp_gap.
    def _get(model, key):
        s = summary.get(model, {})
        if key in s: return s[key]
        if key == "mask_warp_gap" and "mask_gap_warp" in s: return s["mask_gap_warp"]
        return {}

    mask_stats = summary.get("mask_structure", {})
    go = {
        "mask_bbox_gap": _get("mask_structure", "bbox_gap_zero").get("mean", 0) > 0.20,
        "mask_warp_gap": _get("mask_structure", "mask_warp_gap").get("mean", 0) > 0.20,
        "mask_action_probe": _get("mask_structure", "action_probe_acc").get("mean", 0) > 0.90,
        "mask_swap_ns": _get("mask_structure", "swap_acc_ns").get("mean", 0) > 0.85,
        "z_shuffle_2of3": _get("mask_structure", "z_shuffle_order_ok") >= 2,
    }
    go["mask_go"] = all(go[k] for k in ["mask_bbox_gap", "mask_warp_gap",
                                          "mask_action_probe", "mask_swap_ns", "z_shuffle_2of3"])

    # Mask advantage over bbox.
    bbox_stats = summary.get("bbox_only", {})
    mask_niou = mask_stats.get("normal_iou", {}).get("mean", 0)
    bbox_niou = bbox_stats.get("normal_iou", {}).get("mean", 0)
    go["mask_advantage_iou"] = mask_niou > bbox_niou + 0.20
    go["multiseed_go"] = go.get("mask_go", False) and go.get("mask_advantage_iou", False)

    summary["go"] = go

    # Save.
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary → {args.output}")

    # Print summary.
    print(f"\n{'='*70}")
    print(f"  Bridge-2 Multi-Seed Summary ({len(seeds)} seeds)")
    print(f"{'='*70}")

    mk_labels = [
        ("normal_iou", "Box IoU (normal)"),
        ("bbox_gap_zero", "Bbox Gap (normal-zero)"),
        ("mask_warp_gap", "Mask Warp Gap"),
        ("normal_dice_warp", "Mask Warp Dice (normal)"),
        ("action_probe_acc", "Action Probe"),
        ("swap_acc_ns", "Swap Acc (non-stay)"),
        ("per_slot_nmi_avg", "Per-Slot NMI Avg"),
        ("z_var", "z Variance"),
    ]
    header = f"{'Metric':<25} {'Mask mean±std':<22} {'Bbox mean±std':<22}"
    print(f"\n  {header}")
    print(f"  {'-'*65}")
    for mk, label in mk_labels:
        m = _get("mask_structure", mk)
        b = _get("bbox_only", mk)
        mm = f"{m.get('mean',0):.3f}±{m.get('std',0):.3f}" if m else "N/A"
        bb = f"{b.get('mean',0):.3f}±{b.get('std',0):.3f}" if b else "N/A"
        print(f"  {label:<25} {mm:<22} {bb:<22}")

    print(f"\n  GO Checks:")
    for k, v in go.items():
        print(f"    {'✓' if v else '✗'} {k}")
    print(f"\n  Multiseed GO: {'✓ PASS' if go.get('multiseed_go', False) else '✗ FAIL'}")

    # Markdown.
    md_path = os.path.join(args.input, "summary.md") if args.input else None
    if md_path:
        with open(md_path, "w") as f:
            f.write("# Bridge-2 Multi-Seed Summary\n\n")
            f.write(f"Seeds: {seeds}\n\n")
            f.write(f"| Metric | Mask mean±std | Bbox mean±std |\n")
            f.write(f"|---|---|---|\n")
            for mk, label in mk_labels:
                m = mask_stats.get(mk, {})
                b = bbox_stats.get(mk, {})
                mm = f"{m.get('mean',0):.3f}±{m.get('std',0):.3f}" if m else "N/A"
                bb = f"{b.get('mean',0):.3f}±{b.get('std',0):.3f}" if b else "N/A"
                f.write(f"| {label} | {mm} | {bb} |\n")
            f.write(f"\n## Per-Seed GO\n\n")
            f.write(f"| Seed | Mask bbox_gap | Mask action_probe | Mask swap_ns | Bbox gap |\n")
            f.write(f"|---|---|---|---|---|\n")
            for seed in seeds:
                mk = mask_stats.get("bbox_gap_zero", {}).get("values", [])
                ma = mask_stats.get("action_probe_acc", {}).get("values", [])
                ms = mask_stats.get("swap_acc_ns", {}).get("values", [])
                bk = bbox_stats.get("bbox_gap_zero", {}).get("values", [])
                v = lambda lst, i: f"{lst[i]:.3f}" if i < len(lst) else "N/A"
                f.write(f"| {seed} | {v(mk,seed)} | {v(ma,seed)} | {v(ms,seed)} | {v(bk,seed)} |\n")
            f.write(f"\n## GO\n\n")
            f.write(f"Multiseed GO: **{'PASS' if go.get('multiseed_go',False) else 'FAIL'}**\n\n")
            for k, v in go.items():
                f.write(f"- {'✓' if v else '✗'} {k}\n")
        print(f"  Summary.md → {md_path}")


if __name__ == "__main__":
    main()
