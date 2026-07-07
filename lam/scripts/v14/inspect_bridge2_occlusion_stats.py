"""
Inspect Bridge-2 occlusion dataset — detailed per-slot per-level stats.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/inspect_bridge2_occlusion_stats.py \
      --dataset_dir data/bridgebench/bridge2_robustness/light \
      --output result/v14/bridge2_robustness/light/stats_detailed.json
"""
import argparse, json, os, sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    val_dir = os.path.join(args.dataset_dir, "val")
    if not os.path.isdir(val_dir):
        print(f"ERROR: val dir not found: {val_dir}"); sys.exit(1)

    files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".pt")])
    if args.max_samples > 0:
        files = files[:args.max_samples]
    print(f"Scanning {len(files)} files from {val_dir}")

    stats = {"num_samples": 0, "valid_object_count": 0, "visible_valid_count": 0}
    all_occlusion = []
    all_visible_area = []
    all_full_area = []
    actions_count = {a: 0 for a in range(5)}
    per_slot_occ = {k: [] for k in range(4)}
    per_slot_vis = {k: [] for k in range(4)}
    empty_full = 0; empty_visible = 0

    for fp in files:
        s = torch.load(fp, map_location="cpu", weights_only=False)
        T, K = s["valid"].shape
        occ = s.get("occlusion_ratio", torch.zeros(T, K))
        masks = s.get("masks", s.get("full_masks", torch.zeros(T, K, 1, 1)))
        vis = s.get("visible_masks", masks)
        actions = s.get("actions", torch.full((max(T-1,0), K), -1))

        for t in range(T):
            for k in range(K):
                if not s["valid"][t, k]: continue
                stats["valid_object_count"] += 1
                r = float(occ[t, k])
                all_occlusion.append(r)
                fa = masks[t, k].sum().item()
                va = vis[t, k].sum().item()
                all_full_area.append(fa)
                all_visible_area.append(va)
                per_slot_occ[k].append(r)
                if s.get("visible_valid", torch.ones(T, K, dtype=torch.bool))[t, k]:
                    stats["visible_valid_count"] += 1
                    per_slot_vis[k].append(1)
                else:
                    per_slot_vis[k].append(0)
                if fa < 1: empty_full += 1
                if va < 1: empty_visible += 1
        for t in range(T - 1):
            for k in range(K):
                if actions[t, k] >= 0:
                    actions_count[int(actions[t, k])] += 1
        stats["num_samples"] += 1

    occ_arr = np.array(all_occlusion)
    vis_arr = np.array(all_visible_area)
    full_arr = np.array(all_full_area)

    result = {
        "num_samples": stats["num_samples"],
        "valid_object_count": stats["valid_object_count"],
        "visible_valid_count": stats["visible_valid_count"],
        "percent_occluded": 100 * sum(1 for r in all_occlusion if r > 0.05) / max(len(all_occlusion), 1),
        "percent_heavy": 100 * sum(1 for r in all_occlusion if r > 0.4) / max(len(all_occlusion), 1),
        "mean_occlusion_ratio": float(occ_arr.mean()),
        "median_occlusion_ratio": float(np.median(occ_arr)),
        "p75_occlusion_ratio": float(np.percentile(occ_arr, 75)),
        "p90_occlusion_ratio": float(np.percentile(occ_arr, 90)),
        "std_occlusion_ratio": float(occ_arr.std()),
        "mean_visible_area": float(vis_arr.mean()),
        "mean_full_area": float(full_arr.mean()),
        "visible_area_ratio_mean": float((vis_arr / (full_arr + 1e-6)).mean()),
        "visible_area_ratio_median": float(np.median(vis_arr / (full_arr + 1e-6))),
        "empty_full_mask_count": empty_full,
        "empty_visible_mask_count": empty_visible,
        "action_distribution": actions_count,
        "per_slot_occlusion_mean": {str(k): float(np.mean(v)) if v else 0 for k, v in per_slot_occ.items()},
        "per_slot_visible_valid_ratio": {str(k): float(np.mean(v)) if v else 0 for k, v in per_slot_vis.items()},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")
    print(f"  occ%={result['percent_occluded']:.1f} heavy%={result['percent_heavy']:.1f} "
          f"mean={result['mean_occlusion_ratio']:.3f} median={result['median_occlusion_ratio']:.3f}")
    print(f"  vis_area_mean={result['mean_visible_area']:.0f} full_mean={result['mean_full_area']:.0f} "
          f"vis_ratio={result['visible_area_ratio_mean']:.3f}")


if __name__ == "__main__":
    main()
