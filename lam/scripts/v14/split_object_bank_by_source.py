"""
Split object bank by source_video for unseen-source evaluation.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/split_object_bank_by_source.py \
      --object_bank data/bridgebench/object_bank_clean.pt \
      --train_output data/bridgebench/object_bank_clean_train.pt \
      --test_output data/bridgebench/object_bank_clean_test.pt \
      --test_ratio 0.2 --seed 0
"""
import argparse, json, os, sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_bank", default="data/bridgebench/object_bank_clean.pt")
    parser.add_argument("--train_output", default="data/bridgebench/object_bank_clean_train.pt")
    parser.add_argument("--test_output", default="data/bridgebench/object_bank_clean_test.pt")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats_output", default="data/bridgebench/object_bank_source_split_stats.json")
    args = parser.parse_args()

    bank = torch.load(args.object_bank, map_location="cpu", weights_only=False)
    print(f"Loaded {len(bank)} objects")

    # Group by source_video.
    sources = {}
    for o in bank:
        vid = o.get("source_video", "unknown")
        sources.setdefault(vid, []).append(o)

    video_ids = list(sources.keys())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(video_ids)
    n_test_vids = max(1, int(len(video_ids) * args.test_ratio))
    test_vids = set(video_ids[:n_test_vids])
    train_vids = set(video_ids[n_test_vids:])

    train_bank = [o for o in bank if o.get("source_video", "unknown") in train_vids]
    test_bank = [o for o in bank if o.get("source_video", "unknown") in test_vids]

    print(f"Train: {len(train_bank)} objects from {len(train_vids)} videos")
    print(f"Test:  {len(test_bank)} objects from {len(test_vids)} videos")
    assert len(train_vids & test_vids) == 0, "Source overlap detected!"

    torch.save(train_bank, args.train_output)
    torch.save(test_bank, args.test_output)
    print(f"Saved to {args.train_output}, {args.test_output}")

    # Stats.
    def cat_dist(bk):
        d = {}
        for o in bk:
            d[o["category"]] = d.get(o["category"], 0) + 1
        return d

    stats = {
        "train_num_objects": len(train_bank),
        "test_num_objects": len(test_bank),
        "train_source_count": len(train_vids),
        "test_source_count": len(test_vids),
        "source_overlap_count": 0,
        "train_category_distribution": cat_dist(train_bank),
        "test_category_distribution": cat_dist(test_bank),
        "split_method": "source_video",
        "strict_source_disjoint": True,
    }
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats → {args.stats_output}")


if __name__ == "__main__":
    main()
