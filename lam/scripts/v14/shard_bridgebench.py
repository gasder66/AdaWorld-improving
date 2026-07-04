"""
Pack per-file BridgeBench data into shards for fast loading.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/shard_bridgebench.py \
      --input data/bridgebench/bridge1_clean \
      --output data/bridgebench/bridge1_clean_sharded \
      --shard_size 512
"""
import argparse, json, os, sys

import torch


def collate_list(samples):
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard_size", type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: input dir not found: {args.input}"); sys.exit(1)

    meta = {}
    for split in ["train", "val"]:
        split_dir = os.path.join(args.input, split)
        if not os.path.isdir(split_dir):
            print(f"  Skip {split}: not found")
            continue

        files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir)
                        if f.endswith(".pt")])
        print(f"  {split}: {len(files)} files")

        out_split_dir = os.path.join(args.output, split)
        os.makedirs(out_split_dir, exist_ok=True)

        shard_files = []
        for si in range(0, len(files), args.shard_size):
            batch_files = files[si:si + args.shard_size]
            samples = [torch.load(f, map_location="cpu", weights_only=False) for f in batch_files]
            batch = collate_list(samples)
            shard_file = f"shard_{si // args.shard_size:03d}.pt"
            torch.save(batch, os.path.join(out_split_dir, shard_file))
            shard_files.append(os.path.join(split, shard_file))
            if si % (args.shard_size * 4) == 0:
                print(f"    shard {si // args.shard_size} ({len(batch_files)} samples)")

        meta[split] = {
            "shard_files": shard_files,
            "shard_size": args.shard_size,
            "total": len(files),
        }

    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json saved. Shards: {args.output}")


if __name__ == "__main__":
    main()
