"""
BridgeBench Bridge-1 generator: real object crops + controlled sprite motion.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/generate_bridgebench.py \
      --object_bank data/bridgebench/object_bank.pt \
      --out data/bridgebench/bridge1 \
      --n_train 5000 --n_val 500 \
      --image_size 128 --num_objects 4 --step_size 8
"""
import argparse, os, sys

import numpy as np
import torch
import torch.nn.functional as F

ACTION_DELTA = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ACTION_NAMES = ["stay", "up", "down", "left", "right"]


def _paste_crop(canvas, crop, mask, cx, cy, cw, ch, H, W):
    """Paste object crop at (cx,cy) with size (cw,ch) using mask alpha.
    canvas: (H, W, 3) uint8, crop: (3, cs, cs) float [0,1], mask: (1, cs, cs) float.
    cx,cy,cw,ch in pixel coordinates.
    """
    cs = crop.shape[1]
    x1 = int(cx - cw / 2)
    y1 = int(cy - ch / 2)
    x2 = int(cx + cw / 2)
    y2 = int(cy + ch / 2)
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return canvas

    crop_resized = F.interpolate(crop.unsqueeze(0), size=(ph, pw), mode="bilinear",
                                 align_corners=False).squeeze(0)  # (3, ph, pw)
    mask_resized = F.interpolate(mask.unsqueeze(0), size=(ph, pw), mode="bilinear",
                                 align_corners=False).squeeze(0)  # (1, ph, pw)
    alpha = mask_resized.clamp(0, 1)  # (1, ph, pw)
    patch = (crop_resized * 255).clamp(0, 255).byte()

    for c in range(3):
        canvas_c = canvas[y1:y2, x1:x2, c].astype(np.float32)
        alpha_np = alpha[0].numpy()
        patch_np = patch[c].numpy().astype(np.float32)
        blended = canvas_c * (1 - alpha_np) + patch_np * alpha_np
        canvas[y1:y2, x1:x2, c] = np.clip(blended, 0, 255).astype(np.uint8)

    return canvas


def _generate_sample(bank_objects, rng, image_size, num_objects, T, step_size):
    H = W = image_size
    cs = bank_objects[0]["image_crop"].shape[1]

    # Pick K random objects of different categories if possible.
    cats = list(set(o["category"] for o in bank_objects))
    picked_cats = rng.choice(cats, size=min(num_objects, len(cats)), replace=False)
    picked = []
    for cat in picked_cats:
        cat_objs = [o for o in bank_objects if o["category"] == cat]
        picked.append(cat_objs[rng.randint(len(cat_objs))])
    while len(picked) < num_objects:
        picked.append(bank_objects[rng.randint(len(bank_objects))])
    rng.shuffle(picked)

    # Assign initial positions (quadrant-based, no overlap).
    positions = np.zeros((T, num_objects, 2), dtype=np.float32)
    init_offsets = [
        (W * 0.25, H * 0.25), (W * 0.75, H * 0.25),
        (W * 0.25, H * 0.75), (W * 0.75, H * 0.75),
    ]
    for k in range(min(num_objects, 4)):
        positions[0, k] = init_offsets[k]

    obj_sizes = np.zeros((num_objects, 2), dtype=np.float32)
    for k, obj in enumerate(picked):
        w_h = obj["bbox"][2:4].numpy()
        obj_sizes[k] = w_h * image_size  # pixel size

    actions = np.zeros((T - 1, num_objects), dtype=np.int64)
    for k in range(num_objects):
        px, py = positions[0, k]
        for t in range(1, T):
            act = rng.randint(0, 5)
            actions[t - 1, k] = act
            dx, dy = ACTION_DELTA[act]
            px += dx * step_size
            py += dy * step_size
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))
            positions[t, k] = [px, py]

    # Generate frames.
    videos = np.zeros((T, 3, H, W), dtype=np.uint8)
    masks = np.zeros((T, num_objects, H, W), dtype=np.uint8)
    boxes = np.zeros((T, num_objects, 4), dtype=np.float32)  # cxcywh norm
    valid = np.ones((T, num_objects), dtype=bool)

    bg_color = rng.randint(60, 200, (3,)).astype(np.uint8)  # random gray bg

    for t in range(T):
        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)
        for k, obj in enumerate(picked):
            px, py = positions[t, k]
            ow, oh = obj_sizes[k]
            canvas = _paste_crop(canvas,
                                 obj["image_crop"], obj["mask_crop"],
                                 px, py, ow, oh, H, W)
            # Save mask.
            x1 = int(px - ow / 2)
            y1 = int(py - oh / 2)
            x2 = int(px + ow / 2)
            y2 = int(py + oh / 2)
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            if x2 > x1 and y2 > y1:
                m = F.interpolate(obj["mask_crop"].unsqueeze(0), size=(y2 - y1, x2 - x1),
                                  mode="bilinear", align_corners=False).squeeze(0)
                masks[t, k, y1:y2, x1:x2] = (m[0].clamp(0, 1) * 255).byte().numpy()
            boxes[t, k] = [px / W, py / H, ow / W, oh / H]
        videos[t] = torch.from_numpy(canvas).permute(2, 0, 1).numpy()  # (3, H, W)

    # visible masks = full masks (no occlusion in Bridge-1)
    visible_masks = masks.copy()
    full_masks = masks.copy()

    # Unique actor IDs.
    actor_ids = np.arange(num_objects, dtype=np.int64)
    categories = np.array([picked[k].get("category_id", k) for k in range(num_objects)],
                          dtype=np.int64)

    sample = {
        "video": torch.from_numpy(videos).float() / 255.0,  # (T, 3, H, W)
        "masks": torch.from_numpy(masks.astype(np.float32)) / 255.0,
        "visible_masks": torch.from_numpy(visible_masks.astype(np.float32)) / 255.0,
        "full_masks": torch.from_numpy(full_masks.astype(np.float32)) / 255.0,
        "boxes": torch.from_numpy(boxes),
        "valid": torch.from_numpy(valid),
        "actions": torch.from_numpy(actions),
        "actor_id": torch.from_numpy(actor_ids),
        "category": torch.from_numpy(categories),
        "depth_order": torch.from_numpy(np.tile(np.arange(num_objects), (T, 1)).astype(np.int64)),
    }
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_bank", default="data/bridgebench/object_bank.pt")
    parser.add_argument("--out", default="data/bridgebench/bridge1")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_objects", type=int, default=4)
    parser.add_argument("--step_size", type=int, default=8)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.object_bank):
        print(f"ERROR: object bank not found at {args.object_bank}")
        sys.exit(1)

    bank = torch.load(args.object_bank, map_location="cpu", weights_only=False)
    # Assign numeric category IDs.
    cat_set = sorted(set(o["category"] for o in bank))
    cat_to_id = {c: i for i, c in enumerate(cat_set)}
    for o in bank:
        o["category_id"] = cat_to_id[o["category"]]
    print(f"Loaded {len(bank)} objects, {len(cat_set)} categories: {cat_set}")

    rng = np.random.RandomState(args.seed)

    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n):
            sample = _generate_sample(bank, rng, args.image_size, args.num_objects,
                                      args.T, args.step_size)
            torch.save(sample, os.path.join(out_dir, f"sample_{i:06d}.pt"))
        print(f"  Saved {n} samples to {out_dir}")

    print(f"Bridge-1 dataset ready: {args.out}")


if __name__ == "__main__":
    main()
