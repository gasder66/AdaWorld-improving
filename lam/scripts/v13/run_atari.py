"""
V13 Atari BBox training script — Phase B / C.

Usage:
  PYTHONPATH=lam python lam/scripts/v13/run_atari.py \
      --game freeway --phase B --batch_size 8 --steps 5000 --gpu 4

Output: result/v13/{game}/{phase}/
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.datasets.atari_bbox_dataset import AtariBBoxDataset, GAME_CONFIGS
from lam.modules.v13_slot_builder import build_slot_builder
from lam.modules.v13_atari_model import V13AtariBBoxModel

VERSION = "v13"


def collate_atari(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            out[k] = vals[0]
        else:
            out[k] = torch.tensor(vals) if vals[0] is not None else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--phase", type=str, default="B", choices=["B", "C"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    cfg = GAME_CONFIGS[args.game]
    image_size = cfg["image_size"]

    if args.data_root is None:
        data_root = os.path.join(ROOT, "data", "v12_ocatari", f"ocatari_{args.game}")
    else:
        data_root = args.data_root

    RESULTS_DIR = os.path.join(ROOT, "result", VERSION, args.game, f"phase{args.phase}")
    os.makedirs(os.path.join(RESULTS_DIR, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "losses"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V13-A: AtariBBox Structure-Action World Model")
    print(f"  game={args.game}, phase={args.phase}, gpu={args.gpu}")
    print(f"  result => {RESULTS_DIR}")
    print(f"  data: {data_root}")
    print(f"{'='*60}")

    train_ds = AtariBBoxDataset(os.path.join(data_root, "train"), game=args.game, image_size=image_size)
    print(f"  Train samples: {len(train_ds)}")

    sb = build_slot_builder(args.game)
    model = V13AtariBBoxModel(slot_builder=sb, image_size=image_size).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate_atari,
    )

    losses = {"total": [], "struct": [], "kl": [], "recon": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            outputs = model(batch, phase=args.phase)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["struct"].append(float(outputs.get("struct_loss", 0)))
            losses["kl"].append(float(outputs.get("kl_loss", 0)))
            losses["recon"].append(float(outputs.get("recon_loss", 0)))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                parts = [f"loss={float(loss):.4f}"]
                parts.append(f"struct={losses['struct'][-1]:.4f}")
                parts.append(f"kl={losses['kl'][-1]:.4f}")
                if args.phase == "C":
                    parts.append(f"recon={losses['recon'][-1]:.4f}")
                parts.append(f"mem={mem:.1f}GB")
                print(f"  Step {step:4d}/{args.steps}: " + ", ".join(parts))

            if (step + 1) % args.checkpoint_every == 0:
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "ckpts", f"step{step+1}.pt"))

            step += 1

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  Done. time={elapsed:.0f}s, mem={mem:.1f}GB")

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))
    for k, v in losses.items():
        np.savetxt(os.path.join(RESULTS_DIR, "losses", f"{k}.txt"), np.array(v))

    print(f"  Saved => {RESULTS_DIR}")


if __name__ == "__main__":
    main()
