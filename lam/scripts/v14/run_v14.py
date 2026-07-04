"""
V14 BridgeBench training script — Phase A / B / C.

Usage:
  PYTHONPATH=lam python lam/scripts/v14/run_v14.py \
      --dataset bridge1 --phase B --batch_size 8 --steps 5000 --gpu 4
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.modules.v14_model import V14Model

VERSION = "v14"

BRIDGE_DEFAULTS = {
    "bridge1": {"image_size": 128, "max_actors": 4},
}


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
    return out


def _box_iou(b1, b2):
    x1 = torch.max(b1[..., 0], b2[..., 0]); y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2]); y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = b1[..., 2] * b1[..., 3]; a2 = b2[..., 2] * b2[..., 3]
    return inter / (a1 + a2 - inter + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bridge1")
    parser.add_argument("--phase", default="B", choices=["A", "B", "C"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cfg = BRIDGE_DEFAULTS[args.dataset]
    ROOT = os.path.join(os.path.dirname(__file__), "../../..")
    data_dir = args.data_root or os.path.join(ROOT, "data", "bridgebench", args.dataset)
    out_dir = os.path.join(ROOT, "result", VERSION, args.dataset, f"phase{args.phase}")
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "losses"), exist_ok=True)

    print(f"\n{'='*60}\nV14 BridgeBench\n  dataset={args.dataset}, phase={args.phase}\n  out={out_dir}\n{'='*60}")

    # Load data.
    train_dir = os.path.join(data_dir, "train")
    train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".pt")])
    print(f"  Train: {len(train_files)} samples")

    model = V14Model(image_size=cfg["image_size"], max_actors=cfg["max_actors"]).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded: {args.checkpoint}")

    pcount = sum(p.numel() for p in model.parameters())
    print(f"  Params: {pcount:,}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses = {"total": [], "struct": [], "kl": [], "recon": []}
    step, t0 = 0, time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        indices = torch.randperm(len(train_files))[:args.batch_size]
        batch_list = [torch.load(train_files[i], map_location="cpu", weights_only=False) for i in indices]
        batch = collate(batch_list)
        for k in list(batch.keys()):
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        out = model(batch, phase=args.phase)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        losses["total"].append(float(loss))
        losses["struct"].append(float(out.get("struct_loss", 0)))
        losses["kl"].append(float(out.get("kl_loss", 0)))
        losses["recon"].append(float(out.get("recon_loss", 0)))

        if step % 50 == 0:
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            parts = [f"loss={loss.item():.4f}", f"struct={losses['struct'][-1]:.4f}",
                     f"kl={losses['kl'][-1]:.4f}"]
            if args.phase == "C":
                parts.append(f"recon={losses['recon'][-1]:.4f}")
            parts.append(f"mem={mem:.1f}GB")
            print(f"  Step {step:4d}/{args.steps}: " + ", ".join(parts))

        if (step + 1) % args.checkpoint_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, "ckpts", f"step{step+1}.pt"))

        step += 1

    print(f"\n  Done. time={time.time()-t0:.0f}s")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    for k, v in losses.items():
        np.savetxt(os.path.join(out_dir, "losses", f"{k}.txt"), np.array(v))
    print(f"  Saved => {out_dir}")


if __name__ == "__main__":
    main()
