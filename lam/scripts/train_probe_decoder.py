"""
V8 Probe Decoder 训练: 冻结 z_actor, 训练 ActorProbeDecoder 重建下一帧 crop.

关键设计:
  - z_actor 冻结 (不反传到主模型)
  - Video-grouped 80/20 split (同一视频的所有 slot/time 只在 train 或 test)
  - Loss: L1 + 0.5 * (1 - SSIM)
  - 早停: test PSNR 连续 patience 个 epoch 不涨则停止
  - 保存最优 checkpoint

用法:
  PYTHONPATH=lam python lam/scripts/train_probe_decoder.py --name v8_stage1
  PYTHONPATH=lam python lam/scripts/train_probe_decoder.py --name v8_stage1 --use_dbbox
"""
import os, sys, json, time, argparse
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.probe_decoder import ActorProbeDecoder, compute_psnr, compute_ssim_simple


def video_grouped_split(video_ids, train_ratio=0.8, seed=42):
    """按 video_id 分组 split, 同一视频的所有样本只在 train 或 test."""
    unique_videos = np.unique(video_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_videos)
    n_train = int(len(unique_videos) * train_ratio)
    train_videos = set(unique_videos[:n_train])
    train_idx = np.array([i for i, v in enumerate(video_ids) if v in train_videos])
    test_idx = np.array([i for i, v in enumerate(video_ids) if v not in train_videos])
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="latent_pairs name (e.g. v8_stage1)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--use_dbbox", action="store_true", help="mode B: 加 dbbox_pred")
    parser.add_argument("--num_actor_types", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=200, help="早停 patience (steps)")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam"
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    suffix = "_dbbox" if args.use_dbbox else ""
    probe_name = f"probe_{args.name}{suffix}"

    print(f"\n{'='*60}")
    print(f"Train ActorProbeDecoder: {probe_name}")
    print(f"  device={device}, use_dbbox={args.use_dbbox}")
    print(f"{'='*60}")

    data = np.load(os.path.join(results_dir, f"latent_pairs_{args.name}.npz"))
    z_actor = data["z_actor"]
    crop_t = data["crop_t"]
    crop_gt = data["crop_tp1_gtbox"]
    video_ids = data["video_id"]
    dbbox_pred = data["dbbox_pred"] if "dbbox_pred" in data else None
    actor_types = data["actor_types"] if "actor_types" in data else None

    print(f"  Total samples: {len(z_actor)}")
    print(f"  Videos: {len(np.unique(video_ids))}")

    train_idx, test_idx = video_grouped_split(video_ids, seed=args.seed)
    print(f"  Train: {len(train_idx)} (videos: {len(np.unique(video_ids[train_idx]))})")
    print(f"  Test:  {len(test_idx)} (videos: {len(np.unique(video_ids[test_idx]))})")

    def make_tensors(idx):
        z = torch.from_numpy(z_actor[idx]).float().to(device)
        ct = torch.from_numpy(crop_t[idx]).float().to(device)
        cg = torch.from_numpy(crop_gt[idx]).float().to(device)
        db = torch.from_numpy(dbbox_pred[idx]).float().to(device) if dbbox_pred is not None else None
        at = torch.from_numpy(actor_types[idx]).long().to(device) if actor_types is not None else None
        return z, ct, cg, db, at

    z_tr, ct_tr, cg_tr, db_tr, at_tr = make_tensors(train_idx)
    z_te, ct_te, cg_te, db_te, at_te = make_tensors(test_idx)

    model = ActorProbeDecoder(
        z_dim=args.z_dim, crop_size=args.crop_size,
        use_dbbox=args.use_dbbox, num_actor_types=args.num_actor_types,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Probe params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_psnr = -1
    best_step = 0
    steps_since_best = 0
    train_losses = []
    test_psnrs = []

    t0 = time.time()
    n_train = len(train_idx)

    for step in range(args.max_steps):
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            z_b = z_tr[batch_idx]
            ct_b = ct_tr[batch_idx]
            cg_b = cg_tr[batch_idx]
            db_b = db_tr[batch_idx] if db_tr is not None else None
            at_b = at_tr[batch_idx] if at_tr is not None else None

            pred = model(ct_b, z_b, dbbox_pred=db_b, actor_type=at_b)
            l1 = F.l1_loss(pred, cg_b)
            ssim_loss = 1.0 - compute_ssim_simple(pred, cg_b)
            loss = l1 + 0.5 * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))

        if (step + 1) % args.eval_every == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                pred_te = model(ct_te, z_te, dbbox_pred=db_te, actor_type=at_te)
                test_psnr = compute_psnr(pred_te, cg_te)
            model.train()

            test_psnrs.append({"step": step + 1, "psnr": test_psnr})
            elapsed = time.time() - t0
            avg_loss = np.mean(train_losses[-n_train // args.batch_size:]) if train_losses else 0

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                best_step = step + 1
                steps_since_best = 0
                torch.save(model.state_dict(),
                           os.path.join(results_dir, f"model_{probe_name}.pt"))
            else:
                steps_since_best += args.eval_every

            print(f"  Step {step+1:4d}/{args.max_steps}: "
                  f"loss={avg_loss:.4f}, test_psnr={test_psnr:.2f} dB, "
                  f"best={best_psnr:.2f} (step {best_step}), "
                  f"patience={steps_since_best}/{args.patience}, {elapsed:.0f}s")

            if steps_since_best >= args.patience:
                print(f"  Early stop at step {step+1}")
                break

    train_time = time.time() - t0
    print(f"\n  Training done: best PSNR={best_psnr:.2f} dB (step {best_step}), {train_time:.0f}s")

    np.savetxt(os.path.join(results_dir, f"loss_{probe_name}_train.txt"),
               np.array(train_losses))
    with open(os.path.join(results_dir, f"test_psnrs_{probe_name}.json"), "w") as f:
        json.dump(test_psnrs, f, indent=2)

    results = {
        "probe_name": probe_name,
        "base_model": args.name,
        "use_dbbox": args.use_dbbox,
        "best_psnr": round(best_psnr, 4),
        "best_step": best_step,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_videos": len(np.unique(video_ids[train_idx])),
        "n_test_videos": len(np.unique(video_ids[test_idx])),
        "total_steps": step + 1,
        "training_time_s": round(train_time, 1),
        "probe_params": total_params,
    }
    with open(os.path.join(results_dir, f"results_{probe_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Model saved: {results_dir}/model_{probe_name}.pt")
    print(f"  Results saved: {results_dir}/results_{probe_name}.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
