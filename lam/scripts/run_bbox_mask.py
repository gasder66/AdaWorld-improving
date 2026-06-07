"""
验证 bbox 矩形填充 mask 是否能替代 GT mask 训练 LAM。

核心假设：MaskedPool 将 mask 下采样到 patch 级别 (16×16)，
bbox 矩形填充在 patch 级别已经足够精确。

流程：
1. 从 GT mask 提取 bounding box
2. 用 bbox 矩形填充生成新 mask
3. 用新 mask 训练 LAM
4. 对比 GT mask 基线的动作准确率

用法:
  CUDA_VISIBLE_DEVICES=3 python run_bbox_mask.py --gpu 0 --name bbox_mask_baseline --steps 500
"""
import os
import sys
import json
import time
import argparse
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
NUM_ACTIONS = 5


class BBoxMaskDataset(torch.utils.data.Dataset):
    """包装 DiskSyntheticDataset，用 bbox 矩形填充替代 GT mask。"""

    def __init__(self, base_dataset, max_actors=4):
        self.base = base_dataset
        self.max_actors = max_actors

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        masks = sample["masks"]  # (T, A, H, W)
        T, A, H, W = masks.shape

        bbox_masks = torch.zeros_like(masks)
        for t in range(T):
            for a in range(A):
                ys, xs = torch.where(masks[t, a] > 0.5)
                if len(ys) > 0:
                    y1, y2 = ys.min().item(), ys.max().item()
                    x1, x2 = xs.min().item(), xs.max().item()
                    bbox_masks[t, a, y1:y2+1, x1:x2+1] = 1.0

        sample["masks"] = bbox_masks
        return sample


def compute_action_accuracy(action_logits, gt_actions, valid_mask=None):
    B, T1, A, _ = action_logits.shape
    pred = action_logits.argmax(dim=-1)
    if valid_mask is None:
        valid_mask = gt_actions >= 0
    correct = (pred == gt_actions) & valid_mask
    total_valid = valid_mask.sum().float()
    accuracy = (correct.sum().float() / (total_valid + 1e-8)).item()

    per_actor_acc = []
    for a in range(A):
        mask_a = valid_mask[:, :, a]
        if mask_a.sum() > 0:
            acc_a = (correct[:, :, a] & mask_a).sum().float() / mask_a.sum().float()
            per_actor_acc.append(acc_a.item())
        else:
            per_actor_acc.append(0.0)

    per_action_acc = []
    for act in range(NUM_ACTIONS):
        mask_act = (gt_actions == act) & valid_mask
        if mask_act.sum() > 0:
            acc_act = (correct & mask_act).sum().float() / mask_act.sum().float()
            per_action_acc.append(acc_act.item())
        else:
            per_action_acc.append(0.0)

    return accuracy, per_actor_acc, per_action_acc


def main():
    parser = argparse.ArgumentParser(description="BBox 矩形填充 mask 训练 LAM")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--no_interaction", action="store_true")
    parser.add_argument("--action_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.0002)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "bbox_mask_exp"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    print(f"\n{'='*60}")
    print(f"BBox 矩形填充 mask 训练 LAM")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  interaction={not args.no_interaction}")
    print(f"{'='*60}")

    # 数据集
    train_base = DiskSyntheticDataset(
        os.path.join(data_root, "train"), num_frames=5, output_format="t h w c",
    )
    eval_base = DiskSyntheticDataset(
        os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
    )

    # 用 bbox 矩形填充替代 GT mask
    train_dataset = BBoxMaskDataset(train_base)
    eval_dataset = BBoxMaskDataset(eval_base)

    # 模型
    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        max_actors=4, num_actions=NUM_ACTIONS,
        use_interaction=not args.no_interaction,
        interaction_heads=4, interaction_layers=2,
        use_grad_checkpointing=True,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # 训练
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    losses = {"total": [], "recon": [], "kl": [], "action": []}
    step = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            videos = batch["videos"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos, "masks": masks})

                gt = videos[:, 1:]
                recon_loss = ((gt - outputs["recon"]) ** 2).mean()

                z_mu = outputs["z_mu"]
                z_var = outputs["z_var"]
                kl_loss = -0.5 * torch.sum(
                    1 + z_var - z_mu ** 2 - z_var.exp()
                ) / z_mu.shape[0]

                action_logits = outputs["action_logits"]
                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, NUM_ACTIONS),
                    actions.reshape(-1),
                    ignore_index=-1,
                )

                loss = (
                    args.recon_weight * recon_loss
                    + args.kl_beta * kl_loss
                    + args.action_weight * action_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses["total"].append(float(loss))
            losses["recon"].append(float(recon_loss))
            losses["kl"].append(float(kl_loss))
            losses["action"].append(float(action_loss))

            if step % 50 == 0:
                elapsed = time.time() - t0
                mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                print(
                    f"  Step {step:3d}/{args.steps}: "
                    f"loss={float(loss):.4f}, "
                    f"recon={float(recon_loss):.4f}, "
                    f"kl={float(kl_loss):.4f}, "
                    f"action={float(action_loss):.4f}, "
                    f"mem={mem_peak:.1f}GB, {elapsed:.0f}s"
                )

            step += 1

    training_time = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"\n  Training done. Peak mem: {mem_peak:.1f}GB, time: {training_time:.0f}s")

    for key, vals in losses.items():
        np.savetxt(
            os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"),
            np.array(vals),
        )

    # 评估
    model.eval()
    results = {
        "architecture": "mask_guided_bbox_fill",
        "mask_source": "gt_bbox_rectangle_fill",
        "use_interaction": not args.no_interaction,
        "batch_size": args.batch_size,
        "training_steps": args.steps,
        "training_time_s": training_time,
        "peak_memory_gb": round(mem_peak, 2),
    }

    all_preds, all_gts, all_valid = [], [], []
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=4,
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)

            outputs = model({"videos": videos, "masks": masks})
            action_logits = outputs["action_logits"]

            all_preds.append(action_logits.cpu())
            all_gts.append(actions.cpu())
            valid = actions >= 0
            all_valid.append(valid.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_valid = torch.cat(all_valid, dim=0)

    acc, per_actor_acc, per_action_acc = compute_action_accuracy(
        all_preds, all_gts, all_valid
    )
    results["action_accuracy"] = round(acc, 4)
    results["per_actor_accuracy"] = [round(a, 4) for a in per_actor_acc]
    results["per_action_accuracy"] = {
        ACTION_NAMES[i]: round(per_action_acc[i], 4) for i in range(NUM_ACTIONS)
    }

    print(f"\n  Action Accuracy: {acc:.2%}")
    for a, acc_a in enumerate(per_actor_acc):
        print(f"    Actor {a}: {acc_a:.2%}")
    for act_name, acc_act in results["per_action_accuracy"].items():
        print(f"    Action '{act_name}': {acc_act:.2%}")

    # 重建质量
    mse_vals, psnr_vals = [], []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            outputs = model({"videos": videos, "masks": masks})
            gt = videos[:, 1:]
            mse = ((outputs["recon"] - gt) ** 2).mean().item()
            psnr = -10 * np.log10(mse + 1e-10)
            mse_vals.append(mse)
            psnr_vals.append(psnr)

    results["recon_mse"] = round(float(np.mean(mse_vals)), 6)
    results["psnr"] = round(float(np.mean(psnr_vals)), 2)
    print(f"  Recon MSE: {results['recon_mse']:.6f}")
    print(f"  PSNR: {results['psnr']:.1f} dB")

    # 隐动作分析
    z_mu = model.mu_record
    if z_mu is not None:
        z_mu = z_mu.numpy()
        N, A, D = z_mu.shape
        slot_var = [float(z_mu[:, a, :].var(axis=0).mean()) for a in range(A)]
        results["slot_variances"] = slot_var
        slot_means = np.array([z_mu[:, a, :].mean(0) for a in range(A)])
        norms = np.linalg.norm(slot_means, axis=1, keepdims=True)
        slot_means_norm = slot_means / (norms + 1e-8)
        cos_sim = slot_means_norm @ slot_means_norm.T
        results["slot_cosine_sim"] = cos_sim.tolist()
        print(f"  Slot cosine similarity:\n{np.array2string(cos_sim, precision=3)}")

    # 保存
    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Model saved: {ckpt_path}")
    print(f"  Results saved: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
