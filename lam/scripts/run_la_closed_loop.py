"""
闭环训练脚本：用 LocateAnything 检测框 + 矩形填充替代 GT mask 训练 LAM。

流程：
1. 预处理：用 LocateAnything 检测合成数据中每帧的彩色方块，生成 box→矩形填充 mask
2. 训练：用 LA 生成的 mask 替代 GT mask 训练 Mask-Guided LAM
3. 对比：与 GT mask 基线对比动作准确率

用法:
  # 完整流程（预处理 + 训练）
  CUDA_VISIBLE_DEVICES=2 python run_la_closed_loop.py --gpu 0 --name la_closed_loop --steps 500

  # 仅预处理（不训练）
  CUDA_VISIBLE_DEVICES=2 python run_la_closed_loop.py --gpu 0 --name la_closed_loop --preprocess_only

  # 跳过预处理（使用已有缓存）
  CUDA_VISIBLE_DEVICES=2 python run_la_closed_loop.py --gpu 0 --name la_closed_loop --skip_preprocess --steps 500
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
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eagle", "Embodied"))

from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
NUM_ACTIONS = 5

# ===== 预处理：LocateAnything 检测 → 矩形填充 mask =====

DETECTION_PROMPTS = [
    ["colored block", "colored square", "colored shape"],
    ["red object", "green object", "blue object", "yellow object", "purple object"],
    ["brightly colored shape on a patterned background"],
]


def boxes_to_mask(boxes, H, W, max_actors=4):
    """将检测到的 bounding box 转换为矩形填充 mask。

    Args:
        boxes: list of dict {"x1", "y1", "x2", "y2"} 像素坐标
        H, W: 图像尺寸
        max_actors: 最大主体数

    Returns:
        mask: (max_actors, H, W) float32 binary
    """
    mask = torch.zeros(max_actors, H, W, dtype=torch.float32)
    for i, box in enumerate(boxes[:max_actors]):
        x1 = max(0, int(box["x1"]))
        y1 = max(0, int(box["y1"]))
        x2 = min(W, int(box["x2"]))
        y2 = min(H, int(box["y2"]))
        mask[i, y1:y2, x1:x2] = 1.0
    return mask


def deduplicate_boxes(boxes, iou_threshold=0.5):
    """基于 IoU 去重检测框。"""
    if not boxes:
        return boxes
    kept = []
    for box in boxes:
        is_dup = False
        for existing in kept:
            xa = max(box["x1"], existing["x1"])
            ya = max(box["y1"], existing["y1"])
            xb = min(box["x2"], existing["x2"])
            yb = min(box["y2"], existing["y2"])
            inter = max(0, xb - xa) * max(0, yb - ya)
            box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            exist_area = (existing["x2"] - existing["x1"]) * (existing["y2"] - existing["y1"])
            union = box_area + exist_area - inter
            iou = inter / (union + 1e-6)
            if iou > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(box)
    return kept


def match_boxes_to_gt(detected_boxes, gt_boxes, iou_threshold=0.3):
    """将检测框与 GT 框匹配，按 GT 顺序排列。

    Args:
        detected_boxes: list of dict
        gt_boxes: list of [x1, y1, x2, y2]
        iou_threshold: 匹配阈值

    Returns:
        matched_boxes: list of dict, 按 GT 顺序排列
    """
    if not gt_boxes or not detected_boxes:
        return [None] * len(gt_boxes)

    matched = [None] * len(gt_boxes)
    used = set()

    for i, gt in enumerate(gt_boxes):
        best_iou = 0
        best_j = -1
        for j, dt in enumerate(detected_boxes):
            if j in used:
                continue
            xa = max(gt[0], dt["x1"])
            ya = max(gt[1], dt["y1"])
            xb = min(gt[2], dt["x2"])
            yb = min(gt[3], dt["y2"])
            inter = max(0, xb - xa) * max(0, yb - ya)
            gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            dt_area = (dt["x2"] - dt["x1"]) * (dt["y2"] - dt["y1"])
            union = gt_area + dt_area - inter
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched[i] = detected_boxes[best_j]
            used.add(best_j)

    return matched


def extract_gt_boxes_from_masks(masks):
    """从 GT mask 中提取 bounding box。"""
    A, H, W = masks.shape
    boxes = []
    for a in range(A):
        ys, xs = torch.where(masks[a] > 0.5)
        if len(ys) == 0:
            boxes.append(None)
        else:
            boxes.append([xs.min().item(), ys.min().item(),
                          xs.max().item(), ys.max().item()])
    return boxes


def preprocess_dataset(dataset, la_worker, device, save_path, max_actors=4):
    """用 LocateAnything 预处理数据集，生成 LA mask 缓存。

    对每个样本的每一帧：
    1. 用 LA 检测彩色方块
    2. 去重检测框
    3. 与 GT 框匹配（保持主体顺序一致性）
    4. 矩形填充生成 mask

    保存格式：dict of sample_idx -> (T, max_actors, H, W) mask tensor
    """
    from locateanything_worker import LocateAnythingWorker as _LAW
    print(f"\n{'='*60}")
    print(f"Preprocessing {len(dataset)} samples with LocateAnything...")
    print(f"{'='*60}")

    all_la_masks = {}
    stats = {"total_gt": 0, "total_recalled": 0, "total_missed": 0,
             "total_false_pos": 0, "total_frames": 0}

    for idx in tqdm(range(len(dataset)), desc="LA preprocessing"):
        sample = dataset[idx]
        videos = sample["videos"]  # (T, H, W, C) float32 [0,1]
        masks = sample["masks"]    # (T, max_actors, H, W)
        T, H, W, C = videos.shape

        frame_masks = []

        for t in range(T):
            frame = videos[t].numpy()  # (H, W, C)
            frame_pil = Image.fromarray((frame * 255).astype(np.uint8))

            # 用多种 prompt 检测
            all_detected = []
            for prompt_set in DETECTION_PROMPTS:
                try:
                    result = la_worker.detect(frame_pil, prompt_set,
                                              generation_mode="fast",
                                              temperature=0.1)
                    parsed = _LAW.parse_boxes(
                        result["answer"], W, H
                    )
                    if parsed:
                        all_detected.extend(parsed)
                except Exception as e:
                    print(f"    Frame {t} prompt failed: {e}")

            # 去重
            detected_boxes = deduplicate_boxes(all_detected)

            # 与 GT 匹配（保持主体顺序一致性）
            gt_boxes = extract_gt_boxes_from_masks(masks[t])
            gt_boxes_valid = [b for b in gt_boxes if b is not None]
            matched_boxes = match_boxes_to_gt(detected_boxes, gt_boxes_valid)

            # 按 GT 顺序生成 mask
            mask_t = torch.zeros(max_actors, H, W, dtype=torch.float32)
            for a, gt_box in enumerate(gt_boxes):
                if gt_box is not None and a < len(matched_boxes) and matched_boxes[a] is not None:
                    box = matched_boxes[a]
                    x1 = max(0, int(box["x1"]))
                    y1 = max(0, int(box["y1"]))
                    x2 = min(W, int(box["x2"]))
                    y2 = min(H, int(box["y2"]))
                    mask_t[a, y1:y2, x1:x2] = 1.0

            frame_masks.append(mask_t)

            # 统计
            stats["total_frames"] += 1
            stats["total_gt"] += len(gt_boxes_valid)
            recalled = sum(1 for m in matched_boxes if m is not None)
            stats["total_recalled"] += recalled
            stats["total_missed"] += len(gt_boxes_valid) - recalled
            stats["total_false_pos"] += len(detected_boxes) - recalled

        la_masks = torch.stack(frame_masks, dim=0)  # (T, max_actors, H, W)
        all_la_masks[idx] = la_masks

        if (idx + 1) % 50 == 0:
            recall = stats["total_recalled"] / (stats["total_gt"] + 1e-8)
            precision = stats["total_recalled"] / (stats["total_recalled"] + stats["total_false_pos"] + 1e-8)
            print(f"  [{idx+1}/{len(dataset)}] Recall={recall:.1%}, Precision={precision:.1%}")

    # 保存
    torch.save(all_la_masks, save_path)

    recall = stats["total_recalled"] / (stats["total_gt"] + 1e-8)
    precision = stats["total_recalled"] / (stats["total_recalled"] + stats["total_false_pos"] + 1e-8)
    print(f"\n  Preprocessing done. Recall={recall:.1%}, Precision={precision:.1%}")
    print(f"  Saved: {save_path}")

    return all_la_masks, stats


class LAMaskDataset(torch.utils.data.Dataset):
    """包装 DiskSyntheticDataset，用 LA 生成的 mask 替代 GT mask。"""

    def __init__(self, base_dataset, la_masks_cache):
        self.base = base_dataset
        self.la_masks = la_masks_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if idx in self.la_masks:
            sample["masks"] = self.la_masks[idx]  # 替换为 LA mask
        return sample


# ===== 训练 =====

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
    parser = argparse.ArgumentParser(description="LA 闭环训练")
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
    parser.add_argument("--preprocess_only", action="store_true",
                        help="仅预处理，不训练")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="跳过预处理，使用已有缓存")
    parser.add_argument("--preprocess_split", type=str, default="train",
                        choices=["train", "val", "both"],
                        help="预处理哪个 split")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "la_closed_loop"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
    else:
        data_root = args.data_root

    # ===== 数据集 =====
    train_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "train"), num_frames=5, output_format="t h w c",
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"), num_frames=5, output_format="t h w c",
    )

    # ===== 预处理 =====
    train_cache_path = os.path.join(RESULTS_DIR, "la_masks_train.pt")
    val_cache_path = os.path.join(RESULTS_DIR, "la_masks_val.pt")

    train_la_masks = None
    val_la_masks = None

    if not args.skip_preprocess:
        print("Loading LocateAnythingWorker...")
        from locateanything_worker import LocateAnythingWorker
        la_worker = LocateAnythingWorker(
            "nvidia/LocateAnything-3B", device=device, dtype=torch.bfloat16,
        )
        print("  LocateAnythingWorker loaded")

        if args.preprocess_split in ("train", "both"):
            train_la_masks, train_stats = preprocess_dataset(
                train_dataset, la_worker, device, train_cache_path,
            )

        if args.preprocess_split in ("val", "both"):
            val_la_masks, val_stats = preprocess_dataset(
                eval_dataset, la_worker, device, val_cache_path,
            )

        if args.preprocess_only:
            print("Preprocess only mode. Done.")
            return
    else:
        # 加载缓存
        if os.path.exists(train_cache_path):
            train_la_masks = torch.load(train_cache_path, map_location="cpu")
            print(f"  Loaded train LA masks cache: {len(train_la_masks)} samples")
        if os.path.exists(val_cache_path):
            val_la_masks = torch.load(val_cache_path, map_location="cpu")
            print(f"  Loaded val LA masks cache: {len(val_la_masks)} samples")

    # 包装数据集
    if train_la_masks is not None:
        train_dataset = LAMaskDataset(train_dataset, train_la_masks)
    if val_la_masks is not None:
        eval_dataset = LAMaskDataset(eval_dataset, val_la_masks)

    # ===== 训练 =====
    print(f"\n{'='*60}")
    print(f"LA 闭环训练")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  interaction={not args.no_interaction}")
    print(f"  LA masks: train={'loaded' if train_la_masks else 'GT'}, val={'loaded' if val_la_masks else 'GT'}")
    print(f"{'='*60}")

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

    # 保存损失曲线
    for key, vals in losses.items():
        np.savetxt(
            os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"),
            np.array(vals),
        )

    # ===== 评估 =====
    model.eval()
    results = {
        "architecture": "mask_guided_la_closed_loop",
        "mask_source": "locateanything_box_fill",
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
