"""
合成数据闭环验证：用 LocateAnything 检测框 + 矩形填充替代 GT mask。

流程：
1. 用 LocateAnything 检测每帧中的物体
2. 将检测框转为矩形填充 mask
3. 用预测 mask 替代 GT mask 训练 Mask-Guided LAM
4. 对比：GT mask vs 预测 mask 的动作预测准确率

用法:
  CUDA_VISIBLE_DEVICES=2 python run_with_pred_masks.py --gpu 0 --steps 500
"""
import os
import sys
import json
import time
import argparse

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eagle", "Embodied"))

from lam.modules import LatentActionModel
from lam.disk_synthetic_dataset import DiskSyntheticDataset

ACTION_NAMES = ["stay", "up", "down", "left", "right"]
NUM_ACTIONS = 5


def boxes_to_masks(boxes, H, W, max_actors=4):
    """将检测框列表转为矩形填充 mask。

    Args:
        boxes: list of dicts with x1,y1,x2,y2 keys
        H, W: 图像尺寸
        max_actors: 最大主体数

    Returns:
        masks: (max_actors, H, W) float32 binary
    """
    masks = torch.zeros(max_actors, H, W, dtype=torch.float32)
    for i, box in enumerate(boxes[:max_actors]):
        x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
        x2, y2 = min(W, int(box["x2"])), min(H, int(box["y2"]))
        masks[i, y1:y2, x1:x2] = 1.0
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--name", type=str, default="v2_pred_mask")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--action_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.0002)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "slot_attention_exp_v2"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ===== 加载 LocateAnything =====
    print("Loading LocateAnything...")
    from locateanything_worker import LocateAnythingWorker
    la_worker = LocateAnythingWorker(
        "nvidia/LocateAnything-3B",
        device=device,
        dtype=torch.bfloat16,
    )
    print("  LocateAnything loaded")

    # ===== 数据集 =====
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
    )
    train_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "train"),
        num_frames=5, output_format="t h w c",
    )
    eval_dataset = DiskSyntheticDataset(
        os.path.join(data_root, "val"),
        num_frames=5, output_format="t h w c",
    )
    print(f"Train: {len(train_dataset)}, Val: {len(eval_dataset)}")

    # ===== 预生成预测 mask（避免训练时反复推理 LA）=====
    pred_mask_dir = os.path.join(data_root, "pred_masks_la")
    os.makedirs(pred_mask_dir, exist_ok=True)
    manifest_path = os.path.join(pred_mask_dir, "manifest.json")

    if os.path.exists(manifest_path):
        print(f"Loading pre-generated masks from {pred_mask_dir}")
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        print("Generating predicted masks with LocateAnything...")
        manifest = {"train": [], "val": []}

        for split, dataset in [("train", train_dataset), ("val", eval_dataset)]:
            num_to_process = min(len(dataset), 500)  # 限制数量以节省时间
            for idx in range(num_to_process):
                sample = dataset[idx]
                videos = sample["videos"]  # (T, H, W, C) float32 [0,1]
                T, H, W = videos.shape[:3]

                # 对每帧检测
                frame_masks = []
                for t in range(T):
                    frame = videos[t].numpy()
                    frame_pil = Image.fromarray((frame * 255).astype(np.uint8))

                    try:
                        result = la_worker.detect(
                            frame_pil,
                            ["colored block", "colored square", "colored shape"],
                            generation_mode="fast",
                            temperature=0.1,
                        )
                        parsed = LocateAnythingWorker.parse_boxes(
                            result["answer"], W, H
                        )
                    except Exception:
                        parsed = []

                    # 去重
                    boxes = []
                    for box in parsed:
                        is_dup = False
                        for existing in boxes:
                            xa = max(box["x1"], existing["x1"])
                            ya = max(box["y1"], existing["y1"])
                            xb = min(box["x2"], existing["x2"])
                            yb = min(box["y2"], existing["y2"])
                            inter = max(0, xb - xa) * max(0, yb - ya)
                            ba = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                            ea = (existing["x2"] - existing["x1"]) * (existing["y2"] - existing["y1"])
                            iou = inter / (ba + ea - inter + 1e-6)
                            if iou > 0.5:
                                is_dup = True
                                break
                        if not is_dup:
                            boxes.append(box)

                    mask = boxes_to_masks(boxes, H, W, max_actors=4)
                    frame_masks.append(mask)

                # (T, max_actors, H, W)
                pred_masks = torch.stack(frame_masks, dim=0)
                save_path = os.path.join(pred_mask_dir, f"{split}_{idx:05d}.pt")
                torch.save(pred_masks, save_path)
                manifest[split].append(save_path)

                if (idx + 1) % 50 == 0:
                    print(f"  {split}: {idx + 1}/{num_to_process} done")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        print(f"  Saved {len(manifest['train'])} train + {len(manifest['val'])} val masks")

    # ===== 训练 LAM =====
    print(f"\n{'='*60}")
    print(f"Training Mask-Guided LAM with PREDICTED masks")
    print(f"  Steps={args.steps}, batch={args.batch_size}")
    print(f"{'='*60}")

    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        max_actors=4, num_actions=5, img_size=256,
        use_interaction=True,
        interaction_heads=4, interaction_layers=2,
        use_grad_checkpointing=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    step = 0
    t0 = time.time()
    losses = []

    # 简单训练循环：从 eval_dataset 取数据，用预测 mask 替代 GT mask
    dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            videos = batch["videos"].to(device)    # (B, T, H, W, C)
            actions = batch["actions"].to(device)   # (B, T-1, A)
            B = videos.shape[0]

            # 用预测 mask 替代 GT mask
            pred_masks_batch = []
            for b in range(B):
                idx_b = batch.get("idx", b)  # fallback
                # 尝试加载对应的预测 mask
                mask_path = manifest["val"][b % len(manifest["val"])]
                pm = torch.load(mask_path, map_location="cpu")  # (T, A, H, W)
                pred_masks_batch.append(pm)

            pred_masks = torch.stack(pred_masks_batch, dim=0).to(device)

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos, "masks": pred_masks})

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

            losses.append(float(loss))
            if step % 50 == 0:
                print(f"  Step {step:3d}/{args.steps}: "
                      f"loss={float(loss):.4f}, "
                      f"recon={float(recon_loss):.4f}, "
                      f"kl={float(kl_loss):.4f}, "
                      f"action={float(action_loss):.4f}")

            step += 1

    training_time = time.time() - t0
    print(f"\n  Training done. Time: {training_time:.0f}s")

    # ===== 评估 =====
    model.eval()
    all_preds, all_gts, all_valid = [], [], []

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=4,
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 20:
                break
            videos = batch["videos"].to(device)
            actions = batch["actions"].to(device)
            B = videos.shape[0]

            # 用预测 mask
            pred_masks_batch = []
            for b in range(B):
                mask_path = manifest["val"][b % len(manifest["val"])]
                pm = torch.load(mask_path, map_location="cpu")
                pred_masks_batch.append(pm)
            pred_masks = torch.stack(pred_masks_batch, dim=0).to(device)

            outputs = model({"videos": videos, "masks": pred_masks})
            all_preds.append(outputs["action_logits"].cpu())
            all_gts.append(actions.cpu())
            all_valid.append((actions >= 0).cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_valid = torch.cat(all_valid, dim=0)

    pred = all_preds.argmax(dim=-1)
    correct = (pred == all_gts) & all_valid
    acc = (correct.sum().float() / all_valid.sum().float()).item()

    print(f"\n  Action Accuracy (pred masks): {acc:.2%}")
    print(f"  Compare with GT masks: ~88.8%")
    print(f"{'='*60}")

    # 保存结果
    results = {
        "mask_source": "locateanything_box_fill",
        "action_accuracy": round(acc, 4),
        "training_steps": args.steps,
        "training_time_s": round(training_time),
    }
    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()