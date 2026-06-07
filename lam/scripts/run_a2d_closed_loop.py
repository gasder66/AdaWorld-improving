"""
A2D 数据集完整训练 Mask-Guided LAM（修正版）。

核心策略变化：
1. 使用 COCO 预训练 YOLOv8n（不在 A2D 上微调）作为检测器
   - COCO 检测 ALL 实例（person, car, bird, cat, dog, sports ball）
   - A2D 标注不完整（只标注了动作相关主体），不应用来训练检测器
2. 类别映射：person→adult/baby, car→car, bird→bird, cat→cat, dog→dog, sports ball→ball
3. 加载合成数据预训练权重
4. 用 A2D 标注中的 action label 做监督（未匹配的 actor 设为 -1 忽略）

用法:
  CUDA_VISIBLE_DEVICES=2 python run_a2d_closed_loop.py \
      --name a2d_coco_yolo \
      --steps 2000
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
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules import LatentActionModel
from lam.a2d_dataset import A2DDataset, MAX_ACTORS, NUM_ACTIONS, ACTION_NAMES

# COCO 类别到 A2D actor 的映射
# COCO classes: person(0), car(2), bird(15), cat(16), dog(17), sports ball(32)
COCO_TO_A2D = {
    0: 1,    # person → adult (1)
    2: 5,    # car → car (5)
    15: 4,   # bird → bird (4)
    16: 6,   # cat → cat (6)
    17: 7,   # dog → dog (7)
    32: 3,   # sports ball → ball (3)
}
COCO_NAMES = {0: 'person', 2: 'car', 15: 'bird', 16: 'cat', 17: 'dog', 32: 'sports_ball'}
VALID_COCO_IDS = list(COCO_TO_A2D.keys())


def preprocess_a2d_with_yolo(
    dataset, yolo_model, save_path, conf=0.25, max_actors=MAX_ACTORS, img_size=256,
):
    """用 COCO 预训练 YOLO + ByteTrack 预处理 A2D 数据集。

    直接从视频文件加载帧（原始分辨率），运行 YOLO 检测，
    然后按 track_id 排序填充 mask。
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing A2D with COCO YOLO + ByteTrack...")
    print(f"  Samples: {len(dataset.valid_samples)}")
    print(f"  Conf: {conf}, Max actors: {max_actors}")
    print(f"  COCO classes: {list(COCO_NAMES.values())}")
    print(f"{'='*60}")

    all_masks = {}
    all_actions = {}
    stats = {"frames": 0, "detected": 0, "matched": 0, "no_detection": 0}

    for idx, sample_info in enumerate(tqdm(dataset.valid_samples, desc="YOLO+")):
        vid = sample_info["video_id"]
        mat_files = sample_info["mat_files"]
        frame_nums = sample_info["frame_nums"]
        T = len(mat_files)
        video_path = os.path.join(dataset.video_dir, f"{vid}.mp4")
        info = dataset.video_info[vid]
        orig_w, orig_h = info["width"], info["height"]

        frame_masks = []
        frame_actions = []

        cap = cv2.VideoCapture(video_path)
        for t in range(T):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nums[t] - 1)
            ret, frame_bgr = cap.read()

            if not ret:
                frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                orig_h, orig_w = 480, 640

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = frame_rgb.shape[:2]

            # YOLO 检测（原始分辨率）
            results = yolo_model.track(
                frame_rgb,
                tracker="bytetrack.yaml",
                persist=(t > 0),
                conf=conf,
                iou=0.45,
                verbose=False,
                imgsz=max(h0, w0),
                classes=VALID_COCO_IDS,
            )

            mask_256 = np.zeros((max_actors, img_size, img_size), dtype=np.float32)
            action_t = np.full(max_actors, -1, dtype=np.int64)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # 按 track_id 排序去重
                unique_data = {}
                for i in range(len(track_ids)):
                    tid = track_ids[i]
                    if tid not in unique_data:
                        unique_data[tid] = boxes[i]
                    else:
                        # 保留面积更大的
                        old_area = (unique_data[tid][2] - unique_data[tid][0]) * (unique_data[tid][3] - unique_data[tid][1])
                        new_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                        if new_area > old_area:
                            unique_data[tid] = boxes[i]

                # 按 track_id 排序
                sorted_tids = sorted(unique_data.keys())
                sorted_boxes = [unique_data[tid] for tid in sorted_tids]

                # 填充 mask（缩放到 256x256）
                scale_x = img_size / w0
                scale_y = img_size / h0
                for i, box in enumerate(sorted_boxes[:max_actors]):
                    x1 = max(0, int(box[0] * scale_x))
                    y1 = max(0, int(box[1] * scale_y))
                    x2 = min(img_size, int(box[2] * scale_x))
                    y2 = min(img_size, int(box[3] * scale_y))
                    if x2 > x1 and y2 > y1:
                        mask_256[i, y1:y2, x1:x2] = 1.0

                stats["detected"] += len(sorted_boxes)

                # 匹配检测框与 A2D 标注（获取 action label）
                mat_path = os.path.join(dataset.annot_dir, vid, mat_files[t])
                try:
                    import h5py
                    with h5py.File(mat_path, 'r') as f:
                        gt_bbox = np.array(f['reBBox'])  # (4, N) — 也是原始分辨率
                        gt_ids = np.array(f['id']).flatten()
                        n_gt = gt_bbox.shape[1]
                except:
                    n_gt = 0

                for i, box in enumerate(sorted_boxes[:max_actors]):
                    if i >= n_gt:
                        continue
                    gt_box = gt_bbox[:, i]
                    xa = max(box[0], gt_box[0])
                    ya = max(box[1], gt_box[1])
                    xb = min(box[2], gt_box[2])
                    yb = min(box[3], gt_box[3])
                    inter = max(0, xb - xa) * max(0, yb - ya)
                    area1 = (box[2] - box[0]) * (box[3] - box[1])
                    area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                    union = area1 + area2 - inter
                    iou = inter / (union + 1e-6)
                    if iou >= 0.3:
                        action_id = int(gt_ids[i]) % 10
                        if 1 <= action_id <= 8:
                            action_t[i] = action_id - 1
                            stats["matched"] += 1
            else:
                stats["no_detection"] += 1

            frame_masks.append(mask_256)
            if t > 0:
                frame_actions.append(action_t)

        cap.release()

        # Pad actions if needed
        while len(frame_actions) < T - 1:
            frame_actions.append(np.full(max_actors, -1, dtype=np.int64))

        all_masks[idx] = torch.from_numpy(np.stack(frame_masks))
        all_actions[idx] = torch.from_numpy(np.stack(frame_actions[:T-1]))
        stats["frames"] += T

        if (idx + 1) % 500 == 0:
            print(f"    [{idx+1}/{len(dataset.valid_samples)}] "
                  f"det={stats['detected']}, match={stats['matched']}, "
                  f"no_det={stats['no_detection']}")

    torch.save({"masks": all_masks, "actions": all_actions}, save_path)
    det_rate = stats["detected"] / max(stats["frames"], 1)
    print(f"\n  Done. Avg detections/frame: {stats['detected']/max(stats['frames'],1):.1f}")
    print(f"         Matched: {stats['matched']}")
    print(f"         No det frames: {stats['no_detection']}")
    print(f"  Saved: {save_path}")
    return all_masks, all_actions


def compute_action_accuracy(action_logits, gt_actions, valid_mask=None):
    B, T1, A, _ = action_logits.shape
    pred = action_logits.argmax(dim=-1)
    if valid_mask is None:
        valid_mask = gt_actions >= 0
    correct = (pred == gt_actions) & valid_mask
    total_valid = valid_mask.sum().float()
    accuracy = (correct.sum().float() / (total_valid + 1e-8)).item()

    per_action_acc = []
    for act in range(NUM_ACTIONS):
        mask_act = (gt_actions == act) & valid_mask
        if mask_act.sum() > 0:
            acc_act = (correct & mask_act).sum().float() / mask_act.sum().float()
            per_action_acc.append(acc_act.item())
        else:
            per_action_acc.append(0.0)

    return accuracy, per_action_acc


def main():
    parser = argparse.ArgumentParser(description="A2D 闭环训练（COCO YOLO + ByteTrack）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--no_interaction", action="store_true")
    parser.add_argument("--action_weight", type=float, default=2.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--kl_beta", type=float, default=0.0001)
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results", "a2d_closed_loop"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "a2d")
    release_root = os.path.join(os.path.dirname(__file__), "..", "..", "Release")

    print(f"\n{'='*60}")
    print(f"A2D 闭环训练（COCO YOLO + ByteTrack）")
    print(f"  GPU={args.gpu}, name={args.name}, batch={args.batch_size}, steps={args.steps}")
    print(f"  interaction={not args.no_interaction}, num_frames={args.num_frames}")
    print(f"  pretrained={args.pretrained}")
    print(f"{'='*60}")

    # ===== 数据集 =====
    train_dataset = A2DDataset(
        data_root, release_root, split="train",
        num_frames=args.num_frames, max_actors=MAX_ACTORS,
    )
    eval_dataset = A2DDataset(
        data_root, release_root, split="test",
        num_frames=args.num_frames, max_actors=MAX_ACTORS,
    )

    # ===== 预处理（COCO YOLO + ByteTrack） =====
    cache_path = os.path.join(RESULTS_DIR, f"coco_yolo_masks_{args.num_frames}f.pt")

    if not args.skip_preprocess and not args.eval_only:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')

        train_masks, train_actions = preprocess_a2d_with_yolo(
            train_dataset, yolo_model, cache_path.replace('.pt', '_train.pt'),
            conf=args.conf, max_actors=MAX_ACTORS, img_size=args.img_size,
        )
        eval_masks, eval_actions = preprocess_a2d_with_yolo(
            eval_dataset, yolo_model, cache_path.replace('.pt', '_test.pt'),
            conf=args.conf, max_actors=MAX_ACTORS, img_size=args.img_size,
        )
    elif args.eval_only:
        # 只评估
        pass
    else:
        data = torch.load(cache_path.replace('.pt', '_train.pt'), map_location='cpu')
        train_masks, train_actions = data["masks"], data["actions"]
        data = torch.load(cache_path.replace('.pt', '_test.pt'), map_location='cpu')
        eval_masks, eval_actions = data["masks"], data["actions"]
        print(f"  Loaded cached masks: train={len(train_masks)}, test={len(eval_masks)}")

    # ===== 包装数据集 =====
    class A2DYOLODataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, masks, actions):
            self.base = base_dataset
            self.masks = masks
            self.actions = actions

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            sample = self.base[idx]
            if idx in self.masks:
                sample["masks"] = self.masks[idx]
            if idx in self.actions:
                sample["actions"] = self.actions[idx]
            return sample

    if not args.eval_only:
        train_dataset = A2DYOLODataset(train_dataset, train_masks, train_actions)
        eval_dataset = A2DYOLODataset(eval_dataset, eval_masks, eval_actions)

    # ===== 模型 =====
    model = LatentActionModel(
        in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
        enc_blocks=4, dec_blocks=4, num_heads=8,
        max_actors=MAX_ACTORS, num_actions=NUM_ACTIONS,
        use_interaction=not args.no_interaction,
        interaction_heads=4, interaction_layers=2,
        use_grad_checkpointing=True,
    ).to(device)

    if args.pretrained:
        print(f"  Loading pretrained: {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"    Missing: {len([k for k in state_dict if k not in model.state_dict()])}")

    if args.model_path:
        print(f"  Loading model: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # ===== 训练 =====
    if not args.eval_only:
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
                    z_mu, z_var = outputs["z_mu"], outputs["z_var"]
                    kl_loss = -0.5 * torch.sum(
                        1 + z_var - z_mu ** 2 - z_var.exp()
                    ) / z_mu.shape[0]

                    action_logits = outputs["action_logits"]
                    action_loss = F.cross_entropy(
                        action_logits.reshape(-1, NUM_ACTIONS),
                        actions.reshape(-1),
                        ignore_index=-1,
                    )

                    loss = (args.recon_weight * recon_loss +
                            args.kl_beta * kl_loss +
                            args.action_weight * action_loss)

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

                if step % 100 == 0:
                    elapsed = time.time() - t0
                    mem = torch.cuda.max_memory_allocated(device) / 1024**3
                    print(f"  Step {step:3d}: loss={float(loss):.3f} recon={float(recon_loss):.3f} "
                          f"kl={float(kl_loss):.1f} act={float(action_loss):.3f} mem={mem:.1f}GB")

                step += 1

        training_time = time.time() - t0
        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\n  Training done. Mem: {mem_peak:.1f}GB, time: {training_time:.0f}s")

        for key, vals in losses.items():
            np.savetxt(os.path.join(RESULTS_DIR, f"loss_{args.name}_{key}.txt"), np.array(vals))

        ckpt_path = os.path.join(RESULTS_DIR, f"model_{args.name}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Model: {ckpt_path}")

    # ===== 评估 =====
    model.eval()
    results = {
        "architecture": "mask_guided_a2d_coco_yolo",
        "mask_source": "coco_yolov8n_bytetrack",
        "use_interaction": not args.no_interaction,
        "num_frames": args.num_frames,
        "training_steps": args.steps if hasattr(args, 'steps') else 0,
        "pretrained": args.pretrained,
        "detector": "yolov8n_coco_pretrained",
    }

    all_preds, all_gts, all_valid = [], [], []
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=16, num_workers=4,
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 50:
                break
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)
            outputs = model({"videos": videos, "masks": masks})
            all_preds.append(outputs["action_logits"].cpu())
            all_gts.append(actions.cpu())
            all_valid.append((actions >= 0).cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_valid = torch.cat(all_valid, dim=0)

    acc, per_action_acc = compute_action_accuracy(all_preds, all_gts, all_valid)
    results["action_accuracy"] = round(acc, 4)
    results["per_action_accuracy"] = {
        ACTION_NAMES[i]: round(per_action_acc[i], 4) for i in range(NUM_ACTIONS)
    }

    print(f"\n  Action Accuracy: {acc:.2%} (random: {1/NUM_ACTIONS:.1%})")
    for act_name, acc_act in results["per_action_accuracy"].items():
        print(f"    {act_name}: {acc_act:.2%}")

    save_path = os.path.join(RESULTS_DIR, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()