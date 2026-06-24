"""
V8 Probe Decoder: 从训练好的 V8 checkpoint 提取 z + crop pairs.

对每个 valid (video, time, slot) 三元组, 提取:
  - z_actor, z_bg (冻结, 不反传)
  - crop_t = crop(I_t, bbox_t)
  - crop_tp1_gtbox = crop(I_{t+1}, bbox_{t+1})
  - crop_tp1_predbox = crop(I_{t+1}, bbox_t + Δbbox_pred)
  - dbbox_pred, dbbox_obs, actions, actor_types
  - video_id, frame_idx, slot_id (用于 video-grouped split + 可视化)

用法:
  # 合成数据 (v8_stage1)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/extract_latents_v8.py \\
      --name v8_stage1 --dataset synthetic

  # A2D YOLO 数据 (v8_yolo)
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=lam python lam/scripts/extract_latents_v8.py \\
      --name v8_yolo --dataset a2d_yolo
"""
import os, sys, argparse, time
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lam.modules.slot_time_lam import LatentActionModelV8
from lam.modules.motion_token_encoder import _crop_resize


def _remap_legacy_keys(state_dict):
    """Remap old actor_action_head keys to new structure + drop incompatible camera_motion_pred.
    
    Old: net.0 (LayerNorm) → norm1, net.1 (Linear) → fc1,
         net.3 (LayerNorm) → norm2, net.4 (Linear) → fc2
    camera_motion_pred: old takes z_bg(16), new takes z_bg+bbox(20) — drop old, use random init.
    """
    remap = {
        "actor_action_head.net.0": "actor_action_head.norm1",
        "actor_action_head.net.1": "actor_action_head.fc1",
        "actor_action_head.net.3": "actor_action_head.norm2",
        "actor_action_head.net.4": "actor_action_head.fc2",
    }
    drop_prefixes = ["camera_motion_pred."]  # old z_bg-only version, incompatible
    new_sd = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        for old_prefix, new_prefix in remap.items():
            if k.startswith(old_prefix):
                k = k.replace(old_prefix, new_prefix)
                break
        new_sd[k] = v
    return new_sd


def build_model(args, device):
    """根据 dataset 类型构建对应配置的 V8 模型."""
    if args.dataset == "synthetic":
        model = LatentActionModelV8(
            model_dim=256, z_dim=16, z_bg_dim=16,
            num_temporal_layers=2, num_slot_layers=1, num_heads=4,
            max_actors=4, crop_size=32, img_size=256,
            free_bits_lambda=0.5, bbox_scale=32.0,
            use_bg_slot=True, num_actor_types=0,
        ).to(device)
    elif args.dataset == "a2d_yolo":
        model = LatentActionModelV8(
            model_dim=256, z_dim=16, z_bg_dim=16,
            num_temporal_layers=2, num_slot_layers=1, num_heads=4,
            max_actors=4, crop_size=32, img_size=256,
            free_bits_lambda=0.5, bbox_scale=48.0,
            use_bg_slot=True, num_actor_types=0,
        ).to(device)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    ckpt_path = os.path.join(args.results_dir, f"model_{args.name}.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = _remap_legacy_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (OK for legacy ckpt): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


def build_eval_dataset(args):
    """构建评估数据集."""
    if args.dataset == "synthetic":
        from lam.mot_slot_dataset import MOTSlotDataset
        data_root = args.data_root or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "synthetic_multi_actor"
        )
        dataset = MOTSlotDataset(
            os.path.join(data_root, "val"),
            max_actors=4, num_frames=5,
        )
    elif args.dataset == "a2d_yolo":
        from lam.yolo_box_dataset import YOLOBoxDataset
        video_dir = args.video_dir or "data/a2d/test"
        dataset = YOLOBoxDataset(
            video_dir=video_dir, release_root=args.release_root,
            split="test", T=5, stride=10,
            max_actors=4, img_size=256,
            cache_dir=args.cache_dir, max_samples=args.max_samples or 200,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="checkpoint name (e.g. v8_stage1)")
    parser.add_argument("--dataset", type=str, required=True, choices=["synthetic", "a2d_yolo"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--release_root", type=str, default="Release")
    parser.add_argument("--cache_dir", type=str, default="result/v8_mot_lam/yolo_cache")
    args = parser.parse_args()

    args.results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "result", "v8_mot_lam"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"V8 Latent Extraction: {args.name} ({args.dataset})")
    print(f"  device={device}")
    print(f"{'='*60}")

    model = build_model(args, device)
    dataset = build_eval_dataset(args)
    print(f"  Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    all_z_actor, all_z_bg = [], []
    all_crop_t, all_crop_gt, all_crop_pred = [], [], []
    all_dbbox_pred, all_dbbox_obs = [], []
    all_actions, all_actor_types = [], []
    all_video_id, all_frame_idx, all_slot_id = [], [], []

    t0 = time.time()
    n_collected = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            video = batch_gpu["videos"]
            boxes = batch_gpu["boxes"]
            valid_mask = batch_gpu["valid_mask"]
            actions = batch.get("actions")
            actor_labels = batch.get("actor_labels")
            track_ids = batch.get("track_ids")

            out = model(batch_gpu)

            mu_actor = out["mu_actor"]
            mu_bg = out["mu_bg"]
            dbbox_pred = out["dbbox_pred"]
            dbbox_obs = out["dbbox_obs"]

            B, T, K, _ = boxes.shape
            T1 = T - 1

            crops_t = _crop_resize(video[:, :T1], boxes[:, :T1], 32)
            crops_tp1_gt = _crop_resize(video[:, 1:], boxes[:, 1:], 32)

            pred_boxes_tp1 = boxes[:, :T1] + dbbox_pred
            crops_tp1_pred = _crop_resize(video[:, 1:], pred_boxes_tp1, 32)

            v_np = valid_mask[:, 1:].cpu().numpy()
            act_raw = actions if actions is not None else None
            al_np = actor_labels.cpu().numpy() if actor_labels is not None else np.zeros((B, K), dtype=int)
            tid_np = track_ids.cpu().numpy() if track_ids is not None else np.zeros((B, K), dtype=int)

            # Pad actions to K if needed (data may have fewer actor slots)
            if act_raw is not None:
                act_np = act_raw.cpu().numpy()
                if act_np.shape[2] < K:
                    pad = np.full((B, act_np.shape[1], K - act_np.shape[2]), -1, dtype=int)
                    act_np = np.concatenate([act_np, pad], axis=2)
            else:
                act_np = np.full((B, T1, K), -1, dtype=int)

            mu_a_np = mu_actor.cpu().numpy()
            mu_b_np = mu_bg.cpu().numpy()
            dbp_np = dbbox_pred.cpu().numpy()
            dbo_np = dbbox_obs.cpu().numpy()
            ct_np = crops_t.cpu().numpy()
            cg_np = crops_tp1_gt.cpu().numpy()
            cp_np = crops_tp1_pred.cpu().numpy()

            for b in range(B):
                vid = batch_idx * args.batch_size + b
                for t in range(T1):
                    for k in range(K):
                        if not v_np[b, t, k]:
                            continue
                        all_z_actor.append(mu_a_np[b, t, k])
                        all_z_bg.append(mu_b_np[b, t])
                        all_crop_t.append(ct_np[b, t, k])
                        all_crop_gt.append(cg_np[b, t, k])
                        all_crop_pred.append(cp_np[b, t, k])
                        all_dbbox_pred.append(dbp_np[b, t, k])
                        all_dbbox_obs.append(dbo_np[b, t, k])
                        all_actions.append(int(act_np[b, t, k]))
                        all_actor_types.append(int(al_np[b, k]))
                        all_video_id.append(vid)
                        all_frame_idx.append(t)
                        all_slot_id.append(int(tid_np[b, k]))
                        n_collected += 1

            elapsed = time.time() - t0
            print(f"  Batch {batch_idx}: {n_collected} samples collected, {elapsed:.0f}s")

            if args.max_samples and n_collected >= args.max_samples:
                break

    save_dict = dict(
        z_actor=np.array(all_z_actor),
        z_bg=np.array(all_z_bg),
        crop_t=np.array(all_crop_t),
        crop_tp1_gtbox=np.array(all_crop_gt),
        crop_tp1_predbox=np.array(all_crop_pred),
        dbbox_pred=np.array(all_dbbox_pred),
        dbbox_obs=np.array(all_dbbox_obs),
        actions=np.array(all_actions),
        actor_types=np.array(all_actor_types),
        video_id=np.array(all_video_id),
        frame_idx=np.array(all_frame_idx),
        slot_id=np.array(all_slot_id),
    )

    save_path = os.path.join(args.results_dir, f"latent_pairs_{args.name}.npz")
    np.savez(save_path, **save_dict)

    print(f"\n  Total samples: {n_collected}")
    print(f"  z_actor: {save_dict['z_actor'].shape}")
    print(f"  crop_t: {save_dict['crop_t'].shape}")
    print(f"  Actions: {np.unique(save_dict['actions'], return_counts=True)}")
    print(f"  Videos: {len(np.unique(save_dict['video_id']))}")
    print(f"  Saved: {save_path}")
    print(f"  Time: {time.time() - t0:.0f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
