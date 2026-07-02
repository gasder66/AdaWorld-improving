"""E1 实验：T=2, stride=30 的快速训练脚本。

用预生成的 ExperimentDataset 避免 YOLO on-the-fly 延迟。
"""
import os, sys, json, time, torch, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch.nn.functional as F
import numpy as np
from lam.modules import LatentActionModel
from lam.experiment_dataset import ExperimentDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--kl_beta", type=float, default=2e-4)
    parser.add_argument("--obj_recon_weight", type=float, default=0.01)
    parser.add_argument("--free_bits_lambda", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    dataset = ExperimentDataset(args.data_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = LatentActionModel(in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
                              enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
                              keep_background=True, use_obj_st_attention=True,
                              free_bits_lambda=args.free_bits_lambda).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}, 可训练: {trainable_params:,}")
    print(f"  设备: {device}, 数据: {len(dataset)} 样本, 训练 {args.steps} 步")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    step = 0
    t0 = time.time()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps: break
            videos = batch["videos"].to(device)  # (B, T, H, W, C)
            masks = batch["masks"].to(device)     # (B, T, A, H, W)

            with torch.cuda.amp.autocast():
                outputs = model({"videos": videos, "masks": masks})
                recon_loss = ((videos[:, 1:] - outputs["recon"]) ** 2).mean()
                kl_loss = outputs["kl_loss"]
                obj_recon_loss = outputs["obj_recon_loss"]
                loss = recon_loss + args.kl_beta * kl_loss + args.obj_recon_weight * obj_recon_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Step {step}: loss={float(loss):.4f}, recon={float(recon_loss):.4f}, "
                      f"kl={float(kl_loss):.4f}, obj={float(obj_recon_loss):.4f}, {elapsed:.0f}s")
            step += 1

    training_time = time.time() - t0
    print(f"\n  训练完成. 耗时: {training_time:.0f}s")

    # 评估
    model.eval()
    mse_vals = []
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            masks = batch["masks"].to(device)
            out = model({"videos": videos, "masks": masks})
            mse = ((videos[:, 1:] - out["recon"]) ** 2).reshape(videos.shape[0], -1).mean(dim=1)
            mse_vals.extend(mse.cpu().tolist())

    mse_arr = np.array(mse_vals)
    psnr_arr = -10 * np.log10(mse_arr + 1e-10)
    print(f"  PSNR: {psnr_arr.mean():.2f} ± {psnr_arr.std():.2f} dB")

    results = {"psnr": float(psnr_arr.mean()), "steps": args.steps,
               "data": args.data_path, "total_params": total_params}
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "result", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"model_{args.name}.pt")
    res_json = os.path.join(out_dir, f"results_{args.name}.json")
    torch.save(model.state_dict(), ckpt)
    with open(res_json, "w") as f: json.dump(results, f, indent=2)
    print(f"  保存: {ckpt}")
    print("Done!")


if __name__ == "__main__":
    main()
