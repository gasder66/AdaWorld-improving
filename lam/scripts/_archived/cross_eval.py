"""Cross-evaluate all stride models on all stride test sets."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lam.modules import LatentActionModel
from lam.experiment_dataset import ExperimentDataset

device = torch.device("cuda:0")
EXP_DIR = "/home/xiaojy/projects/AdaWorld-improving/result/experiments"
TEST_DIR = "/home/xiaojy/projects/AdaWorld-improving/result/v6_a2d/experiments"

# Models to evaluate
models = {
    "stride1":  f"{EXP_DIR}/model_stride1.pt",
    "stride5":  f"{EXP_DIR}/model_stride5.pt",
    "stride10": f"{EXP_DIR}/model_stride10.pt",
    "stride20": f"{EXP_DIR}/model_stride20.pt",
    "stride30": f"{EXP_DIR}/model_E1_stride30.pt",
    "stride40": f"{EXP_DIR}/model_stride40.pt",
    "stride50": f"{EXP_DIR}/model_stride50.pt",
    "stride60": f"{EXP_DIR}/model_stride60.pt",
}

# Test sets (all strides)
test_strides = [1, 5, 10, 20, 30, 40, 50, 60]

# Create base model
def create_model():
    return LatentActionModel(in_dim=3, model_dim=256, latent_dim=32, patch_size=16,
                             enc_blocks=4, dec_blocks=4, num_heads=8, max_actors=4,
                             keep_background=True, use_obj_st_attention=True,
                             free_bits_lambda=0.1)

results = {}

for model_name, ckpt_path in models.items():
    if not os.path.exists(ckpt_path):
        print(f"  {model_name}: no checkpoint, skipping")
        continue
    model = create_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    print(f"Loaded {model_name}")

    for ts in test_strides:
        test_path = f"{TEST_DIR}/test_stride{ts}.pt"
        if not os.path.exists(test_path):
            continue
        test_data = torch.load(test_path, map_location="cpu")["samples"]
        loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)

        mse_list = []
        with torch.no_grad():
            for batch in loader:
                videos = batch["videos"].to(device)
                masks = batch["masks"].to(device)
                out = model({"videos": videos, "masks": masks})
                gt = videos[:, 1:]
                mse = ((gt - out["recon"]) ** 2).reshape(videos.shape[0], -1).mean(dim=1)
                mse_list.extend(mse.cpu().tolist())

        psnr = -10 * np.log10(np.array(mse_list) + 1e-10)
        if model_name not in results:
            results[model_name] = {}
        results[model_name][ts] = f"{psnr.mean():.2f}±{psnr.std():.2f}"

# Print matrix
print("\n" + "=" * 80)
header = "训练\\测试"
print(f"{header:>10}", end="")
for ts in test_strides:
    print(f" stride={ts:>2}", end="")
print()
print("-" * 80)
for model_name in models:
    print(f"{model_name:>10}", end="")
    if model_name not in results:
        print("  N/A"); continue
    for ts in test_strides:
        val = results[model_name].get(ts, "N/A")
        print(f"  {val:>8}", end="")
    print()
print("=" * 80)
