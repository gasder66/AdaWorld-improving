from __future__ import annotations

import functools
import os
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from lam.dataset import LightningVideoDataset
from lam.model import LAM


def select_resume_checkpoint(ckpt_dir: str) -> str:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    preferred = sorted(ckpt_path.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if preferred:
        return str(preferred[0])

    fallbacks = sorted(ckpt_path.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if fallbacks:
        return str(fallbacks[0])

    raise FileNotFoundError(f"No .ckpt found under: {ckpt_dir}")


def main() -> None:
    torch.set_float32_matmul_precision('high')  # Tensor Cores
    torch.backends.cudnn.benchmark = True

    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus == 0:
        raise RuntimeError("No visible GPU found.")

    resume_ckpt = select_resume_checkpoint("exp_ckpts_a2d")
    checkpoint = torch.load(resume_ckpt, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]
    run_id = os.environ.get("LAM_RUN_ID", Path(resume_ckpt).stem)

    model = LAM(
        image_channels=hparams.get("image_channels", 3),
        lam_model_dim=hparams.get("lam_model_dim", 1024),
        lam_latent_dim=hparams.get("lam_latent_dim", 32),
        lam_patch_size=hparams.get("lam_patch_size", 16),
        lam_enc_blocks=hparams.get("lam_enc_blocks", 16),
        lam_dec_blocks=hparams.get("lam_dec_blocks", 16),
        lam_num_heads=hparams.get("lam_num_heads", 16),
        lam_dropout=hparams.get("lam_dropout", 0.0),
        beta=hparams.get("beta", 0.0002),
        log_interval=hparams.get("log_interval", 1000),
        log_path=str(Path("exp_imgs_a2d") / "lam_a2d_resume" / run_id),
        optimizer=hparams.get("optimizer"),
    )

    data = LightningVideoDataset(
        data_root="../data",
        env_source="a2d",
        padding="repeat",
        randomize=True,
        resolution=256,
        num_frames=2,
        output_format="t h w c",
        samples_per_epoch=100000,
        sampling_strategy="sample",
        batch_size=1,
        num_workers=8,
    )

    trainer = Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=num_visible_gpus,
        strategy="ddp_find_unused_parameters_false",
        precision="16-mixed",
        log_every_n_steps=1000,
        accumulate_grad_batches=8,
        gradient_clip_val=0.3,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                dirpath="exp_ckpts_a2d",
                verbose=True,
                save_last=True,
                save_top_k=-1,
                save_weights_only=False,
            )
        ],
        logger=[TensorBoardLogger(save_dir="exp_logs_a2d", name="lam_a2d_resume", version=run_id)],
    )

    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
