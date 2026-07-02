---
name: "TRAE-experiment-manager"
description: "Project organization and experiment management standards for AdaWorld."
---

# AdaWorld Experiment Manager

## Project Structure

```
project_root/
├── configs/                # YAML configuration files
├── data/                   # Datasets (gitignored)
├── experiments/            # Experiment batch plans + results
│   └── batch_VVV_name/
│       ├── plan.md         # Goals, method, success criteria
│       └── results.md      # Metrics, findings, conclusions
├── result/                 # ALL experiment artifacts
│   ├── _archived/          # Deprecated results (v6, v7, v8, v9)
│   ├── {version}/          # One directory per model version (v10, v11, ...)
│   │   └── {dataset}/      # One directory per dataset
│   │       └── {config}/   # One directory per training run
│   │           ├── model.pt    # Final checkpoint (always this name)
│   │           ├── ckpts/      # Intermediate checkpoints: step{step}.pt
│   │           ├── losses/     # Loss curves: total.txt, recon.txt, kl.txt, ...
│   │           ├── results.json    # Training inline-eval metrics
│   │           ├── eval.json       # Full evaluation metrics
│   │           ├── latents.npz     # Latent codes for analysis
│   │           ├── umap.png        # UMAP visualization (optional)
│   │           └── recon.png       # Reconstruction viz (optional)
├── reports/                # Detailed analysis reports
├── tools/                  # One-off utility scripts
├── lam/                    # Main Python package (PYTHONPATH=lam)
│   ├── lam/                # Source code
│   │   ├── modules/        # Model architectures + building blocks
│   │   │   ├── _archived/  # Deprecated modules (V3, V8, V9)
│   │   │   ├── blocks.py   # Core building blocks
│   │   │   ├── v10_model.py, v11_model.py  # Active models
│   │   │   └── __init__.py
│   │   └── *_dataset.py    # Dataset classes
│   └── scripts/            # Training + evaluation scripts
│       ├── _archived/      # Deprecated scripts
│       ├── common/         # Version-agnostic tools
│       │   ├── analyze_umap.py
│       │   ├── generate_synthetic_dataset.py
│       │   └── vis_reconstruction.py
│       ├── v9/             # V9 legacy
│       ├── v10/            # V10 scripts
│       │   ├── run.py          # synthetic training
│       │   ├── run_a2d.py      # A2D training
│       │   ├── eval.py         # synthetic eval
│       │   ├── eval_a2d.py     # A2D eval
│       │   └── vis_a2d.py      # A2D visualization
│       └── v11/            # V11 scripts
│           ├── run.py          # synthetic training
│           └── eval.py         # synthetic eval
└── worldmodel/             # Original AdaWorld codebase (reference)
```

## Rules

### 1. Each training run gets its own directory

**Structure**: `result/{version}/{dataset}/{config}/`

Examples:
- `result/v10/synthetic/stage1/` — V10 on synthetic, default config
- `result/v11/synthetic/stage1/` — V11 on synthetic, default config
- `result/v11/synthetic/stage1_k01/` — V11 on synthetic, kl_beta=0.01
- `result/v10/a2d/yolo/` — V10 on A2D with YOLO detections
- `result/v10/a2d/gt/` — V10 on A2D with GT bbox
- `result/v11/gridworld/stage1/` — V11 on gridworld
- `result/v11/atari/stage1/` — V11 on atari

**Each run directory contains the SAME named files** — no prefix/suffix with run_name:

```
result/{dataset}/{run_name}/
  model.pt            ← always "model.pt", never "model_{name}.pt"
  ckpts/
    step1000.pt       ← always "step{step}.pt"
    step2000.pt
    ...
  losses/
    total.txt         ← always "{loss_name}.txt"
    recon.txt
    kl.txt
    ...
  results.json        ← always "results.json"
  eval.json           ← always "eval.json"
  latents.npz         ← always "latents.npz"
```

**Why**: Each file's identity (version, dataset, config) is encoded in the directory path, NOT in the filename. This eliminates the 162-file flat directory problem and allows scripts to use stable filenames.

### 2. Script organization — version-first directories

```
lam/scripts/
  common/                     # version-agnostic utilities
  {version}/                  # e.g. v10/, v11/
    run.py                    # training (dataset in filename: run_a2d.py)
    eval.py                   # evaluation
    ...
  _archived/                  # deprecated scripts
```

Version is in the directory, dataset in the filename. Examples:
- `lam/scripts/v11/run.py` — V11 synthetic training
- `lam/scripts/v10/run_a2d.py` — V10 A2D training
- `lam/scripts/v10/eval.py` — V10 synthetic eval
- `lam/scripts/common/analyze_umap.py` — shared analysis tool

Each script hardcodes `VERSION = "vXX"` and constructs paths as `result/{VERSION}/{dataset}/{config}/`.
- sys.path: references `lam/` with `os.path.dirname(__file__), "../.."`
- ROOT: project root with `os.path.dirname(__file__), "../../.."`

### 3. Never delete — archive instead

- Deprecated modules → `lam/lam/modules/_archived/`
- Deprecated scripts → `lam/scripts/_archived/`
- Deprecated results → `result/_archived/`
- `__init__.py` re-exports archived modules with try/except

### 4. Training script template

Every training script MUST:
- Define `VERSION = "vXX"` at module level
- Accept `--dataset` and `--config` arguments
- Save to `result/{VERSION}/{dataset}/{config}/` with standard filenames
- Print the output directory at the start

### 5. Evaluation script template

Every eval script MUST:
- Define `VERSION = "vXX"` at module level
- Accept `--dataset` and `--config` (auto-construct: `result/{VERSION}/{dataset}/{config}/`)
  OR `--checkpoint` (explicit path)
- Load from `{dir}/model.pt`, `{dir}/results.json`, `{dir}/latents.npz`
- Save results to `{dir}/eval.json`

### 6. Git conventions

- Commit message: `V<version>: <change> — <key result>`
- One commit per completed experiment
- NEVER commit: `result/`, `data/`, `pretrained/`, `reports/`, `*.pt`, `*.png`

### 7. Python environment

```bash
conda activate adaworld
PYTHONPATH=lam python lam/scripts/run_v11.py --dataset synthetic --name v11_stage1 ...
```

## Metrics Reference

| Metric | Description | Good | Excellent |
|---|---|---|---|
| Overall NMI | Cross-slot action clustering | > 0.20 | > 0.30 |
| Per-Slot NMI | Within-slot action clustering | > 0.50 | > 0.70 |
| Action Probe | z → action classifier accuracy | > 0.70 | > 0.85 |
| Leakage | z → slot identity classifier accuracy | ≤ 0.90 | ≤ 0.50 |
| PSNR (RGB) | Full-frame reconstruction | > 20 dB | > 25 dB |
| Actor-masked PSNR | Actor-region reconstruction | > 18 dB | > 22 dB |
| Δcopy | PSNR above copy baseline | > 3 dB | > 5 dB |
| z_var | Mean per-dim latent variance | > 0.3 | > 0.5 |
