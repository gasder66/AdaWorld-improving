"""
并行多 Slot 实验启动器
在 GPU 2 和 GPU 3 上同时运行不同实验，最大化 GPU 利用率。

调度策略：
  GPU 2: v1_baseline (batch=128, 500步, 短) → v4_no_comp (batch=48, 1000步, 长)
  GPU 3: v4_comp (batch=48, 1000步, 中等)

利用梯度检查点来支持更大的 batch size，充分使用显存。
"""
import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_SCRIPT = os.path.join(BASE_DIR, "run_single.py")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "slot_attention_exp_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp(gpu, name, num_slots, competition, batch_size, steps, log_file=None, no_checkpoint=False, aux_loss=0.0):
    """启动一个子进程运行实验，返回 Popen 对象。"""
    cmd = [
        sys.executable, SINGLE_SCRIPT,
        "--gpu", str(gpu),
        "--name", name,
        "--num_slots", str(num_slots),
        "--batch_size", str(batch_size),
        "--steps", str(steps),
    ]
    if competition:
        cmd.append("--competition")
    if no_checkpoint:
        cmd.append("--no_checkpoint")
    if aux_loss > 0:
        cmd.extend(["--aux_loss", str(aux_loss)])

    if log_file is None:
        log_file = os.path.join(RESULTS_DIR, f"log_{name}.txt")

    log_fp = open(log_file, "w")
    print(f"  Launching: {' '.join(cmd)} -> {log_file}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True, env=env)
    return proc, log_fp


def wait_for(proc, log_fp, name):
    """等待一个进程完成。"""
    print(f"  Waiting for {name} to complete...")
    proc.wait()
    log_fp.close()
    if proc.returncode == 0:
        print(f"  ✓ {name} completed successfully")
    else:
        print(f"  ✗ {name} failed with return code {proc.returncode}")
    return proc.returncode == 0


def main():
    print("=" * 70)
    print("  Parallel Multi-Slot Experiment Launcher")
    print("  Using gradient checkpointing + larger batch for GPU utilization")
    print("=" * 70)

    # ===== 阶段 1: 并行启动两个实验 =====
    print("\n[Phase 1] Launching 2 experiments in parallel:")
    print("  GPU 2: v1_baseline (1 slot, batch=128, 500 steps, w/ checkpoint)")
    print("  GPU 3: v4_comp (4 slots + competition, batch=32, 1000 steps, w/o checkpoint)")

    # GPU 2: v1_baseline - 1 slot 模型小，用大 batch + checkpoint
    p1, f1 = run_exp(
        gpu=2, name="v1_baseline_par", num_slots=1, competition=False,
        batch_size=128, steps=500,
        log_file=os.path.join(RESULTS_DIR, "log_v1_baseline_par.txt")
    )

    # GPU 3: v4_comp - 竞争机制 + 无 checkpoint（避免 NaN），batch 适中
    p2, f2 = run_exp(
        gpu=3, name="v4_comp_par", num_slots=4, competition=True,
        batch_size=32, steps=1000, no_checkpoint=True,
        log_file=os.path.join(RESULTS_DIR, "log_v4_comp_par.txt")
    )

    t0 = time.time()

    # ===== 阶段 2: 等 v1_baseline 完成后，在 GPU 2 上启动 v4_no_comp =====
    print("\n[Phase 1] Waiting for experiments...")
    ok1 = wait_for(p1, f1, "v1_baseline_par")
    elapsed1 = time.time() - t0
    print(f"  v1_baseline_par done in {elapsed1:.0f}s")

    # GPU 2 空闲了，启动 v4_no_comp
    print("\n[Phase 2] GPU 2 is free, launching v4_no_comp:")
    print("  GPU 2: v4_no_comp (4 slots, no competition, batch=64, 1000 steps, w/ checkpoint)")
    p3, f3 = run_exp(
        gpu=2, name="v4_no_comp_par", num_slots=4, competition=False,
        batch_size=64, steps=1000,
        log_file=os.path.join(RESULTS_DIR, "log_v4_no_comp_par.txt")
    )

    # 同时等 v4_comp 和 v4_no_comp
    print("\n[Phase 2] Waiting for both remaining experiments...")
    ok2 = wait_for(p2, f2, "v4_comp_par")
    ok3 = wait_for(p3, f3, "v4_no_comp_par")

    # ===== 阶段 3: 辅助损失实验（在空闲的 GPU 上）=====
    print("\n" + "=" * 70)
    print("  [Phase 3] Auxiliary Loss Experiments")
    print("=" * 70)

    # GPU 2 和 3 都空闲了，启动两个 aux loss 实验
    p4, f4 = run_exp(
        gpu=2, name="v4_auxloss", num_slots=4, competition=False,
        batch_size=48, steps=800, aux_loss=0.1,
        log_file=os.path.join(RESULTS_DIR, "log_v4_auxloss.txt")
    )
    p5, f5 = run_exp(
        gpu=3, name="v4_comp_auxloss", num_slots=4, competition=True,
        batch_size=24, steps=800, aux_loss=0.1, no_checkpoint=True,
        log_file=os.path.join(RESULTS_DIR, "log_v4_comp_auxloss.txt")
    )

    print("[Phase 3] Waiting for aux loss experiments...")
    ok4 = wait_for(p4, f4, "v4_auxloss")
    ok5 = wait_for(p5, f5, "v4_comp_auxloss")
    total_time = time.time() - t0

    # ===== 收集结果 =====
    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"  Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  v1_baseline_par:    {'✓' if ok1 else '✗'}")
    print(f"  v4_comp_par:        {'✓' if ok2 else '✗'}")
    print(f"  v4_no_comp_par:     {'✓' if ok3 else '✗'}")
    print(f"  v4_auxloss:         {'✓' if ok4 else '✗'}")
    print(f"  v4_comp_auxloss:    {'✓' if ok5 else '✗'}")
    print(f"\n  Results directory: {RESULTS_DIR}")

    # 打印结果摘要
    import json
    for exp_name in ["v1_baseline_par", "v4_comp_par", "v4_no_comp_par",
                     "v4_auxloss", "v4_comp_auxloss"]:
        result_file = os.path.join(RESULTS_DIR, f"results_{exp_name}.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                r = json.load(f)
            print(f"\n  {exp_name}:")
            print(f"    ARI={r.get('ari', 'N/A'):.4f}, PSNR={r.get('psnr', 'N/A'):.1f}dB")
            print(f"    Peak GPU mem={r.get('peak_memory_gb', 'N/A')}GB")
            print(f"    Time={r.get('training_time_s', 'N/A'):.0f}s")
            print(f"    Aux loss weight={r.get('aux_loss_weight', 'N/A')}")
            if 'slot_aris' in r:
                print(f"    Slot ARIs={[f'{a:.4f}' for a in r['slot_aris']]}")


if __name__ == "__main__":
    # 设置环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    main()