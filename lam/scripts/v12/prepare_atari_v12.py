"""
Prepare video-only Atari V12 clips from blanchon/atari_parquet.

This first V12 Atari pass stores RGB clips and ALE/env actions only. Object
annotations are deliberately empty so object-level metrics can skip them
explicitly instead of reporting misleading numbers.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lam.v12_dataset import validate_v12_sample


FRAME_KEYS = ("frame", "image", "rgb", "observation", "obs")
ACTION_KEYS = ("action", "actions", "ale_action", "env_action")
EPISODE_KEYS = ("episode_id", "episode", "trajectory_id", "run_id")
FRAME_INDEX_KEYS = ("frame_index", "frame_idx", "timestep", "step", "index")
DONE_KEYS = ("done", "terminal", "terminated", "truncated")


GAME_DEFAULTS = {
    "freeway": {"num_frames": 5},
    "mspacman": {"num_frames": 5},
    "spaceinvaders": {"num_frames": 8},
}


def _first_key(row: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[str]:
    for key in candidates:
        if key in row:
            return key
    return None


def _to_rgb_array(value: Any) -> np.ndarray:
    if hasattr(value, "convert"):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, dict):
        if "array" in value:
            return _to_rgb_array(value["array"])
        if "bytes" in value and value["bytes"] is not None:
            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"), dtype=np.uint8)
        if "path" in value and value["path"]:
            from PIL import Image

            return np.asarray(Image.open(value["path"]).convert("RGB"), dtype=np.uint8)
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"cannot convert frame to RGB array, shape={arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _row_schema_error(row: Dict[str, Any]) -> str:
    return (
        "Unsupported atari_parquet schema. "
        f"Available columns: {sorted(row.keys())}. "
        f"Need one frame column from {FRAME_KEYS} and one action column from {ACTION_KEYS}."
    )


def _make_sample(
    frames: List[np.ndarray],
    actions: List[int],
    *,
    game: str,
    split: str,
    episode_id: Any,
    frame_start: int,
    sample_index: int,
) -> Dict[str, Any]:
    videos = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    T, _, H, W = videos.shape
    env_actions = torch.tensor(actions[: T - 1], dtype=torch.long)
    sample = {
        "videos": videos,
        "masks": torch.zeros((T, 0, H, W), dtype=torch.uint8),
        "bboxes": torch.zeros((T, 0, 4), dtype=torch.float32),
        "positions": torch.full((T, 0, 2), -1, dtype=torch.long),
        "actions": torch.empty((T - 1, 0), dtype=torch.long),
        "env_actions": env_actions,
        "actor_ids": torch.empty((0,), dtype=torch.long),
        "object_types": torch.empty((0,), dtype=torch.long),
        "valid_mask": torch.zeros((T, 0), dtype=torch.bool),
        "num_actors": 0,
        "metadata": {
            "episode_id": str(episode_id),
            "frame_index": int(frame_start),
            "task_name": f"Atari-{game}",
            "game_name": game,
            "split": split,
            "sample_index": int(sample_index),
            "has_object_annotations": False,
            "annotation_status": "video_only_missing_object_annotations",
        },
    }
    return sample


def _load_hf_split(dataset_name: str, game: str, streaming: bool):
    from datasets import load_dataset

    return load_dataset(dataset_name, split=game, streaming=streaming)


def _iter_direct_parquet_rows(dataset_name: str, game: str, cache_dir: Optional[str], batch_size: int):
    import requests
    import pyarrow.parquet as pq

    filename = f"data/{game}-00000-of-00001.parquet"
    cache_dir = cache_dir or os.path.join("data", "hf_cache", dataset_name.replace("/", "__"))
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, os.path.basename(filename))
    if not os.path.exists(local_path):
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/datasets/{dataset_name}/resolve/main/{filename}"
        tmp_path = f"{local_path}.tmp"
        with requests.get(url, stream=True, allow_redirects=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(tmp_path, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=f"download {game}",
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
        os.replace(tmp_path, local_path)
    parquet = pq.ParquetFile(local_path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        table = batch.to_pydict()
        keys = list(table.keys())
        n = len(table[keys[0]]) if keys else 0
        for i in range(n):
            yield {key: table[key][i] for key in keys}


def _iter_rows(ds: Iterable[Dict[str, Any]], limit_rows: int = 0) -> Iterable[Dict[str, Any]]:
    for idx, row in enumerate(ds):
        if limit_rows > 0 and idx >= limit_rows:
            break
        yield row


def prepare_game(
    *,
    dataset_name: str,
    game: str,
    out_root: str,
    train: int,
    val: int,
    num_frames: int,
    stride: int,
    streaming: bool,
    source: str,
    cache_dir: Optional[str],
    parquet_batch_size: int,
    limit_rows: int,
) -> Dict[str, Any]:
    if source == "direct_parquet":
        iterator = _iter_rows(
            _iter_direct_parquet_rows(dataset_name, game, cache_dir, parquet_batch_size),
            limit_rows,
        )
    else:
        ds = _load_hf_split(dataset_name, game, streaming)
        iterator = _iter_rows(ds, limit_rows)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"No rows available for split {game}") from exc

    frame_key = _first_key(first, FRAME_KEYS)
    action_key = _first_key(first, ACTION_KEYS)
    episode_key = _first_key(first, EPISODE_KEYS)
    frame_index_key = _first_key(first, FRAME_INDEX_KEYS)
    done_key = _first_key(first, DONE_KEYS)
    if frame_key is None or action_key is None:
        raise RuntimeError(_row_schema_error(first))

    rows = [first]
    rows.extend(iterator)

    out_dirs = {
        "train": os.path.join(out_root, f"atari_{game}", "train"),
        "val": os.path.join(out_root, f"atari_{game}", "val"),
    }
    for path in out_dirs.values():
        os.makedirs(path, exist_ok=True)

    counts = {"train": 0, "val": 0}
    max_total = train + val
    buffer_frames: List[np.ndarray] = []
    buffer_actions: List[int] = []
    buffer_indices: List[int] = []
    current_episode: Any = None

    def flush_windows(force: bool = False) -> None:
        nonlocal buffer_frames, buffer_actions, buffer_indices
        while len(buffer_frames) >= num_frames and sum(counts.values()) < max_total:
            split = "train" if counts["train"] < train else "val"
            if counts[split] >= (train if split == "train" else val):
                break
            frames = buffer_frames[:num_frames]
            acts = buffer_actions[: max(num_frames - 1, 0)]
            start = buffer_indices[0] if buffer_indices else 0
            sample = _make_sample(
                frames,
                acts,
                game=game,
                split=split,
                episode_id=current_episode if current_episode is not None else 0,
                frame_start=start,
                sample_index=counts[split],
            )
            errors = validate_v12_sample(
                {
                    **sample,
                    "videos": sample["videos"].float().permute(0, 2, 3, 1) / 255.0,
                    "masks": sample["masks"].float(),
                }
            )
            if errors:
                raise RuntimeError(f"V12 schema validation failed for {game}: {errors}")
            torch.save(sample, os.path.join(out_dirs[split], f"sample_{counts[split]:06d}.pt"))
            counts[split] += 1
            del buffer_frames[:stride]
            del buffer_actions[:stride]
            del buffer_indices[:stride]
        if force:
            buffer_frames = []
            buffer_actions = []
            buffer_indices = []

    for row in tqdm(rows, desc=f"atari {game}"):
        episode_id = row.get(episode_key, 0) if episode_key else 0
        if current_episode is None:
            current_episode = episode_id
        if episode_id != current_episode:
            flush_windows(force=True)
            current_episode = episode_id
        frame_index = int(row.get(frame_index_key, len(buffer_frames))) if frame_index_key else len(buffer_frames)
        buffer_frames.append(_to_rgb_array(row[frame_key]))
        buffer_actions.append(int(row[action_key]))
        buffer_indices.append(frame_index)
        flush_windows()
        if done_key and bool(row.get(done_key)):
            flush_windows(force=True)
            current_episode = None
        if counts["train"] >= train and counts["val"] >= val:
            break

    report = {
        "game": game,
        "counts": counts,
        "schema": {
            "frame_key": frame_key,
            "action_key": action_key,
            "episode_key": episode_key,
            "frame_index_key": frame_index_key,
            "done_key": done_key,
            "columns": sorted(first.keys()),
        },
        "num_frames": num_frames,
        "stride": stride,
    }
    if counts["train"] < train or counts["val"] < val:
        report["warning"] = "requested sample count not reached before data iterator ended"
    with open(os.path.join(out_root, f"atari_{game}", "dataset_config.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="blanchon/atari_parquet")
    parser.add_argument("--games", nargs="+", default=["freeway", "mspacman", "spaceinvaders"])
    parser.add_argument("--out_root", default="data/v12")
    parser.add_argument("--hf_endpoint", default=None, help="optional Hugging Face mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--source", choices=["direct_parquet", "datasets"], default="direct_parquet")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--parquet_batch_size", type=int, default=1024)
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=0, help="0 uses game defaults")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--limit_rows", type=int, default=0)
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    os.makedirs(args.out_root, exist_ok=True)
    reports = []
    for game in args.games:
        t = args.num_frames or GAME_DEFAULTS.get(game, {}).get("num_frames", 5)
        reports.append(
            prepare_game(
                dataset_name=args.dataset_name,
                game=game,
                out_root=args.out_root,
                train=args.train,
                val=args.val,
                num_frames=t,
                stride=args.stride,
                streaming=args.streaming,
                source=args.source,
                cache_dir=args.cache_dir,
                parquet_batch_size=args.parquet_batch_size,
                limit_rows=args.limit_rows,
            )
        )
    print(json.dumps({"dataset_name": args.dataset_name, "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
