#!/usr/bin/env python

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import write_stats
from lekiwi_paths import local_dataset_root


DEFAULT_REPO_ID = "puffy/lekiwi-trash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete bad episodes from a local LeRobot dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None, help="Dataset root. Overrides --repo-id.")
    parser.add_argument("--delete", type=int, nargs="+", required=True, help="Episode indices to delete.")
    parser.add_argument("--yes", action="store_true", help="Actually modify the dataset. Default is dry-run.")
    parser.add_argument("--no-backup", action="store_true", help="Skip full dataset backup.")
    return parser.parse_args()


def feature_keys(info: dict, dtype_values: set[str]) -> set[str]:
    return {name for name, ft in info["features"].items() if ft.get("dtype") in dtype_values}


def as_array(values) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("Cannot compute stats over zero rows.")

    first = values.iloc[0] if isinstance(values, pd.Series) else values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(v, dtype=np.float64) for v in values])

    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def compute_stats(values) -> dict[str, np.ndarray]:
    arr = as_array(values)
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.asarray([arr.shape[0]], dtype=np.int64),
    }


def normalize_image_stat(value, stat_name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        flat_values = []
        for item in arr.reshape(-1):
            if hasattr(item, "tolist"):
                item = item.tolist()
            flat_values.extend(np.asarray(item, dtype=np.float64).reshape(-1).tolist())
        arr = np.asarray(flat_values, dtype=np.float64)
    else:
        arr = arr.astype(np.float64)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if stat_name != "count" and arr.size == 3:
        arr = arr.reshape(3, 1, 1)
    return arr


def aggregate_image_stats(episodes: pd.DataFrame, image_keys: set[str]) -> dict[str, dict[str, np.ndarray]]:
    stats_list = []
    for _, row in episodes.iterrows():
        ep_stats: dict[str, dict[str, np.ndarray]] = {}
        for col, value in row.items():
            if not col.startswith("stats/"):
                continue
            _, feature, stat_name = col.split("/", 2)
            if feature not in image_keys:
                continue
            ep_stats.setdefault(feature, {})[stat_name] = normalize_image_stat(value, stat_name)
        if ep_stats:
            stats_list.append(ep_stats)

    return aggregate_stats(stats_list) if stats_list else {}


def update_episode_stats(
    episodes: pd.DataFrame,
    data: pd.DataFrame,
    stats_keys: list[str],
) -> pd.DataFrame:
    episodes = episodes.copy()
    for row_idx, episode_index in episodes["episode_index"].items():
        ep_data = data.loc[data["episode_index"] == episode_index]
        for key in stats_keys:
            stats = compute_stats(ep_data[key])
            for stat_name, value in stats.items():
                episodes.at[row_idx, f"stats/{key}/{stat_name}"] = value
    return episodes


def make_episodes_parquet_safe(episodes: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    for col in episodes.columns:
        if not col.startswith("stats/"):
            continue
        if col.endswith("/count"):
            episodes[col] = episodes[col].apply(lambda value: np.asarray(value).reshape(-1).tolist())
        else:
            episodes[col] = episodes[col].apply(lambda value: np.asarray(value).tolist())
    return episodes


def main() -> None:
    args = parse_args()
    root = args.root if args.root is not None else local_dataset_root(args.repo_id)
    delete_set = set(args.delete)

    info_path = root / "meta/info.json"
    data_path = root / "data/chunk-000/file-000.parquet"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"

    info = json.loads(info_path.read_text())
    data = pd.read_parquet(data_path)
    episodes = pd.read_parquet(episodes_path)

    existing = set(int(ep) for ep in episodes["episode_index"].tolist())
    missing = sorted(delete_set - existing)
    if missing:
        raise ValueError(f"Episodes not found: {missing}. Existing episodes: {sorted(existing)}")

    keep_old = [int(ep) for ep in episodes["episode_index"].tolist() if int(ep) not in delete_set]
    if not keep_old:
        raise ValueError("Refusing to delete all episodes.")

    old_to_new = {old: new for new, old in enumerate(keep_old)}
    kept_data = data.loc[data["episode_index"].isin(keep_old)].copy()
    kept_episodes = episodes.loc[episodes["episode_index"].isin(keep_old)].copy()

    kept_data["episode_index"] = kept_data["episode_index"].map(old_to_new).astype(np.int64)
    kept_episodes["episode_index"] = kept_episodes["episode_index"].map(old_to_new).astype(np.int64)

    kept_data["index"] = np.arange(len(kept_data), dtype=np.int64)

    cursor = 0
    for row_idx, length in kept_episodes["length"].items():
        length = int(length)
        kept_episodes.at[row_idx, "dataset_from_index"] = cursor
        kept_episodes.at[row_idx, "dataset_to_index"] = cursor + length
        cursor += length

    scalar_or_vector_keys = [
        key
        for key, ft in info["features"].items()
        if ft.get("dtype") not in {"image", "video"}
        and key in kept_data.columns
    ]
    image_keys = feature_keys(info, {"image", "video"})

    kept_episodes = update_episode_stats(kept_episodes, kept_data, scalar_or_vector_keys)

    stats = {key: compute_stats(kept_data[key]) for key in scalar_or_vector_keys}
    stats.update(aggregate_image_stats(kept_episodes, image_keys))

    info["total_episodes"] = len(kept_episodes)
    info["total_frames"] = len(kept_data)
    info["splits"] = {"train": f"0:{len(kept_episodes)}"}

    print(f"Dataset: {root}")
    print(f"Delete episodes: {sorted(delete_set)}")
    print(f"Episode remap: {old_to_new}")
    print(f"New total_episodes: {info['total_episodes']}")
    print(f"New total_frames: {info['total_frames']}")

    if not args.yes:
        print("Dry-run only. Re-run with --yes to modify the dataset.")
        return

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = root.with_name(f"{root.name}.before-prune-{stamp}")
        shutil.copytree(root, backup)
        print(f"Backup saved to {backup}")

    kept_data.to_parquet(data_path, index=False)
    make_episodes_parquet_safe(kept_episodes).to_parquet(episodes_path, index=False)
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    write_stats(stats, root)
    print("Done.")


if __name__ == "__main__":
    main()
