#!/usr/bin/env python

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats, estimate_num_samples
from lerobot.datasets.utils import write_stats
from lerobot.datasets.video_utils import get_video_duration_in_s
from lekiwi_paths import local_dataset_root


DEFAULT_REPO_ID = "puffy/lekiwi-trash"
STAT_NAMES = {"min", "max", "mean", "std", "count"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair a local LeRobot dataset after a record run wrote data/video but failed to save episode metadata."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None, help="Dataset root. Overrides --repo-id.")
    parser.add_argument("--yes", action="store_true", help="Deprecated; repairs are applied by default.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be repaired.")
    parser.add_argument("--no-backup", action="store_true", help="Skip full dataset backup.")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def stats_columns(episodes: pd.DataFrame) -> list[str]:
    return [col for col in episodes.columns if col.startswith("stats/")]


def parse_stats_column(col: str) -> tuple[str, str]:
    parts = col.split("/")
    if len(parts) < 3 or parts[-1] not in STAT_NAMES:
        raise ValueError(f"Unexpected stats column: {col}")
    return "/".join(parts[1:-1]), parts[-1]


def as_feature_array(values) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("Cannot compute stats over zero rows.")

    first = values.iloc[0] if isinstance(values, pd.Series) else values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(value, dtype=np.float64) for value in values])

    return np.asarray(values, dtype=np.float64)


def compute_stats(values) -> dict[str, np.ndarray]:
    arr = as_feature_array(values)
    keepdims = arr.ndim == 1
    return {
        "min": np.min(arr, axis=0, keepdims=keepdims),
        "max": np.max(arr, axis=0, keepdims=keepdims),
        "mean": np.mean(arr, axis=0, keepdims=keepdims),
        "std": np.std(arr, axis=0, keepdims=keepdims),
        "count": np.asarray([arr.shape[0]], dtype=np.int64),
    }


def normalize_stat_value(value, feature_key: str, stat_name: str, info: dict) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        flattened = []
        for item in arr.reshape(-1):
            flattened.extend(np.asarray(item).reshape(-1).tolist())
        arr = np.asarray(flattened)

    if stat_name == "count":
        return arr.astype(np.int64).reshape(-1)

    arr = arr.astype(np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    dtype = info["features"][feature_key]["dtype"]
    if dtype in {"image", "video"} and arr.size == 3:
        arr = arr.reshape(3, 1, 1)

    return arr


def row_to_episode_stats(row: pd.Series, info: dict) -> dict[str, dict[str, np.ndarray]]:
    ep_stats: dict[str, dict[str, np.ndarray]] = {}
    for col, value in row.items():
        if not col.startswith("stats/"):
            continue
        feature_key, stat_name = parse_stats_column(col)
        ep_stats.setdefault(feature_key, {})[stat_name] = normalize_stat_value(
            value, feature_key, stat_name, info
        )
    return ep_stats


def make_episodes_parquet_safe(episodes: pd.DataFrame) -> pd.DataFrame:
    episodes = episodes.copy()
    for col in stats_columns(episodes):
        episodes[col] = episodes[col].apply(lambda value: normalize_to_flat_list(value))

        feature_key, stat_name = parse_stats_column(col)
        if stat_name != "count":
            if feature_key.startswith("observation.images.") and episodes[col].map(len).eq(3).all():
                episodes[col] = episodes[col].apply(
                    lambda value: np.asarray(value, dtype=np.float64).reshape(3, 1, 1).tolist()
                )
        elif episodes[col].map(len).eq(1).all():
            episodes[col] = episodes[col].apply(lambda value: int(value[0]))

    for col in stats_columns(episodes):
        feature_key, stat_name = parse_stats_column(col)
        if stat_name == "count" or feature_key.startswith("observation.images."):
            continue
        if episodes[col].map(len).eq(1).all():
            episodes[col] = episodes[col].apply(lambda value: float(value[0]))

    return episodes


def normalize_to_flat_list(value) -> list:
    arr = np.asarray(value)
    if arr.dtype != object:
        return arr.reshape(-1).tolist()

    flattened = []
    for item in arr.reshape(-1):
        flattened.extend(np.asarray(item).reshape(-1).tolist())
    return flattened


def load_task_names(root: Path) -> dict[int, str]:
    tasks_path = root / "meta/tasks.parquet"
    if not tasks_path.exists():
        return {}

    tasks_df = pd.read_parquet(tasks_path)
    return {int(row["task_index"]): str(index) for index, row in tasks_df.iterrows()}


def task_names_for_episode(ep_data: pd.DataFrame, task_names: dict[int, str]) -> list[str]:
    names = [
        task_names.get(int(task_index), str(task_index))
        for task_index in sorted(ep_data["task_index"].dropna().unique())
    ]
    return names or ["Pick up the trash and put it in the trash bin."]


def video_metadata_for_orphan(
    root: Path,
    info: dict,
    episodes: pd.DataFrame,
    video_key: str,
    episode_length: int,
) -> dict[str, int | float]:
    latest_ep = episodes.iloc[-1]
    chunk_idx = int(latest_ep[f"videos/{video_key}/chunk_index"])
    file_idx = int(latest_ep[f"videos/{video_key}/file_index"])
    from_timestamp = float(latest_ep[f"videos/{video_key}/to_timestamp"])

    video_path = root / info["video_path"].format(
        video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
    )
    try:
        to_timestamp = float(get_video_duration_in_s(video_path))
    except Exception:
        to_timestamp = from_timestamp + episode_length / float(info["fps"])

    if to_timestamp <= from_timestamp:
        to_timestamp = from_timestamp + episode_length / float(info["fps"])

    return {
        f"videos/{video_key}/chunk_index": chunk_idx,
        f"videos/{video_key}/file_index": file_idx,
        f"videos/{video_key}/from_timestamp": from_timestamp,
        f"videos/{video_key}/to_timestamp": to_timestamp,
    }


def build_orphan_episode_row(
    root: Path,
    info: dict,
    episodes: pd.DataFrame,
    data: pd.DataFrame,
    episode_index: int,
) -> pd.Series:
    ep_data = data.loc[data["episode_index"] == episode_index].copy()
    if ep_data.empty:
        raise ValueError(f"No frames found for orphan episode {episode_index}.")

    row = {col: None for col in episodes.columns}
    row["episode_index"] = int(episode_index)
    row["tasks"] = task_names_for_episode(ep_data, load_task_names(root))
    row["length"] = int(len(ep_data))
    row["data/chunk_index"] = int(episodes.iloc[-1]["data/chunk_index"])
    row["data/file_index"] = int(episodes.iloc[-1]["data/file_index"])
    row["dataset_from_index"] = int(ep_data["index"].min())
    row["dataset_to_index"] = int(ep_data["index"].max()) + 1
    row["meta/episodes/chunk_index"] = int(episodes.iloc[-1]["meta/episodes/chunk_index"])
    row["meta/episodes/file_index"] = int(episodes.iloc[-1]["meta/episodes/file_index"])

    for key, feature in info["features"].items():
        dtype = feature["dtype"]
        if key not in ep_data.columns or dtype in {"image", "video", "string"}:
            continue
        for stat_name, value in compute_stats(ep_data[key]).items():
            col = f"stats/{key}/{stat_name}"
            if col in row:
                row[col] = value

    previous = episodes.iloc[-1]
    image_keys = [key for key, feature in info["features"].items() if feature["dtype"] in {"image", "video"}]
    for key in image_keys:
        for stat_name in STAT_NAMES:
            col = f"stats/{key}/{stat_name}"
            if col not in row:
                continue
            if stat_name == "count":
                row[col] = np.asarray([estimate_num_samples(len(ep_data))], dtype=np.int64)
            else:
                row[col] = normalize_stat_value(previous[col], key, stat_name, info)

        row.update(video_metadata_for_orphan(root, info, episodes, key, len(ep_data)))

    return pd.Series(row)


def main() -> None:
    args = parse_args()
    root = args.root if args.root is not None else local_dataset_root(args.repo_id)

    info_path = root / "meta/info.json"
    data_path = root / "data/chunk-000/file-000.parquet"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"

    info = json.loads(info_path.read_text())
    data = pd.read_parquet(data_path)
    episodes = pd.read_parquet(episodes_path)

    episode_rows = set(int(ep) for ep in episodes["episode_index"].tolist())
    data_episodes = sorted(int(ep) for ep in data["episode_index"].dropna().unique())
    orphan_episodes = [ep for ep in data_episodes if ep not in episode_rows]

    repaired = episodes.copy()
    for episode_index in orphan_episodes:
        if episode_index != len(repaired):
            raise ValueError(
                f"Cannot auto-repair non-sequential orphan episode {episode_index}; "
                f"metadata currently has {len(repaired)} episodes."
            )
        repaired = pd.concat(
            [repaired, build_orphan_episode_row(root, info, repaired, data, episode_index).to_frame().T],
            ignore_index=True,
        )

    repaired = make_episodes_parquet_safe(repaired)
    episode_stats = [row_to_episode_stats(row, info) for _, row in repaired.iterrows()]
    stats = aggregate_stats(episode_stats)

    info["total_episodes"] = len(repaired)
    info["total_frames"] = len(data)
    info["splits"] = {"train": f"0:{len(repaired)}"}

    print(f"Dataset: {root}")
    print(f"Data episodes: {data_episodes}")
    print(f"Metadata episodes before: {sorted(episode_rows)}")
    print(f"Orphan episodes to rescue: {orphan_episodes or 'none'}")
    print(f"New total_episodes: {info['total_episodes']}")
    print(f"New total_frames: {info['total_frames']}")

    if args.dry_run:
        print("Dry-run only. Re-run without --dry-run to modify the dataset.")
        return

    if not args.no_backup:
        backup = root.with_name(f"{root.name}.before-repair-{timestamp()}")
        shutil.copytree(root, backup)
        print(f"Backup saved to {backup}")

    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(episodes_path, index=False)
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    write_stats(stats, root)
    print("Done.")


if __name__ == "__main__":
    main()
