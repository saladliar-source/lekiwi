#!/usr/bin/env python

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lekiwi_paths import local_dataset_root


DEFAULT_REPO_ID = "puffy/lekiwi-trash"
DEFAULT_OUTPUT_DIR = Path("outputs/lekiwi_data_qc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-check LeKiwi episodes and optionally delete failed episodes."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None, help="Dataset root. Overrides --repo-id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--gripper-index", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--min-relative-frames",
        type=float,
        default=0.5,
        help="Fail episodes shorter than this fraction of the dataset median frame count.",
    )
    parser.add_argument("--min-gripper-range", type=float, default=3.0)
    parser.add_argument("--min-close-frames", type=int, default=1)
    parser.add_argument("--close-delta-threshold", type=float, default=-0.5)
    parser.add_argument(
        "--max-action-jump",
        type=float,
        default=None,
        help="Optional max allowed absolute frame-to-frame action jump.",
    )
    parser.add_argument("--ignore-gripper", action="store_true", help="Do not fail episodes by gripper metrics.")
    parser.add_argument("--delete", type=int, nargs="*", default=[], help="Also delete these episodes explicitly.")
    parser.add_argument("--yes", action="store_true", help="Actually delete failed episodes. Default is dry-run.")
    parser.add_argument("--no-backup", action="store_true", help="Pass --no-backup to prune_dataset_episodes.py.")
    return parser.parse_args()


def load_parquet_group(root: Path, subdir: str) -> pd.DataFrame:
    files = sorted((root / subdir).glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / subdir}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def stack_values(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.empty((0, 0), dtype=np.float64)

    rows = []
    for value in values:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        rows.append(arr)

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Inconsistent vector widths inside one episode.")

    return np.stack(rows)


def finite_or_empty(arr: np.ndarray) -> bool:
    return arr.size == 0 or bool(np.isfinite(arr).all())


def analyze_episode(
    episode_index: int,
    ep_data: pd.DataFrame,
    meta_row: pd.Series | None,
    args: argparse.Namespace,
    median_frames: float,
) -> dict:
    reasons: list[str] = []
    frame_count = int(len(ep_data))
    meta_length = int(meta_row["length"]) if meta_row is not None and "length" in meta_row else None

    if frame_count == 0:
        reasons.append("no_data_frames")
    if meta_row is None:
        reasons.append("missing_metadata")
    if meta_length is not None and meta_length != frame_count:
        reasons.append(f"length_mismatch:{meta_length}!={frame_count}")
    if frame_count < args.min_frames:
        reasons.append(f"too_short:{frame_count}<{args.min_frames}")
    if median_frames > 0 and frame_count < median_frames * args.min_relative_frames:
        reasons.append(
            f"too_short_relative:{frame_count}<{args.min_relative_frames:.2f}*median({median_frames:.1f})"
        )
    if args.max_frames is not None and frame_count > args.max_frames:
        reasons.append(f"too_long:{frame_count}>{args.max_frames}")

    metrics = {
        "episode_index": episode_index,
        "frames": frame_count,
        "metadata_length": meta_length,
        "relative_frames": frame_count / median_frames if median_frames > 0 else None,
        "action_finite": True,
        "action_jump_max": None,
        "gripper_min": None,
        "gripper_max": None,
        "gripper_range": None,
        "gripper_close_frames": None,
    }

    if frame_count > 0 and args.action_key in ep_data.columns:
        try:
            actions = stack_values(ep_data[args.action_key])
        except Exception as exc:
            actions = np.empty((0, 0), dtype=np.float64)
            reasons.append(f"bad_action_shape:{exc}")

        action_finite = finite_or_empty(actions)
        metrics["action_finite"] = action_finite
        if not action_finite:
            reasons.append("action_has_nan_or_inf")

        if actions.shape[0] > 1 and actions.shape[1] > 0:
            metrics["action_jump_max"] = float(np.max(np.abs(np.diff(actions, axis=0))))
            if args.max_action_jump is not None and metrics["action_jump_max"] > args.max_action_jump:
                reasons.append(f"action_jump_too_large:{metrics['action_jump_max']:.3f}>{args.max_action_jump}")

        if not args.ignore_gripper:
            if actions.shape[1] <= args.gripper_index:
                reasons.append(f"missing_gripper_dim:{actions.shape[1]}<={args.gripper_index}")
            elif actions.shape[0] > 0:
                gripper = actions[:, args.gripper_index]
                metrics["gripper_min"] = float(np.min(gripper))
                metrics["gripper_max"] = float(np.max(gripper))
                metrics["gripper_range"] = float(np.max(gripper) - np.min(gripper))
                close_frames = int(np.sum(np.diff(gripper) < args.close_delta_threshold))
                metrics["gripper_close_frames"] = close_frames

                if metrics["gripper_range"] < args.min_gripper_range:
                    reasons.append(f"gripper_range_too_small:{metrics['gripper_range']:.3f}<{args.min_gripper_range}")
                if close_frames < args.min_close_frames:
                    reasons.append(f"not_enough_gripper_close:{close_frames}<{args.min_close_frames}")
    elif frame_count > 0:
        reasons.append(f"missing_action_key:{args.action_key}")

    metrics["status"] = "bad" if reasons else "ok"
    metrics["reasons"] = ";".join(reasons)
    return metrics


def run_prune(root: Path, repo_id: str, episodes: list[int], no_backup: bool) -> None:
    script = Path(__file__).with_name("prune_dataset_episodes.py")
    command = [sys.executable, str(script), "--delete", *[str(ep) for ep in episodes], "--yes"]
    if root is not None:
        command.extend(["--root", str(root)])
    else:
        command.extend(["--repo-id", repo_id])
    if no_backup:
        command.append("--no-backup")

    subprocess.run(command, check=True)


def run_repair(root: Path, repo_id: str, no_backup: bool) -> None:
    script = Path(__file__).with_name("repair_dataset_after_failed_record.py")
    command = [sys.executable, str(script)]
    if root is not None:
        command.extend(["--root", str(root)])
    else:
        command.extend(["--repo-id", repo_id])
    if no_backup:
        command.append("--no-backup")

    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = args.root if args.root is not None else local_dataset_root(args.repo_id)

    data = load_parquet_group(root, "data")
    episodes = load_parquet_group(root, "meta/episodes")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_by_episode = {
        int(row["episode_index"]): row for _, row in episodes.iterrows()
    }
    data_episodes = sorted(int(ep) for ep in data["episode_index"].dropna().unique())
    meta_episodes = sorted(meta_by_episode)
    all_episodes = sorted(set(data_episodes) | set(meta_episodes))
    frame_counts = [
        int((data["episode_index"] == episode_index).sum())
        for episode_index in all_episodes
    ]
    median_frames = float(np.median(frame_counts)) if frame_counts else 0.0

    rows = []
    for episode_index in all_episodes:
        ep_data = data.loc[data["episode_index"] == episode_index]
        rows.append(
            analyze_episode(episode_index, ep_data, meta_by_episode.get(episode_index), args, median_frames)
        )

    report = pd.DataFrame(rows).sort_values("episode_index")
    bad_from_qc = report.loc[report["status"] == "bad", "episode_index"].astype(int).tolist()
    delete_episodes = sorted(set(bad_from_qc) | set(args.delete))

    csv_path = args.output_dir / "episode_qc_report.csv"
    json_path = args.output_dir / "episode_qc_summary.json"
    report.to_csv(csv_path, index=False)

    summary = {
        "dataset": str(root),
        "total_episodes_seen": len(all_episodes),
        "bad_from_qc": bad_from_qc,
        "explicit_delete": sorted(set(args.delete)),
        "delete_episodes": delete_episodes,
        "criteria": {
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "median_frames": median_frames,
            "min_relative_frames": args.min_relative_frames,
            "min_gripper_range": None if args.ignore_gripper else args.min_gripper_range,
            "min_close_frames": None if args.ignore_gripper else args.min_close_frames,
            "close_delta_threshold": None if args.ignore_gripper else args.close_delta_threshold,
            "max_action_jump": args.max_action_jump,
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"Dataset: {root}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Bad episodes from QC: {bad_from_qc if bad_from_qc else 'none'}")
    if args.delete:
        print(f"Explicit delete episodes: {sorted(set(args.delete))}")
    print(f"Delete episodes: {delete_episodes if delete_episodes else 'none'}")

    if not delete_episodes:
        print("No episodes to delete.")
        return

    if not args.yes:
        print("Dry-run only. Re-run with --yes to delete these episodes.")
        return

    if set(meta_episodes).issubset(delete_episodes):
        raise ValueError("Refusing to delete all metadata episodes.")

    missing_from_metadata = sorted(set(delete_episodes) - set(meta_episodes))
    if missing_from_metadata:
        print(
            "Some delete episodes are present in data but missing metadata; "
            f"repairing first: {missing_from_metadata}"
        )
        run_repair(args.root, args.repo_id, args.no_backup)

    run_prune(args.root, args.repo_id, delete_episodes, args.no_backup)


if __name__ == "__main__":
    main()
