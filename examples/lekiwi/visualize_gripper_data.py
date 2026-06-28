# !/usr/bin/env python

from argparse import ArgumentParser
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from lekiwi_paths import local_dataset_root


DEFAULT_DATASET_ROOT = local_dataset_root("puffy/lekiwi-trash")
DEFAULT_OUTPUT_DIR = Path("outputs/lekiwi_data_viz")


def parse_args():
    parser = ArgumentParser(description="Visualize LeKiwi gripper action values from a local dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the local LeRobot dataset root.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to plot. If omitted, all valid episodes are plotted together.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the plot image will be written.",
    )
    return parser.parse_args()


def load_episode_count(dataset_root: Path) -> int:
    info = pd.read_json(dataset_root / "meta" / "info.json", typ="series")
    return int(info["total_episodes"])


def load_actions(dataset_root: Path) -> pd.DataFrame:
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    return pd.read_parquet(data_path, columns=["episode_index", "frame_index", "action"])


def plot_gripper(df: pd.DataFrame, output_path: Path, episode: int | None) -> None:
    width = 1200
    height = 520 if episode is None else 420
    margin_left = 72
    margin_right = 28
    margin_top = 48
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if episode is None:
        groups = list(df.groupby("episode_index"))
        title = "LeKiwi gripper action: arm_gripper.pos"
    else:
        sub = df[df["episode_index"] == episode]
        if sub.empty:
            raise ValueError(f"Episode {episode} was not found in the dataset.")
        groups = [(episode, sub)]
        title = f"Episode {episode} gripper action"

    max_frame = max(int(group["frame_index"].max()) for _, group in groups)
    max_frame = max(max_frame, 1)

    def x_scale(frame_index: float) -> float:
        return margin_left + (frame_index / max_frame) * plot_width

    def y_scale(value: float) -> float:
        clamped = max(0.0, min(100.0, value))
        return margin_top + (100.0 - clamped) / 100.0 * plot_height

    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#ca8a04",
        "#7c3aed",
        "#0891b2",
        "#db2777",
        "#ea580c",
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="28" font-family="sans-serif" font-size="20" font-weight="700" fill="#111827">{escape(title)}</text>',
    ]

    for tick in range(0, 101, 20):
        y = y_scale(tick)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#6b7280">{tick}</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>'
    )

    legend_x = margin_left
    legend_y = height - 24
    for idx, (ep, sub) in enumerate(groups):
        actions = np.stack(sub["action"].to_list())
        points = " ".join(
            f"{x_scale(float(frame)):.2f},{y_scale(float(gripper)):.2f}"
            for frame, gripper in zip(sub["frame_index"], actions[:, 5], strict=False)
        )
        color = colors[idx % len(colors)]
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if episode is None:
            item_x = legend_x + idx * 120
            lines.append(f'<line x1="{item_x}" y1="{legend_y}" x2="{item_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            lines.append(
                f'<text x="{item_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#374151">episode {ep}</text>'
            )

    lines.append(
        f'<text x="{width / 2:.2f}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#374151">frame_index</text>'
    )
    lines.append(
        f'<text x="18" y="{height / 2:.2f}" transform="rotate(-90 18 {height / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#374151">arm_gripper.pos (0-100)</text>'
    )
    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(output_path.resolve())


def main() -> None:
    args = parse_args()
    total_episodes = load_episode_count(args.dataset_root)
    df = load_actions(args.dataset_root)

    # Ignore rows from interrupted saves that were written to data but not metadata.
    df = df[df["episode_index"] < total_episodes]

    if args.episode is None:
        output_path = args.output_dir / "lekiwi_gripper_action_all.svg"
    else:
        output_path = args.output_dir / f"lekiwi_gripper_action_episode_{args.episode}.svg"

    plot_gripper(df, output_path, args.episode)


if __name__ == "__main__":
    main()
