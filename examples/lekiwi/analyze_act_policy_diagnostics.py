#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


DEFAULT_REPO_ID = "puffy/lekiwi-trash"
DEFAULT_POLICY_PATH = "outputs/train/act_lekiwi_trash_clean11_smoke/checkpoints/001000/pretrained_model"
DEFAULT_OUTPUT_DIR = "outputs/lekiwi_policy_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LeKiwi ACT policy/data diagnostics.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def stack_actions(series: pd.Series) -> np.ndarray:
    return np.stack(series.map(np.asarray).to_numpy())


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    components = vh[:2]
    return x @ components.T, components


def summarize_dataset(root: Path, action_names: list[str]) -> tuple[pd.DataFrame, dict]:
    data = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    rows = []
    for ep, group in data.groupby("episode_index"):
        actions = stack_actions(group["action"])
        arm_delta = np.abs(np.diff(actions[:, :6], axis=0)).mean() if len(actions) > 1 else 0.0
        rows.append(
            {
                "episode_index": int(ep),
                "length": int(len(group)),
                "duration_s": len(group) / 30.0,
                "gripper_min": float(actions[:, 5].min()),
                "gripper_max": float(actions[:, 5].max()),
                "gripper_range": float(actions[:, 5].max() - actions[:, 5].min()),
                "gripper_mean": float(actions[:, 5].mean()),
                "arm_delta_mean": float(arm_delta),
                "base_max_abs": float(np.abs(actions[:, 6:9]).max()),
            }
        )

    all_actions = stack_actions(data["action"])
    action_summary = {
        "action_names": action_names,
        "action_min": all_actions.min(axis=0).tolist(),
        "action_max": all_actions.max(axis=0).tolist(),
        "action_std": all_actions.std(axis=0).tolist(),
        "base_all_zero": bool(np.allclose(all_actions[:, 6:9], 0.0)),
    }
    return pd.DataFrame(rows), action_summary


def make_sample_indices(dataset: LeRobotDataset, max_samples: int) -> list[int]:
    if len(dataset) <= max_samples:
        return list(range(len(dataset)))
    return np.round(np.linspace(0, len(dataset) - 1, max_samples)).astype(int).tolist()


def detach_cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def scale(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmin, vmax):
        return np.full_like(values, (out_min + out_max) / 2)
    return out_min + (values - vmin) * (out_max - out_min) / (vmax - vmin)


def write_svg_scatter(path: Path, points: np.ndarray, colors: np.ndarray, title: str) -> None:
    width, height, margin = 760, 560, 48
    xs = scale(points[:, 0], margin, width - margin)
    ys = scale(points[:, 1], height - margin, margin)
    unique = sorted(set(int(c) for c in colors))
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
    ]
    color_map = {value: palette[i % len(palette)] for i, value in enumerate(unique)}
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color_map[int(c)]}" opacity="0.78" />'
        for x, y, c in zip(xs, ys, colors)
    )
    legend = "\n".join(
        f'<text x="{width - 120}" y="{40 + i * 16}" font-size="12" fill="{color_map[value]}">ep {value}</text>'
        for i, value in enumerate(unique[:24])
    )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="28" font-size="18" font-family="sans-serif">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
{circles}
{legend}
</svg>
""",
        encoding="utf-8",
    )


def write_svg_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    width, height, margin = 760, 420, 48
    counts, edges = np.histogram(values, bins=30)
    bar_w = (width - 2 * margin) / len(counts)
    max_count = max(int(counts.max()), 1)
    bars = []
    for i, count in enumerate(counts):
        bar_h = (height - 2 * margin) * count / max_count
        x = margin + i * bar_w
        y = height - margin - bar_h
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 1:.2f}" height="{bar_h:.2f}" fill="#4c78a8"/>')
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="28" font-size="18" font-family="sans-serif">{title}</text>
<text x="{margin}" y="{height - 12}" font-size="12" font-family="sans-serif">{xlabel}: {edges[0]:.3f} to {edges[-1]:.3f}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
{''.join(bars)}
</svg>
""",
        encoding="utf-8",
    )


def write_svg_lines(path: Path, series: list[tuple[str, np.ndarray, str]], title: str, ylabel: str) -> None:
    width, height, margin = 920, 420, 48
    all_values = np.concatenate([values for _, values, _ in series])
    x_values = [np.linspace(margin, width - margin, len(values)) for _, values, _ in series]
    y_values = [scale(values, height - margin, margin) for _, values, _ in series]
    lines = []
    for (label, values, color), xs, ys in zip(series, x_values, y_values):
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        lines.append(
            f'<text x="{width - 210}" y="{42 + len(lines) * 8}" font-size="12" fill="{color}">{label}</text>'
        )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="28" font-size="18" font-family="sans-serif">{title}</text>
<text x="{margin}" y="{height - 12}" font-size="12" font-family="sans-serif">{ylabel}: {all_values.min():.3f} to {all_values.max():.3f}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
{''.join(lines)}
</svg>
""",
        encoding="utf-8",
    )


def write_svg_bars(path: Path, labels: list[str], true_values: np.ndarray, pred_values: np.ndarray) -> None:
    width, height, margin = 980, 460, 58
    max_value = max(float(true_values.max()), float(pred_values.max()), 1e-6)
    group_w = (width - 2 * margin) / len(labels)
    bars = []
    for i, label in enumerate(labels):
        x0 = margin + i * group_w
        for offset, value, color in ((0.15, true_values[i], "#4c78a8"), (0.50, pred_values[i], "#f58518")):
            bar_h = (height - 2 * margin) * value / max_value
            x = x0 + offset * group_w
            y = height - margin - bar_h
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{group_w * 0.28:.2f}" height="{bar_h:.2f}" fill="{color}"/>'
            )
        bars.append(
            f'<text x="{x0 + group_w / 2:.2f}" y="{height - margin + 12}" font-size="9" text-anchor="end" transform="rotate(-45 {x0 + group_w / 2:.2f},{height - margin + 12})">{label}</text>'
        )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="28" font-size="18" font-family="sans-serif">Action variation: dataset vs model prediction</text>
<text x="{width - 200}" y="40" font-size="12" fill="#4c78a8">true std</text>
<text x="{width - 200}" y="58" font-size="12" fill="#f58518">pred std</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222"/>
{''.join(bars)}
</svg>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = ACTPolicy.from_pretrained(args.policy_path)
    if args.device is not None:
        policy.config.device = args.device
    device = torch.device(policy.config.device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    delta_timestamps = {"action": [i / 30.0 for i in range(policy.config.chunk_size)]}
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    action_names = dataset.features["action"]["names"]
    dataset_root = dataset.root

    episode_summary, action_summary = summarize_dataset(dataset_root, action_names)
    episode_summary.to_csv(args.output_dir / "episode_action_summary.csv", index=False)

    indices = make_sample_indices(dataset, args.max_samples)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    policy.eval()
    mus = []
    logvars = []
    pred_chunks = []
    true_chunks = []
    pad_masks = []
    episode_indices = []
    frame_indices = []
    losses = []

    with torch.inference_mode():
        for raw_batch in loader:
            batch = preprocessor(raw_batch)
            model_batch = dict(batch)
            model_batch["observation.images"] = [model_batch[key] for key in policy.config.image_features]

            # The top-level ACT module gates VAE encoding on `self.training`.
            # Keep child modules in eval mode, but enable that branch for diagnostics.
            policy.model.training = True
            actions_hat_norm, (mu, logvar) = policy.model(model_batch)
            policy.model.training = False

            valid = ~batch["action_is_pad"].unsqueeze(-1)
            l1 = (torch.abs(actions_hat_norm - batch["action"]) * valid).sum() / valid.sum().clamp_min(1)
            losses.append(float(l1.item()))

            actions_hat = postprocessor(actions_hat_norm)
            mus.append(detach_cpu(mu))
            logvars.append(detach_cpu(logvar))
            pred_chunks.append(detach_cpu(actions_hat))
            true_chunks.append(detach_cpu(raw_batch["action"]))
            pad_masks.append(detach_cpu(raw_batch["action_is_pad"]).astype(bool))
            episode_indices.append(detach_cpu(raw_batch["episode_index"]).astype(int))
            frame_indices.append(detach_cpu(raw_batch["frame_index"]).astype(int))

    mu_arr = np.concatenate(mus, axis=0)
    logvar_arr = np.concatenate(logvars, axis=0)
    pred_arr = np.concatenate(pred_chunks, axis=0)
    true_arr = np.concatenate(true_chunks, axis=0)
    pad_arr = np.concatenate(pad_masks, axis=0)
    ep_arr = np.concatenate(episode_indices, axis=0)
    frame_arr = np.concatenate(frame_indices, axis=0)
    valid = ~pad_arr

    valid_pred = pred_arr[valid]
    valid_true = true_arr[valid]
    mae_by_dim = np.abs(valid_pred - valid_true).mean(axis=0)
    pred_std_by_dim = valid_pred.std(axis=0)
    true_std_by_dim = valid_true.std(axis=0)

    mu_pca, _ = pca2(mu_arr)
    kld_by_sample = -0.5 * (1 + logvar_arr - mu_arr**2 - np.exp(logvar_arr)).sum(axis=1)

    z_summary = {
        "samples": int(len(mu_arr)),
        "mu_abs_mean": float(np.abs(mu_arr).mean()),
        "mu_std_mean": float(mu_arr.std(axis=0).mean()),
        "mu_norm_mean": float(np.linalg.norm(mu_arr, axis=1).mean()),
        "mu_norm_std": float(np.linalg.norm(mu_arr, axis=1).std()),
        "logvar_mean": float(logvar_arr.mean()),
        "kld_mean": float(kld_by_sample.mean()),
        "kld_std": float(kld_by_sample.std()),
    }

    prediction_summary = {
        "mean_normalized_l1": float(np.mean(losses)),
        "mae_by_action": dict(zip(action_names, mae_by_dim.tolist())),
        "true_std_by_action": dict(zip(action_names, true_std_by_dim.tolist())),
        "pred_std_by_action": dict(zip(action_names, pred_std_by_dim.tolist())),
        "true_gripper_min": float(valid_true[:, 5].min()),
        "true_gripper_max": float(valid_true[:, 5].max()),
        "pred_gripper_min": float(valid_pred[:, 5].min()),
        "pred_gripper_max": float(valid_pred[:, 5].max()),
        "pred_gripper_std": float(valid_pred[:, 5].std()),
    }

    report = {
        "repo_id": args.repo_id,
        "policy_path": args.policy_path,
        "dataset_root": str(dataset_root),
        "dataset_total_episodes": int(dataset.meta.total_episodes),
        "dataset_total_frames": int(dataset.meta.total_frames),
        "action_summary": action_summary,
        "z_summary": z_summary,
        "prediction_summary": prediction_summary,
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")

    write_svg_scatter(args.output_dir / "z_mu_pca_by_episode.svg", mu_pca, ep_arr, "ACT VAE z mean (mu) PCA by episode")
    write_svg_hist(args.output_dir / "z_mu_norm_hist.svg", np.linalg.norm(mu_arr, axis=1), "ACT VAE z mean norm distribution", "||mu||")
    write_svg_lines(
        args.output_dir / "gripper_true_vs_pred_first.svg",
        [
            ("true gripper first action", true_arr[:, 0, 5], "#4c78a8"),
            ("pred gripper first action", pred_arr[:, 0, 5], "#f58518"),
        ],
        "First-step gripper target: true vs predicted",
        "arm_gripper.pos",
    )
    write_svg_bars(args.output_dir / "action_std_true_vs_pred.svg", action_names, true_std_by_dim, pred_std_by_dim)

    md = [
        "# LeKiwi ACT Diagnostics",
        "",
        f"- Dataset: `{args.repo_id}`",
        f"- Policy: `{args.policy_path}`",
        f"- Episodes/frames: `{dataset.meta.total_episodes}` / `{dataset.meta.total_frames}`",
        f"- Samples analyzed: `{len(mu_arr)}`",
        "",
        "## Key Metrics",
        "",
        f"- z mu abs mean: `{z_summary['mu_abs_mean']:.4f}`",
        f"- z mu norm mean/std: `{z_summary['mu_norm_mean']:.4f}` / `{z_summary['mu_norm_std']:.4f}`",
        f"- VAE KLD mean/std: `{z_summary['kld_mean']:.4f}` / `{z_summary['kld_std']:.4f}`",
        f"- normalized L1 mean: `{prediction_summary['mean_normalized_l1']:.4f}`",
        f"- true gripper range in sampled chunks: `{prediction_summary['true_gripper_min']:.3f}` to `{prediction_summary['true_gripper_max']:.3f}`",
        f"- predicted gripper range in sampled chunks: `{prediction_summary['pred_gripper_min']:.3f}` to `{prediction_summary['pred_gripper_max']:.3f}`",
        "",
        "## Files",
        "",
        "- `episode_action_summary.csv`",
        "- `diagnostics.json`",
        "- `z_mu_pca_by_episode.svg`",
        "- `z_mu_norm_hist.svg`",
        "- `gripper_true_vs_pred_first.svg`",
        "- `action_std_true_vs_pred.svg`",
    ]
    (args.output_dir / "README.md").write_text("\n".join(md) + "\n")

    print(json.dumps(report, indent=2))
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
