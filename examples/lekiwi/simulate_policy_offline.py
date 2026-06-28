#!/usr/bin/env python

import argparse
from contextlib import nullcontext
from copy import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
try:
    from lekiwi_paths import local_dataset_root
except ModuleNotFoundError:
    from examples.lekiwi.lekiwi_paths import local_dataset_root

DEFAULT_REPO_ID = "puffy/lekiwi-trash"
DEFAULT_POLICY_PATH = "outputs/train/smolvla_lekiwi_trash/checkpoints/last/pretrained_model"
DEFAULT_SUMMARY_PATH = Path("outputs/lekiwi_offline_sim/summary.json")
DEFAULT_PLOT_DIR = Path("outputs/lekiwi_offline_sim/plots")
DEFAULT_HF_DATASETS_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "hf_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline LeKiwi simulation: replay dataset observations through a trained policy and visualize predictions."
    )
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root. Defaults to examples/lekiwi local dataset path.")
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to simulate. Defaults to the latest episode unless --all-episodes is set.",
    )
    parser.add_argument("--all-episodes", action="store_true", help="Run the simulation across all episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per episode.")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback speed for visualization.")
    parser.add_argument("--device", default=None, help="Override policy device, e.g. cpu or cuda.")
    parser.add_argument(
        "--prediction-mode",
        choices=("aligned", "rollout"),
        default="aligned",
        help=(
            "How to compare policy outputs against dataset actions. "
            "'aligned' recomputes the current first-step action at every frame. "
            "'rollout' drains the model's action queue like online execution."
        ),
    )
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun visualization.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory to save gt/pred comparison plots.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable saving gt/pred comparison plots.")
    parser.add_argument(
        "--detailed-summary",
        action="store_true",
        help="Include per-dimension action averages/errors in summary.json.",
    )
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_scalar(value: Any) -> int | float:
    arr = to_numpy(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    raise ValueError(f"Expected scalar value, got shape {arr.shape}")


def to_image_hwc(value: Any) -> np.ndarray:
    # Hugging Face video/image features may arrive wrapped in 1-element containers.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    arr = to_numpy(value)
    if arr.dtype == object and arr.size == 1:
        inner = arr.reshape(-1)[0]
        if inner is not value:
            return to_image_hwc(inner)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 1 and arr.size == 1 and arr.dtype.kind in {"U", "S", "O"}:
        raise ValueError(f"Image value looks like a path/container instead of pixels: {arr!r}")
    if arr.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def to_vector(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim == 2:
        arr = arr[-1]
    return arr.astype(np.float32).reshape(-1)


def make_policy_observation_adapter(policy_config: PreTrainedConfig):
    input_keys = set(policy_config.input_features)
    state_feature = policy_config.input_features.get("observation.state")
    expected_state_dim = state_feature.shape[0] if state_feature is not None else None
    expects_base_cameras = "observation.images.camera1" in input_keys

    def adapt(raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        obs = {key: value for key, value in raw_obs.items() if key in input_keys}

        if expects_base_cameras:
            front = raw_obs.get("observation.images.front")
            wrist = raw_obs.get("observation.images.wrist", front)
            fallback = front if front is not None else wrist
            if fallback is None:
                fallback = np.zeros((480, 640, 3), dtype=np.uint8)

            obs.setdefault("observation.images.camera1", front if front is not None else fallback)
            obs.setdefault("observation.images.camera2", wrist if wrist is not None else fallback)
            obs.setdefault("observation.images.camera3", wrist if wrist is not None else fallback)

        if expected_state_dim is not None and "observation.state" in raw_obs:
            state = raw_obs["observation.state"]
            if state.shape[0] >= expected_state_dim:
                obs["observation.state"] = state[:expected_state_dim]
            else:
                padded = np.zeros((expected_state_dim,), dtype=state.dtype)
                padded[: state.shape[0]] = state
                obs["observation.state"] = padded

        return obs

    return adapt


def get_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    episodes = dataset.meta.episodes
    if hasattr(episodes, "iloc"):
        row = episodes.iloc[episode_index]
    else:
        row = episodes[episode_index]
    start = int(row["dataset_from_index"])
    end = int(row["dataset_to_index"])
    return start, end


class OfflineLeKiwiDatasetEnv(gym.Env):
    metadata = {"render_modes": ["none", "rerun"], "render_fps": 30}

    def __init__(self, dataset: LeRobotDataset, episode_index: int, max_steps: int | None = None):
        super().__init__()
        self.dataset = dataset
        self.episode_index = episode_index
        self.robot_type = dataset.meta.robot_type
        self.action_names = list(dataset.features["action"]["names"])

        start, end = get_episode_bounds(dataset, episode_index)
        self.frame_indices = list(range(start, end))
        if max_steps is not None:
            self.frame_indices = self.frame_indices[:max_steps]
        if not self.frame_indices:
            raise ValueError(f"Episode {episode_index} has no frames to simulate.")

        first_item = dataset[self.frame_indices[0]]
        first_obs = self._build_raw_observation(first_item)
        first_action = self._extract_teacher_action(first_item)

        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(low=0, high=255, shape=value.shape, dtype=np.uint8)
                if key.startswith("observation.images.")
                else spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)
                for key, value in first_obs.items()
            }
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=first_action.shape,
            dtype=np.float32,
        )

        self._cursor = 0
        self._current_item: dict[str, Any] | None = None

    def _extract_teacher_action(self, item: dict[str, Any]) -> np.ndarray:
        action = to_numpy(item["action"])
        if action.ndim == 2:
            action = action[0]
        return action.astype(np.float32).reshape(-1)

    def _build_raw_observation(self, item: dict[str, Any]) -> dict[str, np.ndarray]:
        raw_obs: dict[str, np.ndarray] = {}
        for key, value in item.items():
            if not key.startswith("observation."):
                continue
            if key.endswith("_is_pad"):
                continue
            if key.startswith("observation.images."):
                try:
                    raw_obs[key] = to_image_hwc(value)
                except ValueError as exc:
                    raise ValueError(f"Failed to decode {key} from dataset item: {exc}") from exc
            else:
                raw_obs[key] = to_vector(value)
        return raw_obs

    def _build_info(self, item: dict[str, Any], pred_action: np.ndarray | None = None) -> dict[str, Any]:
        teacher_action = self._extract_teacher_action(item)
        info = {
            "episode_index": int(to_scalar(item["episode_index"])),
            "frame_index": int(to_scalar(item["frame_index"])),
            "task": item["task"],
            "teacher_action": teacher_action,
        }
        if pred_action is not None:
            pred_action = pred_action.astype(np.float32).reshape(-1)
            abs_error = np.abs(pred_action - teacher_action)
            info["pred_action"] = pred_action
            info["abs_error"] = abs_error
            info["mae"] = float(abs_error.mean())
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._cursor = 0
        self._current_item = self.dataset[self.frame_indices[self._cursor]]
        return self._build_raw_observation(self._current_item), self._build_info(self._current_item)

    def step(self, action: dict[str, float] | np.ndarray):
        if self._current_item is None:
            raise RuntimeError("Call reset() before step().")

        if isinstance(action, dict):
            pred_action = np.asarray([action[name] for name in self.action_names], dtype=np.float32)
        else:
            pred_action = np.asarray(action, dtype=np.float32).reshape(-1)

        info = self._build_info(self._current_item, pred_action=pred_action)
        reward = -info["mae"]

        self._cursor += 1
        terminated = self._cursor >= len(self.frame_indices)
        truncated = False

        if terminated:
            next_obs = self._build_raw_observation(self._current_item)
        else:
            self._current_item = self.dataset[self.frame_indices[self._cursor]]
            next_obs = self._build_raw_observation(self._current_item)

        return next_obs, reward, terminated, truncated, info


def action_tensor_to_dict(action_tensor: torch.Tensor, action_names: list[str]) -> dict[str, float]:
    action_np = to_numpy(action_tensor).reshape(-1)
    action_np = action_np[: len(action_names)]
    return {name: float(action_np[i]) for i, name in enumerate(action_names)}


def predict_first_aligned_action(
    observation: dict[str, np.ndarray],
    policy,
    device: torch.device,
    preprocessor,
    postprocessor,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
) -> torch.Tensor:
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)
        action_chunk = policy.predict_action_chunk(observation)
        action_chunk = postprocessor(action_chunk)
        action = action_chunk[:, 0, :].squeeze(0).to("cpu")

    return action


def summarize_episode_metrics(
    metrics: list[dict[str, Any]], action_names: list[str], detailed: bool = False
) -> dict[str, Any]:
    teacher = np.stack([step["teacher_action"] for step in metrics])
    pred = np.stack([step["pred_action"] for step in metrics])
    mae = np.stack([step["abs_error"] for step in metrics])
    mae_by_dim = mae.mean(axis=0)
    worst_dim = int(np.argmax(mae_by_dim))
    summary = {
        "episode_index": metrics[0]["episode_index"],
        "task": metrics[0]["task"],
        "frames": len(metrics),
        "mean_reward": float(np.mean([-step["mae"] for step in metrics])),
        "mean_mae": float(mae.mean()),
        "max_mae": float(mae.max()),
        "worst_action_dim": action_names[worst_dim],
        "worst_action_mae": float(mae_by_dim[worst_dim]),
    }
    if detailed:
        summary["action_names"] = action_names
        summary["teacher_action_mean"] = teacher.mean(axis=0).tolist()
        summary["pred_action_mean"] = pred.mean(axis=0).tolist()
        summary["mae_by_dim"] = mae_by_dim.tolist()
    return summary


def save_episode_plot(
    metrics: list[dict[str, Any]], action_names: list[str], episode_index: int, output_path: Path
) -> None:
    teacher = np.stack([step["teacher_action"] for step in metrics])
    pred = np.stack([step["pred_action"] for step in metrics])
    frame_index = np.arange(len(metrics))

    cols = 2
    rows = int(np.ceil(len(action_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(3 * rows, 4)), squeeze=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(action_names):
        ax = axes_flat[idx]
        ax.plot(frame_index, teacher[:, idx], label="gt", linewidth=1.5)
        ax.plot(frame_index, pred[:, idx], label="pred", linewidth=1.2, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("frame")
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for idx in range(len(action_names), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"Episode {episode_index}: gt vs pred", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_sim_step(obs: dict[str, np.ndarray], info: dict[str, Any], action_names: list[str]) -> None:
    log_rerun_data(observation=obs)
    rr.log("sim/episode_index", rr.Scalar(info["episode_index"]))
    rr.log("sim/frame_index", rr.Scalar(info["frame_index"]))
    rr.log("sim/action_mae", rr.Scalar(info["mae"]))
    for idx, name in enumerate(action_names):
        rr.log(f"teacher_action/{name}", rr.Scalar(float(info["teacher_action"][idx])))
        rr.log(f"policy_action/{name}", rr.Scalar(float(info["pred_action"][idx])))
        rr.log(f"action_abs_error/{name}", rr.Scalar(float(info["abs_error"][idx])))


def main() -> None:
    args = parse_args()
    dataset_root = args.root if args.root is not None else local_dataset_root(args.repo_id)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_DATASETS_CACHE))

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    if args.device is not None:
        policy_config.device = args.device
    device = torch.device(policy_config.device)

    policy_cls = get_policy_class(policy_config.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=policy_config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    dataset = LeRobotDataset(args.repo_id, root=dataset_root, delta_timestamps=delta_timestamps)

    if args.all_episodes:
        episode_indices = list(range(dataset.meta.total_episodes))
    else:
        episode_indices = [args.episode if args.episode is not None else dataset.meta.total_episodes - 1]

    adapt_observation = make_policy_observation_adapter(policy.config)

    if not args.no_rerun:
        _init_rerun(session_name="lekiwi_offline_sim")
        rr.log(
            "sim/info",
            rr.TextDocument(
                "\n".join(
                    [
                        "# LeKiwi Offline Simulation",
                        f"- policy: `{args.policy_path}`",
                        f"- dataset: `{dataset_root}`",
                        f"- episodes: `{episode_indices}`",
                    ]
                ),
                media_type="text/markdown",
            ),
            static=True,
        )

    overall_summary: dict[str, Any] = {
        "policy_path": str(args.policy_path),
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "prediction_mode": args.prediction_mode,
        "episodes": [],
    }

    for episode_index in episode_indices:
        env = OfflineLeKiwiDatasetEnv(dataset, episode_index=episode_index, max_steps=args.max_steps)
        raw_obs, info = env.reset()
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        episode_metrics: list[dict[str, Any]] = []
        done = False
        while not done:
            policy_obs = adapt_observation(raw_obs)
            current_obs = raw_obs
            if args.prediction_mode == "aligned":
                action_tensor = predict_first_aligned_action(
                    observation=policy_obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=info["task"],
                    robot_type=env.robot_type,
                )
            else:
                action_tensor = predict_action(
                    observation=policy_obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=info["task"],
                    robot_type=env.robot_type,
                )
            action_dict = action_tensor_to_dict(action_tensor, env.action_names)
            raw_obs, reward, terminated, truncated, step_info = env.step(action_dict)
            step_info["reward"] = reward
            episode_metrics.append(step_info)

            if not args.no_rerun:
                log_sim_step(current_obs, step_info, env.action_names)

            done = terminated or truncated
            info = step_info
            if not done and args.fps > 0:
                time.sleep(1.0 / args.fps)

        episode_summary = summarize_episode_metrics(
            episode_metrics,
            action_names=env.action_names,
            detailed=args.detailed_summary,
        )
        if not args.no_plots:
            plot_path = args.plot_dir / f"episode_{episode_index:03d}_gt_vs_pred.png"
            save_episode_plot(episode_metrics, env.action_names, episode_index, plot_path)
            episode_summary["plot_path"] = str(plot_path)
            print(f"Saved plot to {plot_path}")
        overall_summary["episodes"].append(episode_summary)
        print(
            f"Episode {episode_index}: frames={episode_summary['frames']} "
            f"mean_mae={episode_summary['mean_mae']:.4f} max_mae={episode_summary['max_mae']:.4f}"
        )

    if overall_summary["episodes"]:
        mean_maes = [ep["mean_mae"] for ep in overall_summary["episodes"]]
        overall_summary["mean_episode_mae"] = float(np.mean(mean_maes))
        overall_summary["num_episodes"] = len(overall_summary["episodes"])

    args.summary_path.write_text(json.dumps(overall_summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote simulation summary to {args.summary_path}")


if __name__ == "__main__":
    main()
