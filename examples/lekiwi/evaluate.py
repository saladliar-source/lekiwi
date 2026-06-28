# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.errors import DeviceNotConnectedError
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.record import record_loop
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lekiwi_paths import local_dataset_root

DEFAULT_REMOTE_IP = "192.168.3.16"
DEFAULT_POLICY_PATH = "outputs/train/smolvla_lekiwi_trash_clean11/checkpoints/002000/pretrained_model"
FPS = 30
EPISODE_TIME_SEC = None
TASK_DESCRIPTION = "Pick up the trash and put it in the trash bin."
PLAY_SOUNDS = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LeKiwi policy inference without recording a dataset.")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--remote-ip", default=DEFAULT_REMOTE_IP)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--wait-for-host", action="store_true")
    parser.add_argument("--clip-action-repo-id", default="puffy/lekiwi-trash")
    parser.add_argument("--no-action-clip", action="store_true")
    parser.add_argument("--max-action-delta", type=float, default=3.0)
    parser.add_argument("--print-action-every-sec", type=float, default=1.0)
    return parser.parse_args()


def safe_disconnect(device) -> None:
    with suppress(Exception):
        if getattr(device, "is_connected", True):
            device.disconnect()


def make_observation_adapter(policy_config):
    input_keys = set(policy_config.input_features)
    needs_base_camera_names = "observation.images.camera1" in input_keys
    state_feature = policy_config.input_features.get("observation.state")
    expected_state_dim = state_feature.shape[0] if state_feature is not None else None

    if not needs_base_camera_names and expected_state_dim != 6:
        return lambda obs: obs

    print(
        "Using LeKiwi -> SmolVLA-base observation adapter "
        "(front/wrist -> camera1/2/3, 9D state -> first 6 arm joints)."
    )

    def adapt(obs):
        obs = dict(obs)

        if needs_base_camera_names:
            front = obs.get("front")
            wrist = obs.get("wrist", front)
            fallback = front if front is not None else wrist
            if fallback is None:
                fallback = np.zeros((480, 640, 3), dtype=np.uint8)

            obs.setdefault("camera1", front if front is not None else fallback)
            obs.setdefault("camera2", wrist if wrist is not None else fallback)
            # SmolVLA base was trained with 3 camera slots. Duplicate the wrist
            # view rather than feeding an all-black out-of-distribution image.
            obs.setdefault("camera3", wrist if wrist is not None else fallback)

        if expected_state_dim == 6 and "observation.state" in obs:
            obs["observation.state"] = obs["observation.state"][:6]

        return obs

    return adapt


def load_action_bounds(repo_id: str) -> dict[str, tuple[float, float]]:
    root = local_dataset_root(repo_id)
    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    if not info_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"Could not find local dataset stats under {root}")

    info = json.loads(info_path.read_text())
    stats = json.loads(stats_path.read_text())
    names = info["features"]["action"]["names"]
    mins = np.asarray(stats["action"]["min"], dtype=np.float32).reshape(-1)
    maxs = np.asarray(stats["action"]["max"], dtype=np.float32).reshape(-1)
    return {name: (float(mins[i]), float(maxs[i])) for i, name in enumerate(names)}


def make_safe_action_processor(base_processor, action_names, bounds, max_delta, print_every_sec):
    last_action: dict[str, float] = {}
    last_print_t = 0.0

    def process(action_and_obs):
        nonlocal last_print_t
        action, obs = action_and_obs
        action = dict(action)

        if bounds is not None:
            for key, value in list(action.items()):
                if key in bounds:
                    lo, hi = bounds[key]
                    action[key] = float(np.clip(value, lo, hi))

        if max_delta is not None and max_delta > 0:
            state = obs.get("observation.state")
            for key, value in list(action.items()):
                if key not in last_action:
                    if state is not None and key in action_names:
                        idx = action_names.index(key)
                        if idx < len(state):
                            last_action[key] = float(state[idx])
                    else:
                        last_action[key] = float(value)

                limited = float(np.clip(value, last_action[key] - max_delta, last_action[key] + max_delta))
                action[key] = limited
                last_action[key] = limited

        now = time.perf_counter()
        if print_every_sec > 0 and now - last_print_t >= print_every_sec:
            printable = ", ".join(f"{key}={value:.2f}" for key, value in action.items())
            print(f"policy action: {printable}")
            last_print_t = now

        return base_processor((action, obs))

    return process


args = parse_args()

# Create the robot configuration & robot
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_lekiwi")
robot = LeKiwiClient(robot_config)

# Create policy and load saved processors from the checkpoint directory.
policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
policy_cls = get_policy_class(policy_config.type)
policy = policy_cls.from_pretrained(args.policy_path, config=policy_config)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=args.policy_path,
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)

listener = None
try:
    # Connect the robot
    # To connect you already should have this script running on LeKiwi:
    # `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    while True:
        try:
            robot.connect()
            break
        except DeviceNotConnectedError:
            if not args.wait_for_host:
                raise
            print(
                f"Waiting for LeKiwi host at {args.remote_ip}. "
                "Start/restart lekiwi_host on the robot, then evaluate will continue..."
            )
            safe_disconnect(robot)
            time.sleep(2)

    # TODO(Steven): Update this example to use pipelines
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot_observation_processor = make_observation_adapter(policy.config)
    action_bounds = None
    if not args.no_action_clip:
        action_bounds = load_action_bounds(args.clip_action_repo_id)
        print(f"Clipping policy actions to local dataset range from {args.clip_action_repo_id}.")
    robot_action_processor = make_safe_action_processor(
        robot_action_processor,
        list(robot.action_features),
        action_bounds,
        args.max_action_delta,
        args.print_action_every_sec,
    )

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    _init_rerun(session_name="lekiwi_evaluate")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting pure inference loop. Press Ctrl-C to stop.")
    while not events["stop_recording"]:
        log_say(
            "Running policy inference",
            PLAY_SOUNDS,
        )

        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=None,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
except KeyboardInterrupt:
    print("Inference interrupted.")
finally:
    log_say("Stop inference", PLAY_SOUNDS)
    safe_disconnect(robot)
    if listener is not None:
        with suppress(Exception):
            listener.stop()
