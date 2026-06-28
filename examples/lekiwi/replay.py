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
import time
from contextlib import suppress

from lerobot.errors import DeviceNotConnectedError
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import busy_wait
from lekiwi_paths import local_dataset_root

DEFAULT_REPO_ID = "puffy/lekiwi-trash"
DEFAULT_REMOTE_IP = "192.168.3.16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a recorded LeKiwi episode.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episode", type=int, default=None, help="Episode index. Defaults to latest episode.")
    parser.add_argument("--remote-ip", default=DEFAULT_REMOTE_IP)
    parser.add_argument("--wait-for-host", action="store_true", help="Keep retrying until LeKiwi host is available.")
    return parser.parse_args()


def safe_disconnect(robot: LeKiwiClient) -> None:
    with suppress(Exception):
        if robot.is_connected:
            robot.disconnect()


args = parse_args()
episode_idx = args.episode
dataset_root = local_dataset_root(args.repo_id)
if episode_idx is None:
    metadata = LeRobotDataset(args.repo_id, root=dataset_root).meta
    episode_idx = metadata.total_episodes - 1

robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_lekiwi")
robot = LeKiwiClient(robot_config)

dataset = LeRobotDataset(args.repo_id, root=dataset_root, episodes=[episode_idx])
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
actions = episode_frames.select_columns("action")

try:
    while True:
        try:
            robot.connect()
            break
        except DeviceNotConnectedError:
            if not args.wait_for_host:
                raise
            print(
                f"Waiting for LeKiwi host at {args.remote_ip}. "
                "Start/restart lekiwi_host on the robot, then replay will continue..."
            )
            safe_disconnect(robot)
            time.sleep(2)

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print(f"Starting replay loop for {args.repo_id} episode {episode_idx} ({len(episode_frames)} frames)...")
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        action = {
            name: float(actions[idx]["action"][i])
            for i, name in enumerate(dataset.features["action"]["names"])
        }
        _ = robot.send_action(action)

        busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
finally:
    safe_disconnect(robot)
