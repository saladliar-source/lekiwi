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

from contextlib import suppress
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
from threading import Thread
import time
from urllib.parse import urlparse

import pandas as pd
import rerun as rr
from serial.tools import list_ports

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.record import record_loop
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener, sanity_check_dataset_robot_compatibility
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lekiwi_paths import local_dataset_root

NUM_EPISODES = 30
FPS = 30
EPISODE_TIME_SEC = None
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick up the trash and put it in the trash bin."
HF_REPO_ID = "puffy/lekiwi-trash"
DATASET_ROOT = local_dataset_root(HF_REPO_ID)
PLAY_SOUNDS = False
DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0180396-if00"
LEADER_PORT_POLL_SEC = 2.0
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8765


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def move_dataset_aside(dataset_root, suffix: str) -> None:
    backup_root = dataset_root.with_name(f"{dataset_root.name}.{suffix}-{timestamp()}")
    shutil.move(str(dataset_root), str(backup_root))
    print(f"Moved dataset directory to {backup_root}")


def trim_orphan_frames(dataset_root) -> None:
    info_path = dataset_root / "meta" / "info.json"
    data_root = dataset_root / "data"
    if not info_path.exists() or not data_root.exists():
        return

    import json

    with open(info_path) as f:
        total_episodes = int(json.load(f)["total_episodes"])

    for parquet_path in sorted(data_root.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        if "episode_index" not in df:
            continue

        orphan_mask = df["episode_index"] >= total_episodes
        if not orphan_mask.any():
            continue

        backup_path = parquet_path.with_suffix(f".orphan-{timestamp()}.parquet")
        shutil.copy2(parquet_path, backup_path)
        df = df.loc[~orphan_mask].copy()
        df.to_parquet(parquet_path, index=False)
        print(f"Trimmed orphan frames from {parquet_path}; backup saved to {backup_path}")


def ask_dataset_mode(dataset_root: Path) -> str:
    if not dataset_root.exists():
        return "create"

    data_dirs = [dataset_root / "data", dataset_root / "videos", dataset_root / "meta" / "episodes"]
    if not any(path.exists() for path in data_dirs):
        move_dataset_aside(dataset_root, "incomplete")
        return "create"

    trim_orphan_frames(dataset_root)

    while True:
        answer = input(
            f"数据集已存在：{dataset_root}\n"
            "输入 'a' 追加录制，输入 'c' 备份旧数据并清空重录，输入 'q' 退出 [a/c/q]: "
        ).strip().lower()
        if answer in {"a", "append", ""}:
            return "append"
        if answer in {"c", "clear"}:
            move_dataset_aside(dataset_root, "backup")
            return "create"
        if answer in {"q", "quit"}:
            raise SystemExit("Recording cancelled.")
        print("请输入 'a'、'c' 或 'q'。")


def safe_disconnect(device) -> None:
    with suppress(Exception):
        if getattr(device, "is_connected", True):
            device.disconnect()


def start_web_control(events: dict) -> tuple[ThreadingHTTPServer, str]:
    class ControlHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def send_text(self, text: str) -> None:
            payload = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def trigger(self, path: str) -> str | None:
            if path == "/finish":
                events["exit_early"] = True
                return "Finish requested. Current episode will be saved if it has frames."
            if path == "/rerecord":
                events["rerecord_episode"] = True
                events["exit_early"] = True
                return "Rerecord requested. Current episode will be discarded."
            if path == "/stop":
                events["stop_recording"] = True
                events["exit_early"] = True
                return "Stop requested. Recording will stop after the current loop exits."
            return None

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/status":
                self.send_text(
                    f"exit_early={events['exit_early']} "
                    f"rerecord_episode={events['rerecord_episode']} "
                    f"stop_recording={events['stop_recording']}"
                )
                return

            message = self.trigger(path)
            if message is not None:
                self.send_text(message)
            else:
                self.send_text(
                    "LeKiwi record control endpoint. Use the links shown in the Rerun record_controls panel."
                )

        def do_POST(self):
            path = urlparse(self.path).path
            message = self.trigger(path)
            if message is not None:
                self.send_text(message)
            else:
                self.send_response(404)
                self.end_headers()

    last_error = None
    for port in range(CONTROL_PORT, CONTROL_PORT + 10):
        try:
            server = ThreadingHTTPServer((CONTROL_HOST, port), ControlHandler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return server, f"http://{CONTROL_HOST}:{port}"
        except OSError as exc:
            last_error = exc

    raise RuntimeError(f"Could not start web control server: {last_error}")


def log_rerun_controls(control_url: str) -> None:
    rr.log(
        "record_controls",
        rr.TextDocument(
            f"""# LeKiwi Record Controls

Use these links from the existing Rerun page:

- [Finish and save current episode]({control_url}/finish)
- [Discard and rerecord current episode]({control_url}/rerecord)
- [Stop all recording]({control_url}/stop)
- [Show status]({control_url}/status)

If a link opens a small text response page, go back to the Rerun tab after clicking it.
""",
            media_type="text/markdown",
        ),
        static=True,
    )


def available_serial_ports() -> list[str]:
    ports = [port.device for port in list_ports.comports()]
    if ports:
        return sorted(dict.fromkeys(ports))

    candidates: list[str] = []
    for pattern in (
        "/dev/serial/by-id/*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
    ):
        candidates.extend(str(path) for path in sorted(Path("/").glob(pattern.removeprefix("/"))))

    return sorted(dict.fromkeys(candidates))


def find_leader_port(preferred_port: str) -> str | None:
    if Path(preferred_port).exists():
        return preferred_port

    ports = available_serial_ports()
    if len(ports) == 1:
        port = ports[0]
        print(f"Leader port {preferred_port} not found; using detected serial port {port}")
        return port

    return None


def resolve_leader_port(preferred_port: str) -> str:
    printed_help = False
    while True:
        port = find_leader_port(preferred_port)
        if port is not None:
            return port

        ports = available_serial_ports()
        detected = ", ".join(ports) if ports else "none"
        if not printed_help:
            print(
                "Could not find the SO101 leader serial port yet.\n"
                f"Preferred port: {preferred_port}\n"
                f"Detected serial ports: {detected}\n\n"
                "Keep this script running, then plug in / attach the leader. It will continue automatically.\n"
                "If the leader moved to another port, restart with for example:\n"
                "  LEKIWI_LEADER_PORT=/dev/ttyACM1 python examples/lekiwi/record.py\n\n"
                "If you are running inside WSL, attach the USB device from Windows first:\n"
                "  usbipd list\n"
                "  usbipd bind --busid <BUSID>\n"
                "  usbipd attach --wsl --busid <BUSID>\n\n"
                "If no USB device appears on native Linux, try a data-capable USB cable or another port.\n"
                "Press Ctrl-C to stop waiting."
            )
            printed_help = True
        else:
            print(f"Waiting for leader serial port... detected: {detected}")

        time.sleep(LEADER_PORT_POLL_SEC)

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="192.168.3.16", id="my_lekiwi")
# port in Linux: /dev/ttyACM0, /dev/ttyACM1, etc.
# port in MacOS: /dev/tty.usbmodemXXXXXXXXXXXX
# port in Windows: COMX / COMXX
leader_arm_config = SO101LeaderConfig(
    port=resolve_leader_port(os.environ.get("LEKIWI_LEADER_PORT", DEFAULT_LEADER_PORT)),
    id="R07254718",
)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard", backend="stdin")

# Initialize the robot and teleoperator
robot = LeKiwiClient(robot_config)
leader_arm = SO101Leader(leader_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# TODO(Steven): Update this example to use pipelines
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = None
listener = None
control_server = None
try:
    # Connect the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R1225XXXX`
    # where R1225XXXX is the robot serial number
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    _init_rerun(session_name="lekiwi_record")
    control_server, control_url = start_web_control(events)
    log_rerun_controls(control_url)

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    # Create or load the dataset only after hardware is connected, so failed
    # ports do not move or modify existing dataset directories.
    dataset_mode = ask_dataset_mode(DATASET_ROOT)
    if dataset_mode == "append":
        dataset = LeRobotDataset(HF_REPO_ID, root=DATASET_ROOT)
        sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
        dataset.episode_buffer = dataset.create_episode_buffer()
        dataset.start_image_writer(num_threads=4)
        print(f"Appending to dataset with {dataset.meta.total_episodes} existing episodes.")
    else:
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            root=DATASET_ROOT,
            use_videos=True,
            image_writer_threads=4,
        )

    print("Starting record loop...")
    print(
        "Controls in this terminal: press 'q' to finish and save the current episode, "
        "'r' to rerecord it, Esc to stop all recording."
    )
    print("Web controls are shown in the existing Rerun page under 'record_controls'.")
    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes}", PLAY_SOUNDS)

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
        episode_ended_early = events["exit_early"]
        events["exit_early"] = False

        # Reset the environment if not stopping or re-recording
        if not episode_ended_early and not events["stop_recording"] and (
            (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", PLAY_SOUNDS)
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=[leader_arm, keyboard],
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", PLAY_SOUNDS)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode only if at least one frame was recorded. This avoids
        # crashing when an episode is ended before the first dataset frame.
        if dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
            dataset.save_episode()
            recorded_episodes += 1
        else:
            print("No frames recorded for this episode; skipping save.")
            dataset.clear_episode_buffer()
except KeyboardInterrupt:
    print("Recording interrupted. Cleaning up current unsaved episode...")
    if dataset is not None:
        with suppress(Exception):
            dataset.clear_episode_buffer()
finally:
    log_say("Stop recording", PLAY_SOUNDS)
    if control_server is not None:
        with suppress(Exception):
            control_server.shutdown()
            control_server.server_close()
    if dataset is not None:
        with suppress(Exception):
            if dataset.image_writer is not None:
                dataset.image_writer.stop()
    safe_disconnect(robot)
    safe_disconnect(leader_arm)
    safe_disconnect(keyboard)
    if listener is not None:
        with suppress(Exception):
            listener.stop()
# Disable push by default
# dataset.push_to_hub()
