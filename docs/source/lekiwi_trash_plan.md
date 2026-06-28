# LeKiwi 自动扔垃圾执行方案

这份方案描述一条比较稳的路线：先做一个小范围、可重复、可验证的自动扔垃圾闭环；等第一版跑通以后，再逐步加入更强的导航和泛化能力。

## 目标

训练 LeKiwi 在受控工作区内完成“捡起垃圾并放进垃圾桶”的任务。

第一版任务范围：

- 垃圾出现在固定工作区域内。
- 垃圾桶位置固定，或者只做轻微变化。
- 垃圾类型先选容易夹取、安全轻量的物体，例如纸团、小盒子、塑料瓶盖。
- 机器人同时使用底盘和机械臂。
- 观测输入包含 `front` 和 `wrist` 两路 RGB 图像，以及机器人状态。

第一版先不做：

- 全屋探索。
- SLAM 建图。
- 复杂避障。
- 形状、重量差异很大的未知物体。

## 阶段 1：稳定遥操作

录数据之前，先用 `teleoperate.py` 确认完整硬件链路稳定。

树莓派端：

```bash
python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R12254718
```

主控电脑端：

```bash
python examples/lekiwi/teleoperate.py
```

检查项：

- 能看到 `front` 相机。
- 能看到 `wrist` 相机。
- 相机画面方向正确。
- Leader 臂能控制 Follower 臂。
- 夹爪能正常开合。
- `w`、`a`、`s`、`d` 能控制底盘平移。
- `z`、`x` 能控制底盘旋转。
- `r`、`f` 能调节底盘速度，如果该功能启用。
- Rerun 可视化延迟能接受，至少足够用于录数据。

当前工作区已经使用过的重要配置：

- 远程 LeKiwi IP：`192.168.3.16`
- WSL 里的 Leader 串口：`/dev/ttyACM0`
- Leader 标定 ID：`R07254718`
- LeKiwi Follower ID：`R12254718`

远程树莓派连接信息：

- SSH 地址：`puffy@192.168.3.16`
- 用户名：`puffy`
- 密码：单个空格字符

连接命令：

```bash
ssh puffy@192.168.3.16
```

注意：这份信息只适合保存在本地私有文档里，不要提交到公开仓库或公开分享。

远程排障常用命令：

```bash
hostname -I
v4l2-ctl --list-devices
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R12254718
```

## 阶段 2：录制扔垃圾数据

使用 `examples/lekiwi/record.py` 录制示教数据。每一条 episode 都应该包含完整任务流程。

第一批数据建议：

- 最少：`30` 条成功 episode。
- 更推荐：`50` 条成功 episode。
- 更强的第一版模型：`100+` 条成功 episode。
- 每条时长：约 `20-40` 秒。
- 重置时长：约 `10` 秒。

推荐任务描述：

```text
Pick up the trash and put it in the trash bin.
```

示教流程：

1. 机器人从相对一致的初始位置开始。
2. 驱动底盘或调整姿态靠近垃圾。
3. 调整机械臂和夹爪位置。
4. 抓取垃圾。
5. 移动到垃圾桶附近。
6. 把垃圾放进垃圾桶。
7. 停止或回到一个中性姿态。

数据质量规则：

- 第一批数据只录成功 episode。
- 垃圾位置要逐步变化，不要一开始就在整个房间随机摆。
- 第一版模型里，垃圾桶尽量保持固定。
- 模型没跑通前，先不要加入复杂杂物和干扰物。
- 每个垃圾位置都多录几条。
- 保持灯光和相机曝光稳定。

录制前需要修改：

- `examples/lekiwi/record.py`

关键配置建议：

```python
robot_config = LeKiwiClientConfig(remote_ip="192.168.3.16", id="my_lekiwi")
leader_arm_config = SO101LeaderConfig(
    port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0180396-if00",
    id="R07254718",
)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard", backend="stdin")
TASK_DESCRIPTION = "Pick up the trash and put it in the trash bin."
HF_REPO_ID = "<hf_username>/lekiwi-trash"
```

当前 `examples/lekiwi/record.py` 已按上述配置修改。开始录制前只需要把 `HF_REPO_ID` 改成你想保留的逻辑数据集名，例如：

```python
HF_REPO_ID = "puffy/lekiwi-trash"
```

录制命令：

```bash
python examples/lekiwi/record.py
```

当前默认本地落盘目录：

```text
/home/puffy/lerobot/lerobot/datasets/lekiwi-trash
```

如果要把第三方 `v2.1` 数据集转到同一目录体系下，例如 `pepijn223/lekiwi_pen`，可以执行：

```bash
python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id pepijn223/lekiwi_pen \
  --root /home/puffy/lerobot/lerobot/datasets/lekiwi_pen
```

录制快捷键：

- 推荐使用 Rerun 页面里的 `record_controls` 面板。
- `Finish and save current episode`：结束并保存当前 episode。
- `Discard and rerecord current episode`：丢弃当前 episode 并重录。
- `Stop all recording`：停止整次录制。
- 终端备用键：`q` 保存当前 episode，`r` 重录当前 episode，`Esc` 停止整次录制。

### 删除坏 episode

如果某条 episode 是空录、静止片段、失败示教，训练前应该删掉。使用本地清洗脚本：

```bash
python examples/lekiwi/prune_dataset_episodes.py --delete 3
```

默认是 dry-run，只打印将要删除什么，不会改数据。确认无误后加 `--yes`：

```bash
python examples/lekiwi/prune_dataset_episodes.py --delete 3 --yes
```

一次删除多条：

```bash
python examples/lekiwi/prune_dataset_episodes.py --delete 1 3 5 --yes
```

脚本会做这些事：

- 先备份整个数据集目录。
- 删除指定 episode 的 `data` 行。
- 删除对应的 `meta/episodes` 行。
- 将保留的 episode 重新编号为 `0..N-1`。
- 更新 `meta/info.json` 的 `total_episodes`、`total_frames` 和 `splits`。
- 重算并写回 `meta/stats.json`。

注意：脚本不会重编码视频文件；视频文件里可能仍有旧片段尾巴，但 metadata 不再引用被删除的 episode，训练和 replay 不会使用它。

## 阶段 3：训练和评估策略模型

当前优先训练 SmolVLA。它更适合后续接入任务文本和视觉语言先验；ACT 保留为对照和调试备选。LeKiwi 的动作空间是 `9D`，不要直接拿 `lerobot/smolvla_base` 做真机评估，base 模型默认输出维度不等于 LeKiwi 动作维度。

当前数据状态：

- 当前活动数据集：`puffy/lekiwi-trash`
- 当前本地路径：`/home/puffy/lerobot/lerobot/datasets/lekiwi-trash`
- 当前清洗后保存：`15` 条 episode
- 最近一次坏 episode：`15`，已通过质检脚本删除
- 当前数据仍偏少，适合小训练和真机 sanity check；正式可用建议继续录到 `30-50+` 条成功 episode。

训练前先跑质检：

```bash
python examples/lekiwi/qc_dataset_episodes.py
```

如果质检报告确认要删除坏 episode，再执行：

```bash
python examples/lekiwi/qc_dataset_episodes.py --yes
```

### SmolVLA 训练命令

当前推荐先跑 `3000` 步，确认 loss、动作范围和真机行为是否正常：

```bash
lerobot-train \
  --dataset.repo_id=puffy/lekiwi-trash \
  --dataset.root=/home/puffy/lerobot/lerobot/datasets/lekiwi-trash \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolvla_lekiwi_trash_clean15 \
  --job_name=smolvla_lekiwi_trash_clean15 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=3000 \
  --batch_size=4 \
  --save_freq=500 \
  --log_freq=50
```

训练完成后评估：

```bash
python examples/lekiwi/evaluate.py \
  --policy-path outputs/train/smolvla_lekiwi_trash_clean15/checkpoints/003000/pretrained_model \
  --wait-for-host \
  --max-action-delta 0.5
```

如果你当前更希望走更稳、更容易调试的路线，可以先训练 ACT，再决定是否继续做 SmolVLA。

如果继续采集到 `30-50+` 条成功 episode，可以把训练步数提高：

```bash
lerobot-train \
  --dataset.repo_id=puffy/lekiwi-trash \
  --dataset.root=/home/puffy/lerobot/lerobot/datasets/lekiwi-trash \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolvla_lekiwi_trash_clean50 \
  --job_name=smolvla_lekiwi_trash_clean50 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=4 \
  --save_freq=1000 \
  --log_freq=50
```

### ACT 对照训练

如果你后面想做一个更容易排查动作问题的对照实验，再切到 ACT：

为什么先用 ACT：

- 训练和调试更简单。
- LeRobot 支持成熟。
- 适合固定工作区任务。
- 可以学习 LeKiwi 底盘动作：`x.vel`、`y.vel`、`theta.vel`。
- 可以学习机械臂动作：`arm_*.pos`。

ACT 输入：

- RGB 相机图像。
- 机器人状态。
- 数据集统计量，用于归一化。

ACT 输出：

- 一段 robot action chunk，动作空间和 LeKiwi 录制数据一致。

第一版预期能学到：

- 局部靠近垃圾。
- 调整机械臂姿态。
- 抓取或夹取垃圾。
- 短距离移动到垃圾桶。
- 投放垃圾。

第一版预期限制：

- 换到差异很大的场景时泛化弱。
- 超出录制工作区后效果差。
- 对相机位置、灯光变化敏感。
- 它不是完整导航系统。

ACT 冒烟训练命令：

```bash
lerobot-train \
  --dataset.repo_id=puffy/lekiwi-trash \
  --dataset.root=/home/puffy/lerobot/lerobot/datasets/lekiwi-trash \
  --policy.type=act \
  --output_dir=outputs/train/act_lekiwi_trash_smoke \
  --job_name=act_lekiwi_trash_smoke \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=1000 \
  --batch_size=4 \
  --save_freq=500 \
  --log_freq=50
```

如果主控电脑没有 NVIDIA GPU，把：

```bash
--policy.device=cuda
```

改成：

```bash
--policy.device=cpu
```

正式训练命令：

```bash
lerobot-train \
  --dataset.repo_id=puffy/lekiwi-trash \
  --dataset.root=/home/puffy/lerobot/lerobot/datasets/lekiwi-trash \
  --policy.type=act \
  --output_dir=outputs/train/act_lekiwi_trash \
  --job_name=act_lekiwi_trash \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=100000 \
  --batch_size=8 \
  --save_freq=20000 \
  --log_freq=200
```

训练产物位置：

```text
outputs/train/act_lekiwi_trash_smoke
outputs/train/act_lekiwi_trash
```

恢复训练示例：

```bash
lerobot-train \
  --config_path=outputs/train/act_lekiwi_trash/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

评估入口：

- `examples/lekiwi/evaluate.py`

评估前建议设置：

```python
robot_config = LeKiwiClientConfig(remote_ip="192.168.3.16", id="my_lekiwi")
HF_MODEL_ID = "<hf_username>/<trained_model_repo_or_local_path>"
TASK_DESCRIPTION = "Pick up the trash and put it in the trash bin."
```

评估规则：

- 从安全、空旷、可控的工作区开始。
- 保持可以随时断电或急停。
- 先跑短 episode。
- 保存评估视频和失败样例。

## 阶段 4：接入 Uni-NaVid

当 ACT 已经能在受控工作区完成任务后，再把 Uni-NaVid 作为下一阶段升级，用来增强导航和视觉推理能力。

这一阶段目标：

- 从局部模仿学习升级到更强的导航感知行为。
- 使用更丰富的视觉上下文寻找垃圾和垃圾桶。
- 根据 Uni-NaVid 的动作接口，决定是只做导航，还是做端到端控制。

需要完成的工作：

- 把 Uni-NaVid 接成一个 policy backend。
- 定义 Uni-NaVid 从 LeKiwi 读取的观测输入：
  - `front` RGB 图像。
  - 必要时使用 `wrist` RGB 图像。
  - 机器人状态。
  - 任务文本，例如 `Pick up the trash and put it in the trash bin.`
- 定义 Uni-NaVid 的动作输出如何映射到 LeKiwi：
  - 底盘命令：`x.vel`、`y.vel`、`theta.vel`。
  - 机械臂命令：`arm_*.pos`，如果使用端到端 Uni-NaVid。
- 增加 preprocessor 和 postprocessor，用来桥接 Uni-NaVid 与 LeRobot。
- 创建一个类似 `examples/lekiwi/evaluate.py` 的评估脚本。

推荐架构选项：

方案 A：Uni-NaVid 负责导航，ACT 负责操作。

- Uni-NaVid 驱动底盘到垃圾区域或垃圾桶区域。
- ACT 负责近距离抓取和投放。
- 这个方案更安全，也更容易调试。

方案 B：Uni-NaVid 端到端控制。

- Uni-NaVid 同时输出底盘和机械臂动作。
- 这个方案更激进。
- 需要更多数据和更强的安全检查。

推荐路线：

1. 保留 ACT 作为第一版可工作的扔垃圾策略。
2. 先让 Uni-NaVid 只参与导航决策。
3. 近距离抓取和投放继续交给 ACT。
4. 等动作接口和算力预算都合适后，再测试端到端 Uni-NaVid。

Uni-NaVid 接入前需要确认的问题：

- 使用哪个 Uni-NaVid checkpoint？
- checkpoint 输出的是连续机器人动作、waypoint，还是语言/规划决策？
- 它期望的相机格式和分辨率是什么？
- 是否需要深度、里程计、地图等输入？
- 主控电脑能不能以可用频率运行它？
- 推理应该本地运行、GPU 服务器运行，还是异步运行？

## 里程碑

里程碑 1：

- 遥操作稳定运行 10 分钟。
- 两路相机都能看到。
- 底盘和机械臂动作都正常。

里程碑 2：

- 录制 30 条成功扔垃圾 episode。
- 回放几条 episode，没有明显动作错位。

里程碑 3：

- 训练第一版 ACT 模型。
- 在相同设置下跑 10 次评估。
- 至少完成几次完整扔垃圾任务。

里程碑 4：

- 数据扩展到 50-100 条 episode。
- 增加垃圾位置和垃圾类型变化。
- 提升 ACT 成功率。

里程碑 5：

- 开始 Uni-NaVid 接入。
- 决定 Uni-NaVid 只控制导航，还是控制完整机器人。
- 在允许控制真机前，先用录好的 observation 做离线推理测试。

## 安全注意事项

- 早期评估时保持低速。
- 先使用柔软、轻量的垃圾物体。
- 保持工作区空旷。
- 新 policy 不要在人、易碎物体或危险物体附近运行。
- 让模型控制机器人前，一定先检查动作缩放。
- 行为可预测之前，优先使用短 episode 评估。
