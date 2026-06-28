# LeKiwi Trash Workflow

这个仓库当前主要聚焦在一条已经跑通的 LeKiwi `trash` 任务链路：

- 任务：`Pick up the trash and put it in the trash bin.`
- 机器人：LeKiwi
- 本地数据集：`/home/puffy/lerobot/lerobot/datasets/lekiwi-trash`
- 当前清洗后数据集：`19` 个 episodes
- 当前主训练模型：`smolvla`

## 当前目录约定

- 录制脚本：`examples/lekiwi/record.py`
- 真机评估脚本：`examples/lekiwi/evaluate.py`
- 离线仿真评估脚本：`examples/lekiwi/simulate_policy_offline.py`
- LeKiwi 文档：`docs/source/lekiwi.mdx`
- 当前执行方案：`docs/source/lekiwi_trash_plan.md`

## 数据集

当前使用的本地数据集路径：

```bash
/home/puffy/lerobot/lerobot/datasets/lekiwi-trash
```

这个数据集已经删除了质量较差的 `ep14` 和 `ep18`，现在是 `19` 条数据。

如果需要检查元数据：

```bash
cat /home/puffy/lerobot/lerobot/datasets/lekiwi-trash/meta/info.json
```

## 录制

先启动 LeKiwi host：

```bash
python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R12254718
```

然后录制：

```bash
python examples/lekiwi/record.py
```

录制脚本当前是按本地目录保存，不走默认 HF 缓存。

## 训练

当前推荐训练命令：

```bash
lerobot-train \
  --dataset.repo_id=puffy/lekiwi-trash \
  --dataset.root=/home/puffy/lerobot/lerobot/datasets/lekiwi-trash \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolvla_lekiwi_trash_clean19 \
  --job_name=smolvla_lekiwi_trash_clean19 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=4 \
  --save_freq=1000 \
  --log_freq=50
```

训练输出目录：

```bash
outputs/train/smolvla_lekiwi_trash_clean19
```

## 真机评估

训练完成后，直接用：

```bash
python examples/lekiwi/evaluate.py \
  --policy-path outputs/train/smolvla_lekiwi_trash_clean19/checkpoints/last/pretrained_model
```

## 离线评估

为了避免写系统缓存，先准备本地缓存目录：

```bash
mkdir -p /home/puffy/lerobot/lerobot/.cache/hf_datasets
mkdir -p /home/puffy/lerobot/lerobot/.cache/matplotlib
```

离线评估命令：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_CACHE=/home/puffy/lerobot/lerobot/.cache/hf_datasets \
MPLCONFIGDIR=/home/puffy/lerobot/lerobot/.cache/matplotlib \
python examples/lekiwi/simulate_policy_offline.py \
  --policy-path outputs/train/smolvla_lekiwi_trash_clean19/checkpoints/last/pretrained_model \
  --repo-id puffy/lekiwi-trash \
  --root /home/puffy/lerobot/lerobot/datasets/lekiwi-trash \
  --all-episodes \
  --detailed-summary \
  --prediction-mode aligned \
  --summary-path outputs/lekiwi_offline_sim/summary_clean19.json \
  --plot-dir outputs/lekiwi_offline_sim/plots_clean19 \
  --device cuda
```

评估输出：

- 汇总：`outputs/lekiwi_offline_sim/summary_clean19.json`
- 曲线图：`outputs/lekiwi_offline_sim/plots_clean19`

## 当前已知结论

- `gripper` 维度存在明显滞后，`action` 和 `state` 不是同一时刻量。
- 数据里原先的 `ep14`、`ep18` 质量较差，已经删除。
- 离线评估脚本已经改成支持 `aligned` 模式，避免把动作 chunk 错位比较。

## Git 提交注意

当前不希望把大文件推到远端，已经在 `.gitignore` 里排除了：

- `datasets/`
- `local_datasets/`
- `outputs/`
- `.cache/`
