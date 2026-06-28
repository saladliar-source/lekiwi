# LeKiwi 调试 QA

这份文档记录当前 LeKiwi 遥操作和采集前最常见的问题。优先按顺序检查：树莓派 host、主控电脑 teleoperate、Leader 串口、相机、标定文件。

## 当前设备信息

- 树莓派 IP：`192.168.3.16`
- 树莓派 SSH：`puffy@192.168.3.16`
- 树莓派密码：单个空格字符
- LeKiwi Follower ID：`R12254718`
- SO101 Leader ID：`R07254718`
- WSL Leader 串口：`/dev/ttyACM0`
- Windows Leader 设备：`USB-Enhanced-SERIAL CH343`

注意：这份信息只适合保存在本地私有文档里，不要提交到公开仓库。

## 正常启动顺序

先在树莓派上启动 host：

```bash
conda activate lerobot
python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R12254718
```

再在主控电脑 WSL 里启动遥操作：

```bash
conda activate lerobot
cd ~/lerobot/lerobot
python examples/lekiwi/teleoperate.py
```

当前主控电脑配置应包含：

```python
robot_config = LeKiwiClientConfig(remote_ip="192.168.3.16", id="my_lekiwi")
teleop_arm_config = SO101LeaderConfig(port="/dev/ttyACM0", id="R07254718")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard", backend="stdin")
```

## Q1：树莓派 host 一直打印 `No command available`

这是正常现象，表示树莓派 host 已经启动，但还没有收到主控电脑发来的控制命令。

看到下面日志时，说明底盘安全 watchdog 生效了：

```text
Command not received for more than 500 milliseconds. Stopping the base.
```

处理方式：

- 保持树莓派 host 继续运行。
- 在主控电脑上启动 `python examples/lekiwi/teleoperate.py`。
- 确认 `teleoperate.py` 里的 `remote_ip` 是 `192.168.3.16`。

## Q2：主控电脑连接不上 LeKiwi Host

典型报错：

```text
Timeout waiting for LeKiwi Host to connect expired.
```

检查：

```bash
ping 192.168.3.16
```

确认树莓派 host 正在运行：

```bash
ssh puffy@192.168.3.16
python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R12254718
```

如果树莓派 IP 变了，在树莓派上查看：

```bash
hostname -I
```

然后更新 `examples/lekiwi/teleoperate.py`：

```python
LeKiwiClientConfig(remote_ip="新的树莓派IP", id="my_lekiwi")
```

## Q3：WSL 里找不到 `/dev/ttyACM0`

典型报错：

```text
Could not connect on port '/dev/ttyACM0'
```

先在 WSL 里检查：

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

如果没有输出，说明 Leader 驱动板没有透传到 WSL。

在 Windows PowerShell 管理员窗口检查：

```powershell
usbipd list
```

找到 `USB-Enhanced-SERIAL CH343` 对应的 `BUSID`，例如 `1-5`，然后执行：

```powershell
usbipd bind --busid 1-5
usbipd attach --wsl --busid 1-5
```

注意：USB 重新插拔后 `BUSID` 可能会变。每次先以 `usbipd list` 的结果为准。

## Q4：`/dev/ttyACM0` 存在，但找不到电机 ID

典型报错：

```text
Missing motor IDs:
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6

Full found motor list:
{}
```

含义：串口已经打开了，但驱动板后面的舵机总线没有任何回应。

优先检查硬件：

- Leader 臂驱动板是否上电。
- 舵机电源是否接好。
- 舵机线是否插在正确接口。
- 舵机线方向是否插反。
- 是否接错成 Follower 臂或其他 USB 串口。
- 是否有其他程序占用 `/dev/ttyACM0`。

检查占用：

```bash
fuser -v /dev/ttyACM0
```

如果怀疑端口不对，重新查找：

```bash
lerobot-find-port
```

## Q5：提示标定文件不存在

典型提示：

```text
Calibration file not found!
```

Leader 标定文件应在主控电脑：

```bash
~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/R07254718.json
```

LeKiwi Follower 标定文件应在树莓派：

```bash
~/.cache/huggingface/lerobot/calibration/robots/lekiwi/R12254718.json
```

如果是第一次使用，需要按提示移动到中位并完成标定。如果已经标定过，确认脚本里的 ID 和文件名完全一致。

## Q6：提示 `Calibration joints mismatch`

含义：标定文件里记录的关节名和当前代码期望的关节名不一致。

如果你确认这是旧标定文件多了无关字段，而不是换了硬件，可以按提示直接按回车继续使用：

```text
Press ENTER to use provided calibration file
```

如果换过电机、换过舵机 ID、换过机械臂结构，则不要硬用旧标定，应该重新标定。

## Q7：提示 `Calibration offsets mismatch`

含义：当前读到的舵机位置和标定文件中的 offset 不完全匹配。

如果没有换硬件，只是重启或姿态略有变化，通常可以按回车继续使用旧标定。

如果换过舵机、重装过臂、重新分配过 ID，输入 `c` 重新标定。

## Q8：WASD 不能控制底盘

主控电脑在 WSL 里使用时，推荐键盘后端使用 `stdin`：

```python
KeyboardTeleopConfig(id="my_laptop_keyboard", backend="stdin")
```

控制键：

- `w`：前进
- `s`：后退
- `a`：左移
- `d`：右移
- `z`：左转
- `x`：右转
- `r`：加速
- `f`：减速

如果机械臂能跟随但底盘不动，优先确认：

- `keyboard_config` 是否设置了 `backend="stdin"`。
- 终端窗口是否获得键盘焦点。
- 树莓派 host 是否还在运行。
- host 日志是否持续收到命令，而不是一直 `No command available`。

## Q9：Rerun 本地窗口报 GPU 错误

WSL 里可能出现：

```text
Adapter does not support drawing to texture format R32Float
```

解决方式：使用 Rerun Web viewer，而不是本地 GPU viewer。

如果代码已经做过 WSL 自动 web viewer 适配，直接运行 `teleoperate.py` 即可。浏览器里打开 Rerun 页面后，如果延迟大，可以降低图像分辨率或 web 图像最大尺寸。

## Q10：只能看到 `front`，看不到 `wrist`

先在树莓派上检查相机：

```bash
v4l2-ctl --list-devices
lerobot-find-cameras opencv
```

当前期望：

- `front`：`/dev/video0`
- `wrist`：`/dev/video2`

树莓派配置文件：

```bash
/home/puffy/lerobot/src/lerobot/robots/lekiwi/config_lekiwi.py
```

确认包含：

```python
"front": OpenCVCameraConfig(index_or_path="/dev/video0", ...)
"wrist": OpenCVCameraConfig(index_or_path="/dev/video2", ...)
```

如果 `/dev/video2` 不存在，通常是腕部相机没插好、USB 没识别，或者设备号变了。

## Q11：`front` 画面倒着

改树莓派上的配置文件：

```bash
/home/puffy/lerobot/src/lerobot/robots/lekiwi/config_lekiwi.py
```

找到 `front`：

```python
"front": OpenCVCameraConfig(
    index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
),
```

如果方向不对，在这几个值之间试：

```python
Cv2Rotation.NO_ROTATION
Cv2Rotation.ROTATE_90
Cv2Rotation.ROTATE_180
Cv2Rotation.ROTATE_270
```

每次修改后都要重启树莓派 host：

```bash
python -m lerobot.robots.lekiwi.lekiwi_host --robot_id=R12254718
```

## Q12：Web 端画面延迟大

优先处理：

- 降低相机分辨率。
- 降低 FPS。
- 关闭暂时不用的相机。
- 保持主控电脑和树莓派在同一个稳定 Wi-Fi 或有线网络。
- 录数据时优先保证控制稳定，不要只追求高清画面。

可以把相机从 `640x480` 降到更低，例如：

```python
width=320
height=240
fps=15
```

## Q13：应该用哪个现成模型

目前没有确认可用于 LeKiwi “上电就自动扔垃圾”的现成模型。

推荐路线：

- 第一阶段用遥操作录自己的数据。
- 第一版训练 ACT，先做固定工作区内的抓取和投放。
- 等 ACT 可用后，再接 Uni-NaVid 做导航增强。

ACT 适合：

- 局部操作。
- 固定或轻微变化的场景。
- 机械臂和底盘的模仿学习。

Uni-NaVid 更适合后续做：

- 导航。
- 视觉语义理解。
- 从更远位置寻找目标区域。

## Q14：开始采数据前的检查清单

- 树莓派 host 正常运行。
- 主控电脑 `teleoperate.py` 能连接。
- `front` 和 `wrist` 两路图像都正常。
- 图像方向正确。
- Leader 臂能控制 Follower 臂。
- 夹爪开合正常。
- WASD 能控制底盘。
- 标定文件 ID 正确。
- 工作区空旷、安全。
- 垃圾物体柔软轻量。
- 可以随时按急停或断电。

## Q15：`record.py` 提示已有数据集怎么办

如果看到：

```text
数据集已存在
输入 'a' 追加录制，输入 'c' 备份旧数据并清空重录，输入 'q' 退出
```

含义：

- `a`：继续往已有 `puffy/lekiwi-trash` 数据集后面追加 episode。
- `c`：把旧数据集目录改名备份，然后从第 0 条重新录。
- `q`：退出，不改动数据。

如果你只是继续采同一个任务，通常选 `a`。如果前面的数据是乱录的测试数据，选 `c` 更干净。

## Q16：设备突然停住，然后又突然动

常见原因有两个。

第一种是 WSL 终端键盘输入不是真正的“按住/松开”事件，而是一串离散字符。这样 WASD 底盘命令会像脉冲一样出现，树莓派 watchdog 可能在空档里短暂停车。

当前代码已经给 `KeyboardTeleopConfig` 增加了：

```python
stdin_hold_s = 0.25
```

它会把 stdin 里最近出现的按键保持约 `0.25` 秒，让底盘控制更连续。如果仍然一顿一顿，可以把这个值调到 `0.35`；如果松开后停得太慢，可以调到 `0.15`。

第二种是 SO101 Leader 臂的舵机总线偶发丢包，报错类似：

```text
There is no status packet!
```

当前代码已经把 Leader 读位置改成最多重试 `3` 次：

```python
self.bus.sync_read("Present_Position", num_retry=3)
```

如果还是频繁出现，优先检查 Leader 臂：

- 驱动板供电是否稳定。
- 舵机线是否松动。
- USB 线是否松动。
- WSL USB 透传是否仍是 `Attached`。
- 是否有其他进程占用 `/dev/ttyACM0`。
