# License-Plate-Recognition 本地运行指南（PyCharm / VS Code）

这个项目是一个中文车牌识别 Demo，默认使用仓库里已经提供的 `unet.h5` 与 `cnn.h5` 做推理。

## 1. 环境要求

- Python 3.8 ~ 3.10（推荐 3.9）
- Windows / macOS / Linux
- 需要图形界面（Tkinter）

> 说明：项目使用 TensorFlow + OpenCV + Pillow。首次安装依赖会比较慢。

## 2. 创建虚拟环境并安装依赖

在项目根目录执行：

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install tensorflow opencv-python pillow numpy
```

### macOS / Linux

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tensorflow opencv-python pillow numpy
```

## 3. 运行方式

请从项目根目录启动：

```bash
python Chinese_plate/UI.py
```

启动后点击：
1. **选择文件**（选一张图片）
2. **识别车牌**

## 4. PyCharm 运行配置

1. 打开项目根目录 `License-Plate-Recognition`
2. `Settings -> Project -> Python Interpreter`，选择 `.venv`
3. 新建 Run Configuration：
   - Script path: `Chinese_plate/UI.py`
   - Working directory: 项目根目录（非常重要）
4. Run

## 5. VS Code 运行配置

1. 用 VS Code 打开项目根目录
2. `Python: Select Interpreter` 选择 `.venv`
3. 在终端执行：

```bash
python Chinese_plate/UI.py
```

如果你想用 F5 启动，可创建 `.vscode/launch.json`：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run UI",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/Chinese_plate/UI.py",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    }
  ]
}
```

## 6. 常见问题

### Q1: `No module named tensorflow`
- 说明依赖未安装到当前解释器。
- 重新确认 IDE 选择的是 `.venv`，然后安装依赖。

### Q2: `can't open/read file` 或找不到 `unet.h5` / `cnn.h5`
- 需要从项目根目录运行，或者使用本仓库中已经修正的 UI（会按脚本目录加载模型文件）。

### Q3: 选择中文路径图片失败
- 项目已使用 `cv2.imdecode(np.fromfile(...))`，通常可处理中文路径。

## 7. 训练说明（可选）

仓库可以训练，但 `Unet.py`、`CNN.py` 内的数据集路径是示例硬编码路径，需要你先改成自己的数据路径。

