# 中文车牌识别系统

基于深度学习的端到端中文车牌识别系统，支持 Web 端和桌面端两种使用方式。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      输入图片                            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  UNet 车牌定位模型 (unet.h5)                             │
│  → 定位并分割车牌区域                                    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  CRNN+CTC 端到端识别模型 (final_model_90.h5)            │
│  → 识别车牌字符，输出 7 位车牌号                         │
└─────────────────────────────────────────────────────────┘
```

## 版本对比

| 特性 | Web 版 | 桌面版 |
|------|--------|--------|
| 入口 | `app.py` | `UI.py` |
| 识别模型 | CRNN+CTC (97.5%) | CNN (7 字符独立输出) |
| 界面 | 浏览器 Web 界面 | Tkinter 桌面应用 |
| 部署方式 | Flask 服务 | 本地运行 |
| 历史记录 | localStorage | 无 |

## 环境要求

- Python 3.8 ~ 3.10
- TensorFlow 2.x
- OpenCV
- Flask

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型下载

> ⚠️ 由于模型文件较大（~22MB），请自行下载并放置到 `Chinese_plate/` 目录

**模型文件**：`final_model_90.h5`（约 22MB）

**下载链接**：
- 夸克网盘：https://pan.quark.cn/s/d12b9fc6703c

放置路径：
```
Chinese_plate/
├── final_model_90.h5   # ← 放在这里
├── unet.h5
├── cnn.h5
├── app.py
└── index.html
```

## 快速开始

### Web 版（推荐）

```bash
cd Chinese_plate
python app.py
```

启动后访问：http://localhost:5000

功能：
- 上传图片识别（支持 JPG/PNG）
- 输入图片 URL 识别
- 显示识别结果和置信度
- 历史记录自动保存在浏览器

### 桌面版

```bash
cd Chinese_plate
python UI.py
```

## 项目文件说明

```
Chinese_plate/
├── app.py              # Flask Web 入口 ← Web 版启动文件
├── index.html          # Web 前端页面
├── UI.py               # Tkinter 桌面入口 ← 桌面版启动文件
├── core.py             # 车牌定位后处理
├── Unet.py             # UNet 定位模型
├── CNN.py              # CNN 识别模型（旧版）
├── final_model_90.h5   # CRNN+CTC 识别模型（97.5% 准确率）
├── unet.h5             # UNet 定位权重
├── cnn.h5              # CNN 识别权重（旧版）
├── crnn_dataset.py     # CCPD 数据预处理
├── crnn_train.py       # CRNN 训练脚本
├── crnn_eval.py        # 模型评估脚本
└── crnn_test_scenarios.py  # 多场景测试
```

## 训练自己的模型（可选）

### 1. 数据预处理

```bash
python crnn_dataset.py --ccpd_root /path/to/CCPD --output_dir crnn_data
```

### 2. 训练 CRNN+CTC

```bash
python crnn_train.py --data_dir crnn_data --epochs 30 --batch_size 64
```

### 3. 评估模型

```bash
python crnn_eval.py --model crnn.h5 --data_dir crnn_data --split val
```

### 4. 多场景测试

```bash
python crnn_test_scenarios.py --model crnn.h5 --data_dir crnn_data --n 500 --plot
```

## 技术细节

- **定位**：UNet 语义分割 → 轮廓检测 → 透视变换矫正
- **识别**：CRNN (CNN + BiLSTM) + CTC Loss
- **字符集**：65 类（31 省份缩写 + 24 字母 + 10 数字）
- **输入尺寸**：32×128 灰度图
- **准确率**：97.5%（验证集）

## 常见问题

### Q1: 启动失败提示找不到模型
- 确认 `final_model_90.h5` 已放置在 `Chinese_plate/` 目录
- 确认从项目根目录启动

### Q2: 识别结果为空
- 检查图片是否包含车牌
- 尝试更清晰的图片
- 确认模型文件完整（22MB 左右）

### Q3: Web 界面无法访问
- 确认端口 5000 未被占用
- 尝试更换端口：`python app.py --port 8080`

## License

MIT License
