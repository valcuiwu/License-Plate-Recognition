%%writefile /kaggle/working/crnn_test_scenarios.py
# -*- coding: utf-8 -*-
"""
crnn_test_scenarios.py  ——  多场景鲁棒性测试
================================================
场景：
  1. 正常（baseline）
  2. 倾斜（透视变换，±15°）
  3. 模糊（高斯模糊 k=7/11）
  4. 低光照（亮度 ×0.3）
  5. 过曝（亮度 ×2.0）
  6. 噪声（高斯噪声 σ=25）
  7. 组合（倾斜 + 模糊 + 低光照）

对每个场景输出：整板准确率 / CER / 逐字符准确率
并生成可视化报告 scenario_report.png（可选）

用法：
  python crnn_test_scenarios.py --model crnn.h5 --data_dir crnn_data --n 500
  python crnn_test_scenarios.py --model crnn.h5 --data_dir crnn_data --n 500 --plot
"""

import os
import argparse
import numpy as np
import cv2

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow import keras

from crnn_dataset import CHARACTERS, NUM_CLASSES, IMG_H, IMG_W
from crnn_eval    import ctc_greedy_decode, edit_distance

CHAR2IDX = {c: i for i, c in enumerate(CHARACTERS)}


# ── 场景变换函数 ──────────────────────────────────────────────────────────────
def apply_tilt(img: np.ndarray, angle_deg: float = 15.0) -> np.ndarray:
    """透视变换模拟水平倾斜。"""
    h, w = img.shape[:2]
    shift = int(h * np.tan(np.radians(angle_deg)))
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([[shift, 0], [w - shift, 0], [0, h], [w, h]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_blur(img: np.ndarray, k: int = 7) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_brightness(img: np.ndarray, factor: float = 0.3) -> np.ndarray:
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def apply_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def apply_combined(img: np.ndarray) -> np.ndarray:
    img = apply_tilt(img, 12)
    img = apply_blur(img, 5)
    img = apply_brightness(img, 0.4)
    return img


# 场景定义：(名称, 变换函数)
SCENARIOS = [
    ('正常 (Baseline)',          lambda x: x),
    ('倾斜 15°',                 lambda x: apply_tilt(x, 15)),
    ('模糊 k=7',                 lambda x: apply_blur(x, 7)),
    ('模糊 k=11',                lambda x: apply_blur(x, 11)),
    ('低光照 ×0.3',              lambda x: apply_brightness(x, 0.3)),
    ('过曝 ×2.0',                lambda x: apply_brightness(x, 2.0)),
    ('高斯噪声 σ=25',            lambda x: apply_noise(x, 25)),
    ('组合 (倾斜+模糊+低光照)',   apply_combined),
]


# ── 预处理（与 crnn_dataset 一致）────────────────────────────────────────────
def preprocess(img_gray: np.ndarray) -> np.ndarray:
    """灰度图 → (IMG_H, IMG_W, 1) float32 [0,1]"""
    resized = cv2.resize(img_gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[:, :, np.newaxis]


# ── 单场景评估 ────────────────────────────────────────────────────────────────
def eval_scenario(model, raw_images_gray: np.ndarray, labels, transform_fn,
                  batch_size: int = 128):
    """
    raw_images_gray: (N, H, W) uint8 灰度图（原始尺寸，未归一化）
    labels: list[str]
    """
    # 应用变换 + 预处理
    processed = np.stack([preprocess(transform_fn(img)) for img in raw_images_gray])

    # 推理
    all_preds = []
    for start in range(0, len(processed), batch_size):
        batch = processed[start:start + batch_size]
        pred  = model.predict(batch, verbose=0)
        for p in pred:
            all_preds.append(ctc_greedy_decode(p))

    # 指标
    plate_correct = 0
    total_cer     = 0.0
    char_correct  = 0
    char_total    = 0

    for gt, pred in zip(labels, all_preds):
        gt, pred = str(gt), str(pred)
        if gt == pred:
            plate_correct += 1
        ed = edit_distance(gt, pred)
        total_cer += ed / max(len(gt), 1)
        for g, p in zip(gt, pred):
            char_total += 1
            if g == p:
                char_correct += 1
        char_total += abs(len(gt) - len(pred))

    n = len(labels)
    return {
        'plate_acc': plate_correct / n,
        'cer':       total_cer / n,
        'char_acc':  char_correct / max(char_total, 1),
    }


# ── 主函数 ────────────────────────────────────────────────────────────────────
def run_scenarios(model_path: str, data_dir: str, n_samples: int,
                  batch_size: int = 128, plot: bool = False):

    # 加载模型
    print(f'加载模型：{model_path}')
    model = keras.models.load_model(model_path, compile=False)

    # 加载验证集
    npz_path = os.path.join(data_dir, 'val.npz')
    print(f'加载数据：{npz_path}')
    data   = np.load(npz_path, allow_pickle=True)
    images = data['images']   # (N, IMG_H, IMG_W, 1) float32 [0,1]
    labels = data['labels']

    # 取前 n_samples
    n_samples = min(n_samples, len(images))
    images = images[:n_samples]
    labels = labels[:n_samples]

    # 还原为 uint8 灰度图（用于施加变换）
    raw_gray = (images[:, :, :, 0] * 255).astype(np.uint8)   # (N, H, W)

    print(f'\n测试样本数：{n_samples}')
    print('─' * 65)
    print(f'{"场景":<22} {"整板准确率":>10} {"CER":>8} {"字符准确率":>10}')
    print('─' * 65)

    results = {}
    for name, fn in SCENARIOS:
        r = eval_scenario(model, raw_gray, labels, fn, batch_size)
        results[name] = r
        print(f'{name:<22} {r["plate_acc"]*100:>9.2f}%  {r["cer"]*100:>7.2f}%  {r["char_acc"]*100:>9.2f}%')

    print('─' * 65)

    # 可视化
    if plot:
        _plot_results(results)

    return results


def _plot_results(results: dict):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 尝试加载中文字体
        zh_fonts = [f.fname for f in fm.fontManager.ttflist
                    if 'SimHei' in f.name or 'Microsoft YaHei' in f.name
                    or 'WenQuanYi' in f.name or 'Noto' in f.name]
        if zh_fonts:
            plt.rcParams['font.family'] = fm.FontProperties(fname=zh_fonts[0]).get_name()
        plt.rcParams['axes.unicode_minus'] = False

        names      = list(results.keys())
        plate_accs = [r['plate_acc'] * 100 for r in results.values()]
        cers       = [r['cer'] * 100       for r in results.values()]
        char_accs  = [r['char_acc'] * 100  for r in results.values()]

        x     = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(x - width, plate_accs, width, label='整板准确率 (%)', color='#4f86c6')
        ax.bar(x,         char_accs,  width, label='字符准确率 (%)', color='#5cb85c')
        ax.bar(x + width, cers,       width, label='CER (%)',        color='#d9534f')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('百分比 (%)')
        ax.set_title('CRNN 多场景鲁棒性测试')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out = 'scenario_report.png'
        plt.savefig(out, dpi=150)
        print(f'\n可视化报告已保存：{out}')
    except ImportError:
        print('matplotlib 未安装，跳过可视化')


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRNN 多场景鲁棒性测试')
    parser.add_argument('--model',      required=True,  help='crnn.h5 路径')
    parser.add_argument('--data_dir',   default='crnn_data')
    parser.add_argument('--n',          type=int, default=500, help='测试样本数')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--plot',       action='store_true', help='生成可视化报告')
    args = parser.parse_args()

    run_scenarios(
        model_path = args.model,
        data_dir   = args.data_dir,
        n_samples  = args.n,
        batch_size = args.batch_size,
        plot       = args.plot,
    )
