#%%writefile /kaggle/working/crnn_dataset.py
# -*- coding: utf-8 -*-
"""
crnn_dataset.py  ——  CCPD 数据集预处理
================================================
CCPD 文件名编码规则（以 CCPD2019 为例）：
  区域-倾斜水平-倾斜垂直-bbox-顶点-亮度-模糊度.jpg
  例：025-95_113-154&383_386&473-386&473_154&383_170&363_402&453-0_0_22_27_27_33_16-37-15.jpg
  字段 4（索引 4）：车牌字符索引，共 7 个，用 _ 分隔
  字段 3（索引 3）：bbox，格式 x1&y1_x2&y2

字符表与 CNN.py 保持完全一致（65 类）。
"""

import os
import sys
import cv2
import numpy as np
import random
import json
from pathlib import Path

# ── 字符表（与 CNN.py 完全一致）────────────────────────────────────────────
CHARACTERS = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
]
CHAR2IDX = {c: i for i, c in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS)   # 65
BLANK_IDX   = NUM_CLASSES       # CTC blank = 65

# CCPD 省份索引 → 汉字（官方顺序）
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "蒙", "辽", "吉", "黑", "苏",
    "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "川",
    "贵", "云", "藏", "琼", "宁", "青", "陕", "甘", "晋", "桂",
    "新",
]
ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
]
ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

# 目标尺寸（CRNN 输入）
IMG_H = 32
IMG_W = 128


# ── 文件名解析 ───────────────────────────────────────────────────────────────
def parse_ccpd_filename(filename: str):
    """
    解析 CCPD 文件名，返回 (plate_str, bbox) 或 (None, None)。
    plate_str: 7 字符车牌字符串，如 "皖A12345"
    bbox: (x1, y1, x2, y2) 整数
    """
    stem = Path(filename).stem
    parts = stem.split('-')
    if len(parts) < 5:
        return None, None
    try:
        # bbox
        bbox_str = parts[2]          # "x1&y1_x2&y2"
        p1, p2   = bbox_str.split('_')
        x1, y1   = map(int, p1.split('&'))
        x2, y2   = map(int, p2.split('&'))

        # 车牌字符索引
        label_str = parts[4]         # "0_0_22_27_27_33_16"
        indices   = list(map(int, label_str.split('_')))
        if len(indices) != 7:
            return None, None

        chars = (
            PROVINCES[indices[0]]
            + ALPHABETS[indices[1]]
            + ''.join(ADS[i] for i in indices[2:])
        )
        return chars, (x1, y1, x2, y2)
    except Exception:
        return None, None


# ── 图像增强 ─────────────────────────────────────────────────────────────────
def augment(img: np.ndarray) -> np.ndarray:
    """随机增强：亮度/对比度、高斯模糊、透视变换、噪声。"""
    # 亮度 & 对比度
    alpha = random.uniform(0.6, 1.4)
    beta  = random.randint(-40, 40)
    img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 高斯模糊（模拟运动模糊 / 失焦）
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 轻微透视变换（模拟倾斜）
    if random.random() < 0.4:
        h, w = img.shape[:2]
        margin = int(min(h, w) * 0.08)
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
        ])
        M   = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 椒盐噪声
    if random.random() < 0.2:
        noise_mask = np.random.randint(0, 100, img.shape[:2])
        img[noise_mask < 2]  = 0
        img[noise_mask > 97] = 255

    return img


# ── 单张图片预处理 ────────────────────────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    BGR → 灰度 → resize(IMG_W, IMG_H) → 归一化到 [0,1]，shape=(IMG_H, IMG_W, 1)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    return gray[:, :, np.newaxis]   # (32, 128, 1)


# ── 数据集构建 ────────────────────────────────────────────────────────────────
def build_dataset(
    ccpd_root: str,
    output_dir: str,
    max_samples: int = 0,
    val_ratio: float = 0.1,
    augment_train: bool = True,
):
    """
    遍历 ccpd_root 下所有 .jpg，解析标签，裁剪车牌，保存为 npz。

    输出：
      output_dir/train.npz  —— {'images': (N,32,128,1), 'labels': list[str]}
      output_dir/val.npz
      output_dir/meta.json  —— 字符表等元信息
    """
    ccpd_root  = Path(ccpd_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(ccpd_root.rglob('*.jpg'))
    if max_samples > 0:
        jpg_files = jpg_files[:max_samples]

    print(f'共找到 {len(jpg_files)} 张图片，开始处理…')

    images, labels = [], []
    skipped = 0

    for idx, fpath in enumerate(jpg_files):
        if idx % 5000 == 0:
            print(f'  [{idx}/{len(jpg_files)}] 已处理 {idx} 张，跳过 {skipped} 张')

        plate_str, bbox = parse_ccpd_filename(fpath.name)
        if plate_str is None:
            skipped += 1
            continue

        # 验证字符是否全在字符表内
        if not all(c in CHAR2IDX for c in plate_str):
            skipped += 1
            continue

        img = cv2.imdecode(np.fromfile(str(fpath), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        # 裁剪车牌区域
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        plate = img[y1:y2, x1:x2]
        if plate.size == 0:
            skipped += 1
            continue

        images.append(preprocess_image(plate))
        labels.append(plate_str)

    print(f'有效样本：{len(images)}，跳过：{skipped}')

    # 打乱并划分
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    n_val   = max(1, int(len(images) * val_ratio))
    n_train = len(images) - n_val

    def save_split(imgs, lbls, name, do_aug):
        imgs = list(imgs)
        lbls = list(lbls)
        if do_aug:
            aug_imgs, aug_lbls = [], []
            for im, lb in zip(imgs, lbls):
                # 原图 + 1 份增强
                aug_imgs.append(im)
                aug_lbls.append(lb)
                aug_raw = (im[:, :, 0] * 255).astype(np.uint8)
                aug_raw = augment(cv2.cvtColor(aug_raw, cv2.COLOR_GRAY2BGR))
                aug_imgs.append(preprocess_image(aug_raw))
                aug_lbls.append(lb)
            imgs, lbls = aug_imgs, aug_lbls

        arr = np.array(imgs, dtype=np.float32)
        path = output_dir / f'{name}.npz'
        np.savez_compressed(str(path), images=arr, labels=np.array(lbls))
        print(f'  已保存 {path}  ({len(imgs)} 条)')

    save_split(images[:n_train], labels[:n_train], 'train', augment_train)
    save_split(images[n_train:], labels[n_train:], 'val',   False)

    # 元信息
    meta = {
        'characters': CHARACTERS,
        'num_classes': NUM_CLASSES,
        'blank_idx':   BLANK_IDX,
        'img_h': IMG_H,
        'img_w': IMG_W,
        'n_train': n_train,
        'n_val':   n_val,
    }
    with open(output_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('元信息已保存至 meta.json')
    return str(output_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CCPD 数据预处理')
    parser.add_argument('--ccpd_root',  required=True,  help='CCPD 数据集根目录')
    parser.add_argument('--output_dir', default='crnn_data', help='输出目录（默认 crnn_data）')
    parser.add_argument('--max_samples', type=int, default=0, help='最大样本数，0=全部')
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--no_augment',  action='store_true', help='不做训练集增强')
    args = parser.parse_args()

    build_dataset(
        ccpd_root    = args.ccpd_root,
        output_dir   = args.output_dir,
        max_samples  = args.max_samples,
        val_ratio    = args.val_ratio,
        augment_train= not args.no_augment,
    )
