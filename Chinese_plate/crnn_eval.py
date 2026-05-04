#%%writefile /kaggle/working/crnn_eval.py
# -*- coding: utf-8 -*-
"""
crnn_eval.py  ——  CRNN 模型评估
================================================
指标：
  1. 整板准确率（Plate Accuracy）：预测字符串 == 真实字符串
  2. 字符错误率（CER）：编辑距离 / 真实长度，越低越好
  3. 逐字符准确率（Char Accuracy）：正确字符数 / 总字符数

用法：
  python crnn_eval.py --model crnn.h5 --data_dir crnn_data --split val
  python crnn_eval.py --model crnn.h5 --data_dir crnn_data --split val --save_errors errors.csv
"""

import os
import argparse
import json
import csv
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow import keras

from crnn_dataset import CHARACTERS, NUM_CLASSES, BLANK_IDX, IMG_H, IMG_W

CHAR2IDX = {c: i for i, c in enumerate(CHARACTERS)}


# ── CTC Greedy Decode ─────────────────────────────────────────────────────────
def ctc_greedy_decode(pred: np.ndarray) -> str:
    """
    pred: (T, num_classes+1)  softmax 输出
    返回解码后的字符串（去除 blank 和重复）
    """
    indices = np.argmax(pred, axis=-1)   # (T,)
    # 去重复
    deduped = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1]:
            deduped.append(indices[i])
    # 去 blank
    chars = [CHARACTERS[i] for i in deduped if i < NUM_CLASSES]
    return ''.join(chars)


# ── 编辑距离（Levenshtein）────────────────────────────────────────────────────
def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


# ── 评估主函数 ────────────────────────────────────────────────────────────────
def evaluate(model_path: str, data_dir: str, split: str,
             batch_size: int = 128, save_errors: str = None):

    # 加载模型
    print(f'加载模型：{model_path}')
    model = keras.models.load_model(model_path, compile=False)

    # 加载数据
    npz_path = os.path.join(data_dir, f'{split}.npz')
    print(f'加载数据：{npz_path}')
    data   = np.load(npz_path, allow_pickle=True)
    images = data['images'].astype(np.float32)
    labels = data['labels']   # array of str

    n = len(images)
    print(f'样本数：{n}')

    # 批量推理
    all_preds = []
    for start in range(0, n, batch_size):
        batch = images[start:start + batch_size]
        pred  = model.predict(batch, verbose=0)   # (B, T, C+1)
        for p in pred:
            all_preds.append(ctc_greedy_decode(p))
        if (start // batch_size) % 10 == 0:
            print(f'  推理进度：{min(start + batch_size, n)}/{n}')

    # 计算指标
    plate_correct = 0
    total_cer     = 0.0
    char_correct  = 0
    char_total    = 0
    errors        = []   # (真实, 预测)

    for gt, pred in zip(labels, all_preds):
        gt   = str(gt)
        pred = str(pred)

        if gt == pred:
            plate_correct += 1
        else:
            errors.append((gt, pred))

        ed = edit_distance(gt, pred)
        total_cer += ed / max(len(gt), 1)

        # 逐字符（对齐到较短长度）
        for g, p in zip(gt, pred):
            char_total += 1
            if g == p:
                char_correct += 1
        char_total += abs(len(gt) - len(pred))   # 多余/缺失字符算错

    plate_acc  = plate_correct / n
    cer        = total_cer / n
    char_acc   = char_correct / max(char_total, 1)

    print('\n' + '═' * 50)
    print(f'  评估集：{split}  |  样本数：{n}')
    print(f'  整板准确率（Plate Acc）：{plate_acc * 100:.2f}%')
    print(f'  字符错误率（CER）      ：{cer * 100:.2f}%')
    print(f'  逐字符准确率（Char Acc）：{char_acc * 100:.2f}%')
    print('═' * 50)

    # 保存错误样本
    if save_errors:
        with open(save_errors, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['真实标签', '预测结果', '编辑距离'])
            for gt, pred in errors:
                writer.writerow([gt, pred, edit_distance(gt, pred)])
        print(f'错误样本已保存：{save_errors}  ({len(errors)} 条)')

    return {
        'plate_acc': plate_acc,
        'cer':       cer,
        'char_acc':  char_acc,
        'n_samples': n,
        'n_errors':  len(errors),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRNN 模型评估')
    parser.add_argument('--model',       required=True,  help='crnn.h5 路径')
    parser.add_argument('--data_dir',    default='crnn_data')
    parser.add_argument('--split',       default='val',  choices=['train', 'val'])
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--save_errors', default=None,   help='错误样本输出 CSV 路径')
    args = parser.parse_args()

    evaluate(
        model_path  = args.model,
        data_dir    = args.data_dir,
        split       = args.split,
        batch_size  = args.batch_size,
        save_errors = args.save_errors,
    )
