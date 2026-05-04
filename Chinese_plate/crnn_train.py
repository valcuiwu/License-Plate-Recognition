# -*- coding: utf-8 -*-
"""
crnn_train.py  ——  CRNN + CTC 端到端训练
================================================
模型结构：
  输入  (32, 128, 1)
  ↓ CNN 特征提取（VGG-like，输出 (1, T, C)）
  ↓ Reshape → (T, C)
  ↓ 双向 LSTM × 2
  ↓ Dense(NUM_CLASSES+1, softmax)  ← +1 为 CTC blank
  CTC Loss（通过自定义 Lambda 层计算）

训练完成后保存：
  crnn.h5          —— 推理用模型（仅 feature+rnn+dense）
  crnn_train.h5    —— 含 CTC loss 的完整训练模型（备用）
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from crnn_dataset import CHARACTERS, NUM_CLASSES, BLANK_IDX, IMG_H, IMG_W

# ── 超参数默认值 ──────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = 'crnn_data'
DEFAULT_SAVE_DIR  = '.'
DEFAULT_EPOCHS    = 30
DEFAULT_BATCH     = 64
DEFAULT_LR        = 1e-3
DEFAULT_MAX_LABEL = 7   # 车牌固定 7 字符


# ── CTC 工具 ──────────────────────────────────────────────────────────────────
def ctc_loss_fn(y_true, y_pred):
    """
    y_pred: (batch, T, num_classes+1)  logits after softmax
    y_true: (batch, MAX_LABEL)         整数标签，-1 表示 padding
    """
    batch_size = tf.shape(y_pred)[0]
    seq_len    = tf.shape(y_pred)[1]

    input_length  = tf.fill([batch_size, 1], seq_len)
    label_length  = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1, keepdims=True)

    # 将 -1 padding 替换为 0（CTC 不会用到，因为 label_length 已限制）
    y_true_clean = tf.maximum(y_true, 0)

    loss = tf.keras.backend.ctc_batch_cost(
        y_true_clean,
        y_pred,
        input_length,
        label_length,
    )
    return tf.reduce_mean(loss)


# ── 模型构建 ──────────────────────────────────────────────────────────────────
def build_crnn(img_h=IMG_H, img_w=IMG_W, num_classes=NUM_CLASSES):
    """
    返回 (inference_model, train_model)
    inference_model: 输入图片 → 输出 softmax 序列，用于 CTC decode
    train_model:     输入 (图片, 标签) → 输出 CTC loss，用于训练
    """
    # ── CNN 主干 ──
    inp = layers.Input(shape=(img_h, img_w, 1), name='image')

    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1')(inp)
    x = layers.MaxPool2D((2, 2), name='pool1')(x)          # (16, 64, 64)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2')(x)
    x = layers.MaxPool2D((2, 2), name='pool2')(x)          # (8, 32, 128)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_1')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_2')(x)
    x = layers.MaxPool2D((2, 1), name='pool3')(x)          # (4, 32, 256)

    # Block 4
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_2')(x)
    x = layers.MaxPool2D((2, 1), name='pool4')(x)          # (2, 32, 512)

    # Block 5
    x = layers.Conv2D(512, 2, padding='valid', activation='relu', name='conv5')(x)
    # → (1, 31, 512)

    # ── Reshape → 序列 ──
    # x shape: (batch, 1, T, 512)  →  (batch, T, 512)
    x = layers.Squeeze(axis=1, name='squeeze')(x) if hasattr(layers, 'Squeeze') else \
        layers.Lambda(lambda t: tf.squeeze(t, axis=1), name='squeeze')(x)

    # ── 双向 LSTM ──
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25), name='blstm1')(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25), name='blstm2')(x)

    # ── 输出层（num_classes + 1 含 blank）──
    out = layers.Dense(num_classes + 1, activation='softmax', name='output')(x)

    inference_model = models.Model(inputs=inp, outputs=out, name='crnn_inference')

    # ── 训练模型（加入 CTC loss）──
    label_inp = layers.Input(shape=(DEFAULT_MAX_LABEL,), dtype='int32', name='label')
    train_model = models.Model(
        inputs  = [inp, label_inp],
        outputs = out,
        name    = 'crnn_train',
    )
    train_model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=DEFAULT_LR),
        loss      = lambda y_true, y_pred: ctc_loss_fn(y_true, y_pred),
    )

    return inference_model, train_model


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_split(data_dir: str, split: str, max_label: int = DEFAULT_MAX_LABEL):
    """
    加载 npz，返回 (images, label_matrix)
    label_matrix: (N, max_label) int32，不足补 -1
    """
    path = os.path.join(data_dir, f'{split}.npz')
    data = np.load(path, allow_pickle=True)
    images = data['images'].astype(np.float32)   # (N, H, W, 1)
    raw_labels = data['labels']                  # array of str

    label_mat = np.full((len(raw_labels), max_label), -1, dtype=np.int32)
    for i, s in enumerate(raw_labels):
        for j, c in enumerate(s[:max_label]):
            label_mat[i, j] = CHAR2IDX_LOCAL.get(c, -1)

    return images, label_mat


# 本地字符表（避免循环导入）
CHAR2IDX_LOCAL = {c: i for i, c in enumerate(CHARACTERS)}


# ── tf.data pipeline ─────────────────────────────────────────────────────────
def make_dataset(images, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(({'image': images, 'label': labels}, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(images), 10000))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── 训练入口 ──────────────────────────────────────────────────────────────────
def train(args):
    print('── 加载数据 ──')
    train_images, train_labels = load_split(args.data_dir, 'train')
    val_images,   val_labels   = load_split(args.data_dir, 'val')
    print(f'  训练集：{len(train_images)}  验证集：{len(val_images)}')

    print('── 构建模型 ──')
    inference_model, train_model = build_crnn()
    inference_model.summary(line_length=90)

    # ── Callbacks ──
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, 'crnn_best.h5')
    cb_list = [
        callbacks.ModelCheckpoint(
            ckpt_path,
            monitor   = 'val_loss',
            save_best_only = True,
            save_weights_only = False,
            verbose   = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        callbacks.EarlyStopping(
            monitor  = 'val_loss',
            patience = 8,
            restore_best_weights = True,
            verbose  = 1,
        ),
        callbacks.CSVLogger(os.path.join(args.save_dir, 'train_log.csv')),
    ]

    print('── 开始训练 ──')
    # train_model 接受 [image, label] 作为输入，label 同时作为 y
    train_model.fit(
        x = [train_images, train_labels],
        y = train_labels,
        validation_data = ([val_images, val_labels], val_labels),
        batch_size = args.batch_size,
        epochs     = args.epochs,
        callbacks  = cb_list,
    )

    # 保存推理模型
    infer_path = os.path.join(args.save_dir, 'crnn.h5')
    inference_model.save(infer_path)
    print(f'推理模型已保存：{infer_path}')

    # 保存完整训练模型（含 CTC）
    train_path = os.path.join(args.save_dir, 'crnn_train.h5')
    train_model.save(train_path)
    print(f'训练模型已保存：{train_path}')


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRNN+CTC 训练')
    parser.add_argument('--data_dir',   default=DEFAULT_DATA_DIR,  help='预处理后的数据目录')
    parser.add_argument('--save_dir',   default=DEFAULT_SAVE_DIR,  help='模型保存目录')
    parser.add_argument('--epochs',     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int,   default=DEFAULT_BATCH)
    parser.add_argument('--lr',         type=float, default=DEFAULT_LR)
    args = parser.parse_args()
    DEFAULT_LR = args.lr   # 传递给 build_crnn
    train(args)
