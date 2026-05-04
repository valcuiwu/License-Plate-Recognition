# -*- coding: utf-8 -*-
"""
Flask Web 入口 —— 车牌识别 Web 应用（CRNN 97.5% 版本）
使用 final_model_90.h5 端到端识别
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import base64
import tempfile
import numpy as np
import cv2
import requests as http_requests
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from core import locate_and_correct
from Unet import unet_predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 字符表（65 类）
CHARACTERS = ["京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
NUM_CLASSES = len(CHARACTERS)  # 65
BLANK_IDX = NUM_CLASSES  # 65

# 加载模型
print('正在加载模型...')
unet = tf.keras.models.load_model(os.path.join(BASE_DIR, 'unet.h5'))
crnn_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'final_model_90.h5'), compile=False)
print('模型加载完毕！')


def ctc_decode(pred):
    """CTC 贪婪解码，返回车牌字符串"""
    pred = np.squeeze(pred, 0)  # (T, 66)
    indices = np.argmax(pred, axis=-1)
    result = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != BLANK_IDX:
            if idx < NUM_CLASSES:
                result.append(CHARACTERS[idx])
        prev = idx
    return ''.join(result)


def get_confidence(pred):
    """计算预测置信度（取每帧最大概率的平均）"""
    pred = np.squeeze(pred, 0)
    max_probs = np.max(pred, axis=-1)
    return float(np.mean(max_probs))


def recognize_plate(plate_img):
    """识别车牌图片，返回 (车牌号, 置信度)"""
    # BGR 转灰度
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # 缩放到 128x32（CRNN 输入尺寸）
    resized = cv2.resize(gray, (128, 32))
    # 归一化到 [0,1] 并增加 batch 和通道维度
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = input_tensor[np.newaxis, ..., np.newaxis]  # (1,32,128,1)
    # 推理
    pred = crnn_model.predict(input_tensor, verbose=0)  # (1,T,66)
    # 解码
    plate_number = ctc_decode(pred)
    confidence = get_confidence(pred)
    return plate_number, confidence


def ndarray_to_b64(img):
    """ndarray → base64 PNG data URI"""
    success, buf = cv2.imencode('.png', img)
    return 'data:image/png;base64,' + base64.b64encode(buf).decode('utf-8') if success else ''


def run_recognition(img_bytes):
    """核���识别流程"""
    # 解码图片
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_src is None:
        return {'success': False, 'error': '无法解析图片，请确认格式为 JPG 或 PNG'}

    h, w = img_src.shape[:2]

    # 判断是否整张图片就是车牌（无需 UNet 定位）
    if h * w <= 240 * 80 and 2 <= w / h <= 5:
        lic = cv2.resize(img_src, (240, 80))[:, :, :3]
        img_src_copy = img_src.copy()
        Lic_img = [lic]
    else:
        # 需要 UNet 定位
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            img_src_resized, img_mask = unet_predict(unet, tmp_path)
            img_src_copy, Lic_img = locate_and_correct(img_src_resized, img_mask)
        finally:
            os.unlink(tmp_path)

    if not Lic_img or not isinstance(img_src_copy, np.ndarray):
        return {'success': False, 'error': '未检测到车牌区域，请尝试更清晰的图片'}

    # 使用 CRNN 识别
    plates = []
    for lic_img in Lic_img:
        plate_number, confidence = recognize_plate(lic_img)
        if not plate_number:
            continue
        plates.append({
            'plate_image': ndarray_to_b64(lic_img),
            'plate_number': plate_number,
            'confidence': round(confidence, 4),
        })

    if not plates:
        return {'success': False, 'error': '未能识别出车牌，请尝试更清晰的图片'}

    return {
        'success': True,
        'annotated_image': ndarray_to_b64(img_src_copy),
        'plates': plates,
    }


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """统一识别接口：支持文件上传和 URL"""
    img_bytes = None

    # 文件上传
    if 'file' in request.files:
        f = request.files['file']
        if f.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png'):
            return jsonify({'success': False, 'error': '仅支持 JPG / PNG 格式'}), 400
        img_bytes = f.read()

    # URL 输入
    elif request.is_json and request.json.get('url'):
        url = request.json['url'].strip()
        try:
            resp = http_requests.get(url, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content
        except Exception as e:
            return jsonify({'success': False, 'error': f'无法获取图片：{e}'}), 400

    else:
        return jsonify({'success': False, 'error': '请上传图片文件或提供图片 URL'}), 400

    result = run_recognition(img_bytes)
    status = 200 if result['success'] else 422
    return jsonify(result), status


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)