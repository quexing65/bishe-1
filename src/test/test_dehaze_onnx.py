import os
import random
import numpy as np
import cv2
import onnxruntime as ort
from glob import glob

DATA_DIR = r"C:\Users\20336\Desktop\Data\leftImg8bit_foggy\train"
ONNX_PATH = r"e:\Workspace\bishe-1\weights\onnx\nafnet_dehaze.onnx"
OUTPUT_DIR = r"e:\Workspace\bishe-1\output\dehaze_test"
NUM_SAMPLES = 6
PATCH_SIZE = 256


def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return img, h, w


def preprocess(img):
    img_rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    img_nchw = img_rgb.transpose(2, 0, 1)[np.newaxis, ...]
    return img_nchw


def postprocess(output):
    out = output[0]
    out = np.clip(out, 0, 1)
    out = (out.transpose(1, 2, 0) * 255).astype(np.uint8)
    return out[:, :, ::-1]


session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"模型加载成功: {ONNX_PATH}")

all_haze = []
for city in os.listdir(DATA_DIR):
    city_path = os.path.join(DATA_DIR, city)
    if not os.path.isdir(city_path):
        continue
    for f in os.listdir(city_path):
        if "beta_0.02" in f:
            all_haze.append(os.path.join(city_path, f))

samples = random.sample(all_haze, min(NUM_SAMPLES, len(all_haze)))
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, haze_path in enumerate(samples):
    haze_name = os.path.basename(haze_path)
    gt_path = haze_path.replace("beta_0.02", "beta_0.005")
    gt_name = os.path.basename(gt_path)

    haze_img = cv2.imread(haze_path)
    gt_img = cv2.imread(gt_path)

    if haze_img is None or gt_img is None:
        print(f"[SKIP] 无法读取: {haze_name}")
        continue

    padded, orig_h, orig_w = pad_to_multiple(haze_img)
    blob = preprocess(padded)
    output = session.run([output_name], {input_name: blob})[0]
    result = postprocess(output)
    result = result[:orig_h, :orig_w]

    h, w = haze_img.shape[:2]
    scale = min(640 / w, 480 / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)

    haze_r = cv2.resize(haze_img, (new_w, new_h))
    gt_r = cv2.resize(gt_img, (new_w, new_h))
    result_r = cv2.resize(result, (new_w, new_h))

    gap = np.ones((new_h, 6, 3), dtype=np.uint8) * 255
    concat = np.hstack([haze_r, gap, result_r, gap, gt_r])

    label_h = 40
    label_bar = np.ones((label_h, concat.shape[1], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(label_bar, "Input (beta_0.02)", (10, 30), font, 0.6, (0, 0, 200), 2)
    cv2.putText(label_bar, "Model Output", (new_w + 16, 30), font, 0.6, (200, 100, 0), 2)
    cv2.putText(label_bar, "GT (beta_0.005)", (new_w * 2 + 32, 30), font, 0.6, (0, 150, 0), 2)

    final = np.vstack([label_bar, concat])
    out_path = os.path.join(OUTPUT_DIR, f"result_{i+1:02d}.jpg")
    cv2.imwrite(out_path, final)

    diff = np.abs(result.astype(np.float32) - gt_img.astype(np.float32))
    mse = np.mean(diff ** 2)
    psnr_val = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

    print(f"[{i+1}] {haze_name} | PSNR: {psnr_val:.2f} dB")

print(f"\n结果保存在: {OUTPUT_DIR}")
