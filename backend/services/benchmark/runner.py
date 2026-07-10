import os
import sys
import time
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROJECT_DIR
from .metrics import (
    load_yolo_labels, yolo_to_xyxy,
    calculate_map, calculate_map_range,
    calculate_map_per_class, calculate_precision_recall,
    calculate_psnr_ssim,
)


WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")

ENHANCE_FUNC_MAP = {
    "dehaze": "_enhance_dehaze",
    "derain": "_enhance_derain",
    "nafnet_dehaze": "_enhance_nafnet_dehaze",
    "nafnet_derain": "_enhance_nafnet_derain",
    "ffa_dehaze": "_enhance_ffa_dehaze",
    "ffa_dehaze_ots": "_enhance_ffa_dehaze_ots",
    "legacy_dehaze": "_enhance_legacy_dehaze",
    "legacy_derain": "_enhance_legacy_derain",
    "low_light": "_enhance_low_light",
    "mbllen_onnx": "_enhance_mbllen_onnx",
    "auto": "_enhance_auto",
}


class BenchmarkRunner:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self._enhancer = None

    def _load_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]

        from ultralytics import YOLO
        from services.vehicle_detector import VehicleDetector
        model_path = VehicleDetector.get_model_path(
            model_name.replace(".onnx", "").replace(".pt", "")
        )
        print(f"  加载模型: {model_path}")
        model = YOLO(model_path)
        self.models[model_name] = model
        return model

    def _get_enhancer(self):
        if self._enhancer is None:
            from services.image_enhancer import ImageEnhancer
            self._enhancer = ImageEnhancer()
        return self._enhancer

    def _enhance_nafnet_dehaze(self, img):
        enhancer = self._get_enhancer()
        return enhancer.nafnet_dehaze_frame(img)

    def _enhance_nafnet_derain(self, img):
        enhancer = self._get_enhancer()
        return enhancer.nafnet_derain_frame(img)

    def _enhance_ffa_dehaze(self, img):
        enhancer = self._get_enhancer()
        return enhancer.ffa_dehaze_frame(img)

    def _enhance_ffa_dehaze_ots(self, img):
        enhancer = self._get_enhancer()
        return enhancer.ffa_dehaze_ots_frame(img)

    def _enhance_dehaze(self, img):
        return self._enhance_nafnet_dehaze(img)

    def _enhance_derain(self, img):
        return self._enhance_nafnet_derain(img)

    def _enhance_legacy_dehaze(self, img):
        enhancer = self._get_enhancer()
        h, w = img.shape[:2]
        tmp_in = os.path.join(PROJECT_DIR, "output", "_bench_tmp_in.jpg")
        tmp_out = os.path.join(PROJECT_DIR, "output", "_bench_tmp_out.jpg")
        os.makedirs(os.path.dirname(tmp_in), exist_ok=True)
        cv2.imwrite(tmp_in, img)
        result = enhancer.dehaze(tmp_in, tmp_out)
        if result and os.path.exists(tmp_out):
            enhanced = cv2.imread(tmp_out)
            if enhanced is not None:
                if enhanced.shape[:2] != (h, w):
                    enhanced = cv2.resize(enhanced, (w, h))
                return enhanced
        return img

    def _enhance_legacy_derain(self, img):
        enhancer = self._get_enhancer()
        h, w = img.shape[:2]
        tmp_in = os.path.join(PROJECT_DIR, "output", "_bench_tmp_in.jpg")
        tmp_out = os.path.join(PROJECT_DIR, "output", "_bench_tmp_out.jpg")
        os.makedirs(os.path.dirname(tmp_in), exist_ok=True)
        cv2.imwrite(tmp_in, img)
        result = enhancer.derain(tmp_in, tmp_out)
        if result and os.path.exists(tmp_out):
            enhanced = cv2.imread(tmp_out)
            if enhanced is not None:
                if enhanced.shape[:2] != (h, w):
                    enhanced = cv2.resize(enhanced, (w, h))
                return enhanced
        return img

    def _enhance_low_light(self, img):
        enhancer = self._get_enhancer()
        h, w = img.shape[:2]
        tmp_in = os.path.join(PROJECT_DIR, "output", "_bench_tmp_in.jpg")
        tmp_out = os.path.join(PROJECT_DIR, "output", "_bench_tmp_out.jpg")
        os.makedirs(os.path.dirname(tmp_in), exist_ok=True)
        cv2.imwrite(tmp_in, img)
        result = enhancer.enhance_low_light(tmp_in, tmp_out)
        if result and os.path.exists(tmp_out):
            enhanced = cv2.imread(tmp_out)
            if enhanced is not None:
                if enhanced.shape[:2] != (h, w):
                    enhanced = cv2.resize(enhanced, (w, h))
                return enhanced
        return img

    def _enhance_mbllen_onnx(self, img):
        enhancer = self._get_enhancer()
        if enhancer.onnx_mbllen is None:
            print("    MBLLEN ONNX 不可用，回退到 low_light")
            return self._enhance_low_light(img)
        h, w = img.shape[:2]
        tmp_in = os.path.join(PROJECT_DIR, "output", "_bench_tmp_in.jpg")
        tmp_out = os.path.join(PROJECT_DIR, "output", "_bench_tmp_out.jpg")
        os.makedirs(os.path.dirname(tmp_in), exist_ok=True)
        cv2.imwrite(tmp_in, img)
        result = enhancer.mbllen_onnx_enhance(tmp_in, tmp_out)
        out_path = None
        if isinstance(result, dict):
            out_path = result.get("output_path")
        elif isinstance(result, str):
            out_path = result
        if out_path and os.path.exists(out_path):
            enhanced = cv2.imread(out_path)
            if enhanced is not None:
                if enhanced.shape[:2] != (h, w):
                    enhanced = cv2.resize(enhanced, (w, h))
                return enhanced
        return img

    def _enhance_auto(self, img):
        from services.auto_enhancer import AutoEnhancer
        auto = AutoEnhancer()
        condition = auto.detect_condition_from_frame(img)
        if condition == "low_light":
            return self._enhance_low_light(img)
        elif condition == "dehaze":
            return self._enhance_nafnet_dehaze(img)
        elif condition == "derain":
            return self._enhance_nafnet_derain(img)
        return img

    def _run_detection(self, img, model, conf_threshold, vehicle_classes):
        results = model(img, conf=conf_threshold, verbose=False)
        boxes = []
        count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in vehicle_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                boxes.append([x1, y1, x2, y2, conf, cls_id])
                count += 1
        return count, boxes

    def _get_brightness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return round(float(gray.mean()), 2)

    def run_single(self, dataset_name, image_path, label_path, enhancement, model_name):
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]

        gt_yolo = load_yolo_labels(label_path)
        gt_boxes = [yolo_to_xyxy(b, w, h) for b in gt_yolo]

        brightness_before = self._get_brightness(img)

        enhanced_img = img
        enhancement_time = 0.0
        if enhancement and enhancement != "none":
            func_name = ENHANCE_FUNC_MAP.get(enhancement)
            if func_name:
                t0 = time.time()
                enhanced_img = getattr(self, func_name)(img.copy())
                enhancement_time = (time.time() - t0) * 1000

        brightness_after = self._get_brightness(enhanced_img)
        brightness_gain = round(brightness_after / max(brightness_before, 1e-6), 2)

        psnr, ssim = calculate_psnr_ssim(img, enhanced_img)

        model = self._load_model(model_name)
        if model is None:
            return None

        conf = self.config.get("conf_threshold", 0.35)
        vc = self.config.get("vehicle_classes", {1, 2, 3, 5, 7})

        t0 = time.time()
        vehicle_count, pred_boxes = self._run_detection(enhanced_img, model, conf, vc)
        detect_time = (time.time() - t0) * 1000

        avg_conf = round(np.mean([b[4] for b in pred_boxes]), 4) if pred_boxes else 0.0

        map50 = calculate_map(gt_boxes, pred_boxes, 0.5)
        map50_95 = calculate_map_range(gt_boxes, pred_boxes)
        precision, recall = calculate_precision_recall(gt_boxes, pred_boxes, 0.5)

        class_map = calculate_map_per_class(gt_boxes, pred_boxes, vc, 0.5)

        return {
            "dataset": dataset_name,
            "filename": os.path.basename(image_path),
            "enhancement": enhancement or "none",
            "model": model_name,
            "brightness_before": brightness_before,
            "brightness_after": brightness_after,
            "brightness_gain": brightness_gain,
            "enhancement_time_ms": round(enhancement_time, 2),
            "detect_time_ms": round(detect_time, 2),
            "total_time_ms": round(enhancement_time + detect_time, 2),
            "vehicle_count": vehicle_count,
            "avg_confidence": avg_conf,
            "gt_count": len(gt_boxes),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "psnr": psnr,
            "ssim": ssim,
            "class_mAP50": json.dumps({str(k): round(v, 4) for k, v in class_map.items()}),
        }

    def run_all(self, progress_callback=None):
        all_results = []
        datasets = self.config["datasets"]
        models = self.config["models"]

        total_tasks = 0
        for ds_cfg in datasets.values():
            imgs = ds_cfg.get_images()
            enh_list = ds_cfg.enhancements + ["none"]
            total_tasks += len(imgs) * len(enh_list) * len(models)

        done = 0
        for ds_name, ds_cfg in datasets.items():
            images = ds_cfg.get_images()
            if not images:
                print(f"  [{ds_name}] 无图片，跳过")
                continue

            print(f"\n  [{ds_name}] {len(images)} 张图片")

            enh_list = ds_cfg.enhancements + ["none"]

            for model_name in models:
                self._load_model(model_name)

                for img_name in images:
                    img_path = os.path.join(ds_cfg.image_dir, img_name)
                    lbl_path = ds_cfg.get_label_path(img_name)

                    for enh in enh_list:
                        result = self.run_single(
                            ds_name, img_path, lbl_path, enh, model_name
                        )
                        if result:
                            all_results.append(result)
                        done += 1
                        if progress_callback:
                            progress_callback(done, total_tasks, ds_name, img_name, enh)
                        elif done % 20 == 0:
                            print(f"    进度: {done}/{total_tasks}")

        return all_results
