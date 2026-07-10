from flask import Blueprint, request, jsonify
import os
import gc
import cv2
import uuid
import numpy as np
import base64
import time
import logging
from datetime import datetime

from utils.database import get_db_connection
from config import PROJECT_DIR

logger = logging.getLogger(__name__)

camera_bp = Blueprint("camera", __name__)

WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "detected")
MAX_CAMERA_IMAGES = 50

_model_cache = {}

# 跟踪器状态
_tracked_objects = {}
_tracker_last_update = 0


def cleanup_old_camera_images():
    """清理旧的摄像头检测结果，只保留最新的N张"""
    try:
        if not os.path.exists(OUTPUT_DIR):
            return
        files = [
            f
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith("detected_") and f.endswith(".jpg")
        ]
        if len(files) <= MAX_CAMERA_IMAGES:
            return
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
        for old_file in files[: len(files) - MAX_CAMERA_IMAGES]:
            os.remove(os.path.join(OUTPUT_DIR, old_file))
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def load_yolo_model(model_name):
    if model_name not in _model_cache:
        from services.vehicle_detector import VehicleDetector
        try:
            model, actual_name = VehicleDetector.load_model_safe(model_name)
            _model_cache[model_name] = model
            logger.info(f"摄像头检测加载模型: {model_name} (实际: {actual_name})")
        except Exception as e:
            logger.error(f"摄像头模型 {model_name} 加载失败: {e}")
            raise
    return _model_cache[model_name]


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def get_enhance_func(enh_type, strength=1.5):
    from services.auto_enhancer import AutoEnhancer
    from services.image_enhancer import ImageEnhancer
    auto_enhancer = AutoEnhancer()
    enhancer = ImageEnhancer()

    def enhance(img):
        if enh_type == "auto":
            return auto_enhancer.enhance_frame(img, auto_enhancer.detect_condition_from_frame(img))
        elif enh_type == "dehaze" or enh_type == "nafnet_dehaze":
            return enhancer.nafnet_dehaze_frame(img)
        elif enh_type == "ffa_dehaze":
            return enhancer.ffa_dehaze_frame(img, strength)
        elif enh_type == "ffa_dehaze_ots":
            return enhancer.ffa_dehaze_ots_frame(img, strength)
        elif enh_type == "low_light":
            return auto_enhancer._enhance_low_light_frame(img)
        elif enh_type == "mbllen":
            m = enhancer.mbllen
            return m.enhance_frame(img) if m else img
        elif enh_type == "mbllen_onnx":
            m = enhancer.onnx_mbllen
            return m.enhance_frame(img) if m else img
        elif enh_type == "zero_dce":
            return enhancer.dce.enhance_frame(img)
        elif enh_type == "derain" or enh_type == "nafnet_derain":
            return enhancer.nafnet_derain_frame(img)
        elif enh_type == "legacy_dehaze":
            return ImageEnhancer._clahe_dehaze_frame(img)
        elif enh_type == "legacy_derain":
            return ImageEnhancer._morph_derain_frame(img)
        return img
    return enhance


@camera_bp.route("/detect", methods=["POST"])
def detect_from_camera():
    global _tracked_objects, _tracker_last_update

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data"}), 400

        frame_b64 = data.get("frame")
        model_type = data.get("model_type", "yolov8n")
        enh_type = data.get("enhancement_type", "auto")
        strength = float(data.get("strength", 1.5))

        if not frame_b64:
            return jsonify({"success": False, "error": "No frame"}), 400

        # 解码
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Decode error"}), 400

        # 增强
        if enh_type == "auto":
            from services.auto_enhancer import AutoEnhancer

            ae = AutoEnhancer()
            condition = ae.detect_condition_from_frame(frame)
            frame = ae.enhance_frame(frame, condition)
        elif enh_type != "normal":
            enhance_fn = get_enhance_func(enh_type, strength)
            frame = enhance_fn(frame)

        # 生成文件名（不保存原始帧到磁盘，直接处理）
        filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 检测
        gc.collect()
        try:
            model = load_yolo_model(model_type)
        except Exception as e:
            return jsonify({"success": False, "error": f"模型加载失败（可能内存不足）: {str(e)}"}), 500
        results = model(frame, conf=0.35, imgsz=640, verbose=False)

        # 处理结果
        current_time = time.time()
        detected = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in [1, 2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vtype = {
                        2: "car",
                        7: "truck",
                        5: "bus",
                        3: "motorcycle",
                        1: "bicycle",
                    }.get(cls_id, "unknown")
                    detected.append({"box": (x1, y1, x2, y2), "type": vtype})

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        vtype,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # 跟踪 - 使用IOU匹配避免重复计数
        new_tracked = {}
        for obj in detected:
            matched = False
            for tid, tinfo in _tracked_objects.items():
                iou = calculate_iou(obj["box"], tinfo["box"])
                if iou > 0.3:
                    new_tracked[tid] = {
                        "box": obj["box"],
                        "type": obj["type"],
                        "time": current_time,
                    }
                    matched = True
                    break
            if not matched:
                new_id = str(uuid.uuid4())[:8]
                new_tracked[new_id] = {
                    "box": obj["box"],
                    "type": obj["type"],
                    "time": current_time,
                }

        # 移除超时(2秒)的跟踪
        if current_time - _tracker_last_update > 3:
            _tracked_objects = new_tracked
            _tracker_last_update = current_time
        else:
            for k, v in new_tracked.items():
                _tracked_objects[k] = v

        # 使用当前跟踪器的结果统计
        counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
        for tinfo in _tracked_objects.values():
            if tinfo["type"] in counts:
                counts[tinfo["type"]] += 1

        total = sum(counts.values())

        cv2.imwrite(output_path, frame)

        cleanup_old_camera_images()

        detected_blob = None
        try:
            with open(output_path, "rb") as f:
                detected_blob = f.read()
        except Exception as e:
            logger.error(f"读取检测图片失败: {e}")

        created_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO detection_records 
                        (file_type, file_name, file_path, total_vehicles, car_count, 
                         truck_count, bus_count, motorcycle_count, bicycle_count,
                         detected_image_path, detected_image, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            "camera",
                            filename,
                            output_path,
                            total,
                            counts["car"],
                            counts["truck"],
                            counts["bus"],
                            counts["motorcycle"],
                            counts["bicycle"],
                            output_path,
                            detected_blob,
                            created_at,
                        ),
                    )
                    conn.commit()
                    record_id = cursor.lastrowid
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
            record_id = 0

        return jsonify(
            {
                "success": True,
                "record_id": record_id,
                "total_vehicles": total,
                "counts": counts,
                "detected_image": f"/output/detected/detected_{filename}",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
