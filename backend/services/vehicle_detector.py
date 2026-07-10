import os
import sys
import gc
import logging
import cv2

YOLO = None
YOLO_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[YOLO] ultralytics 导入失败（不影响 NAFNet 推理）: {e}")

logger = logging.getLogger(__name__)

WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights"
)

MODEL_FALLBACK_ORDER = ["yolov8x", "yolov8s", "yolov8n"]

YOLO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleDetector:
    _model_cache = {}

    @staticmethod
    def get_model_path(model_name="yolov8x"):
        base = os.path.join(WEIGHTS_DIR, "detection")
        onnx_path = os.path.join(base, f"{model_name}.onnx")
        pt_path = os.path.join(base, f"{model_name}.pt")
        if os.path.exists(onnx_path):
            return onnx_path
        if os.path.exists(pt_path):
            return pt_path
        return f"{model_name}.pt"

    @staticmethod
    def load_model_safe(model_name="yolov8x"):
        """安全加载模型，内存不足时自动降级到更小的模型"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics 未安装或导入失败")

        start_idx = MODEL_FALLBACK_ORDER.index(model_name) if model_name in MODEL_FALLBACK_ORDER else 0

        for candidate in MODEL_FALLBACK_ORDER[start_idx:]:
            path = VehicleDetector.get_model_path(candidate)
            if path in VehicleDetector._model_cache:
                logger.info(f"使用缓存模型: {candidate}")
                return VehicleDetector._model_cache[path], candidate

            gc.collect()
            try:
                model = YOLO(path)
                VehicleDetector._model_cache[path] = model
                logger.info(f"成功加载模型: {candidate} ({path})")
                return model, candidate
            except Exception as e:
                logger.warning(f"模型 {candidate} 加载失败: {type(e).__name__}: {e}, 尝试降级...")
                gc.collect()

        raise RuntimeError("所有 YOLO 模型均无法加载（内存不足或文件缺失）")

    def __init__(self, weights_path=None, auto_fallback=True):
        if weights_path is None:
            weights_path = self.get_model_path("yolov8x")

        if weights_path in VehicleDetector._model_cache:
            self.model = VehicleDetector._model_cache[weights_path]
            self._actual_model = "cached"
        elif auto_fallback and YOLO_AVAILABLE:
            self.model, self._actual_model = self.load_model_safe("yolov8x")
        else:
            self.model = YOLO(weights_path)
            VehicleDetector._model_cache[weights_path] = self.model
            self._actual_model = "custom"

    def detect_vehicles(self, image_path, output_path):
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 运行检测
        results = self.model(img)

        # 统计车辆数量
        counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}

        # 绘制检测结果
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 2:  # car
                    counts["car"] += 1
                elif cls == 7:  # truck
                    counts["truck"] += 1
                elif cls == 5:  # bus
                    counts["bus"] += 1
                elif cls == 3:  # motorcycle
                    counts["motorcycle"] += 1
                elif cls == 1:  # bicycle
                    counts["bicycle"] += 1

                # 绘制边界框
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if cls in YOLO_CLASSES:
                    cv2.putText(
                        img,
                        f"{YOLO_CLASSES[cls]} {box.conf[0]:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        img,
                        f"class {cls} {box.conf[0]:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # 保存检测结果
        cv2.imwrite(output_path, img)

        return counts



