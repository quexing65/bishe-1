import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROJECT_DIR


class DatasetConfig:
    def __init__(self, name, image_dir, label_dir, enhancements):
        self.name = name
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.enhancements = enhancements

    def get_images(self):
        if not os.path.exists(self.image_dir):
            return []
        return sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def get_label_path(self, image_name):
        label_name = os.path.splitext(image_name)[0] + ".txt"
        return os.path.join(self.label_dir, label_name)


DEFAULT_CONFIG = {
    "datasets": {
        "foggy": {
            "image_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "foggy", "images"),
            "label_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "foggy", "labels"),
            "enhancements": ["dehaze", "ffa_dehaze", "ffa_dehaze_ots"],
        },
        "rainy": {
            "image_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "rainy", "images"),
            "label_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "rainy", "labels"),
            "enhancements": ["derain"],
        },
        "lowlight": {
            "image_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "lowlight", "images"),
            "label_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "lowlight", "labels"),
            "enhancements": ["mbllen_onnx"],
        },
        "normal": {
            "image_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "normal", "images"),
            "label_dir": os.path.join(PROJECT_DIR, "weights", "test_data", "normal", "labels"),
            "enhancements": [],
        },
    },
    "models": ["yolov8x"],
    "detection": {
        "conf_threshold": 0.35,
        "vehicle_classes": [1, 2, 3, 5, 7],
    },
}


def load_test_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        raw = DEFAULT_CONFIG

    datasets = {}
    for name, cfg in raw.get("datasets", {}).items():
        img_dir = cfg["image_dir"]
        lbl_dir = cfg["label_dir"]
        if not os.path.isabs(img_dir):
            img_dir = os.path.join(PROJECT_DIR, img_dir)
        if not os.path.isabs(lbl_dir):
            lbl_dir = os.path.join(PROJECT_DIR, lbl_dir)
        datasets[name] = DatasetConfig(
            name=name,
            image_dir=img_dir,
            label_dir=lbl_dir,
            enhancements=cfg.get("enhancements", []),
        )

    return {
        "datasets": datasets,
        "models": raw.get("models", ["yolov8x"]),
        "conf_threshold": raw.get("detection", {}).get("conf_threshold", 0.35),
        "vehicle_classes": set(raw.get("detection", {}).get("vehicle_classes", [1, 2, 3, 5, 7])),
    }
