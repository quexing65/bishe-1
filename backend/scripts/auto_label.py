import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from ultralytics import YOLO

TEST_DATA = os.path.join(PROJECT_DIR, "weights", "test_data")
MODEL_PATH = os.path.join(PROJECT_DIR, "weights", "detection", "yolov8x.onnx")
VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}
CONF_THRESHOLD = 0.25


def auto_label():
    model = YOLO(MODEL_PATH)
    total = 0
    start = time.time()

    for split in ["foggy", "rainy", "lowlight", "normal"]:
        img_dir = os.path.join(TEST_DATA, split, "images")
        lbl_dir = os.path.join(TEST_DATA, split, "labels")

        if not os.path.exists(img_dir):
            print(f"[{split}] images 不存在，跳过")
            continue

        images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if not images:
            print(f"[{split}] 无图片，跳过")
            continue

        os.makedirs(lbl_dir, exist_ok=True)
        count = 0

        for fname in images:
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

            results = model(img_path, conf=CONF_THRESHOLD, verbose=False)
            lines = []

            for r in results:
                h, w = r.orig_shape
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(lbl_path, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            count += len(lines)
            total += len(lines)

        elapsed = time.time() - start
        print(f"[{split}] {len(images)} 张图, {count} 个标注, 耗时 {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"\n完成! 共 {total} 个标注, 总耗时 {elapsed:.1f}s")
    print(f"标注目录: {TEST_DATA}/{{split}}/labels/")


if __name__ == "__main__":
    auto_label()
