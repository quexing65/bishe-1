import os
import random
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATA = os.path.join(PROJECT_DIR, "weights", "test_data")
OUTPUT_DIR = os.path.join(TEST_DATA, "verification")

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {0: (0,255,255), 1: (255,0,255), 2: (0,255,0), 3: (255,255,0), 5: (255,0,0), 7: (0,0,255)}


def draw_labels(image_path, label_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = COLORS.get(cls_id, (128, 128, 128))
                name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{name} {cls_id}", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    for split in ["foggy", "rainy", "lowlight", "normal"]:
        img_dir = os.path.join(TEST_DATA, split, "images")
        lbl_dir = os.path.join(TEST_DATA, split, "labels")
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(img_dir):
            print(f"[{split}] images 目录不存在，跳过")
            continue

        images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not images:
            print(f"[{split}] 无图片，跳过")
            continue

        samples = random.sample(images, min(5, len(images)))

        for fname in samples:
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            out_path = os.path.join(out_dir, f"verify_{fname}")
            draw_labels(img_path, lbl_path, out_path)

        print(f"[{split}] {len(samples)} 张验证图已保存到 {out_dir}")

    print(f"\n验证图片目录: {OUTPUT_DIR}")
    print("打开 verification/ 下的图片，检查标注框是否和实际车辆对得上。")


if __name__ == "__main__":
    main()
