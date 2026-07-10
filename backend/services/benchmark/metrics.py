import os
import numpy as np
import cv2


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


def calculate_map(gt_boxes, pred_boxes, iou_threshold=0.5):
    if not gt_boxes:
        if not pred_boxes:
            return 1.0
        return 0.0
    if not pred_boxes:
        return 0.0

    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    tp = [0] * len(pred_boxes_sorted)
    fp = [0] * len(pred_boxes_sorted)
    used = [False] * len(gt_boxes)

    for i, pred in enumerate(pred_boxes_sorted):
        best_iou = 0
        best_idx = -1

        for j, gt in enumerate(gt_boxes):
            if used[j] or pred[5] != gt[5]:
                continue
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= iou_threshold and best_idx != -1:
            used[best_idx] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / len(gt_boxes)

    ap = 0
    prev_recall = 0
    for p, r in zip(precision, recall):
        if r > prev_recall:
            ap += p * (r - prev_recall)
            prev_recall = r

    return ap


def calculate_map_range(gt_boxes, pred_boxes):
    if not gt_boxes or not pred_boxes:
        return 0.0
    aps = []
    for iou in np.arange(0.5, 1.0, 0.05):
        ap = calculate_map(gt_boxes, pred_boxes, iou)
        aps.append(ap)
    return float(np.mean(aps))


def calculate_map_per_class(gt_boxes, pred_boxes, class_ids, iou_threshold=0.5):
    result = {}
    for cls_id in class_ids:
        gt_cls = [b for b in gt_boxes if b[5] == cls_id]
        pred_cls = [b for b in pred_boxes if b[5] == cls_id]
        result[cls_id] = calculate_map(gt_cls, pred_cls, iou_threshold)
    return result


def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    if not gt_boxes or not pred_boxes:
        return 0.0, 0.0

    pred_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    used = [False] * len(gt_boxes)
    tp_count = 0

    for pred in pred_sorted:
        best_iou = 0
        best_idx = -1
        for j, gt in enumerate(gt_boxes):
            if used[j] or pred[5] != gt[5]:
                continue
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx != -1:
            used[best_idx] = True
            tp_count += 1

    precision = tp_count / len(pred_sorted) if pred_sorted else 0
    recall = tp_count / len(gt_boxes) if gt_boxes else 0
    return precision, recall


def calculate_psnr_ssim(img1, img2):
    if img1 is None or img2 is None:
        return 0.0, 0.0
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0, 1.0
    psnr = 10 * np.log10(255.0 ** 2 / mse)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1_f = img1.astype(float)
    img2_f = img2.astype(float)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = float(ssim_map.mean())

    return round(psnr, 4), round(ssim, 4)


def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append([cls_id, xc, yc, w, h])
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    cls_id, xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2, 1.0, cls_id]
