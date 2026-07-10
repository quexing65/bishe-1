import os
import csv
import json
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROJECT_DIR


OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "benchmark")


class ReportGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_detailed_csv(self, results):
        path = os.path.join(self.output_dir, f"detailed_{self.timestamp}.csv")
        if not results:
            print("  无结果可保存")
            return path

        fieldnames = list(results[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"  详细结果已保存: {path}")
        return path

    def save_summary_csv(self, results):
        path = os.path.join(self.output_dir, f"summary_{self.timestamp}.csv")
        if not results:
            return path

        groups = {}
        for r in results:
            key = (r.get("dataset", ""), r.get("enhancement", "none"), r.get("model", ""))
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        summary_rows = []
        for (dataset, enhancement, model), items in groups.items():
            row = {
                "dataset": dataset,
                "enhancement": enhancement,
                "model": model,
                "image_count": len(items),
                "avg_brightness_before": round(np.mean([i["brightness_before"] for i in items]), 2),
                "avg_brightness_after": round(np.mean([i["brightness_after"] for i in items]), 2),
                "avg_brightness_gain": round(np.mean([i["brightness_gain"] for i in items]), 2),
                "avg_enhancement_time_ms": round(np.mean([i["enhancement_time_ms"] for i in items]), 2),
                "avg_detect_time_ms": round(np.mean([i["detect_time_ms"] for i in items]), 2),
                "avg_total_time_ms": round(np.mean([i["total_time_ms"] for i in items]), 2),
                "avg_vehicle_count": round(np.mean([i["vehicle_count"] for i in items]), 2),
                "avg_confidence": round(np.mean([i["avg_confidence"] for i in items]), 4),
                "mAP50": round(np.mean([i["mAP50"] for i in items]), 4),
                "mAP50_95": round(np.mean([i["mAP50_95"] for i in items]), 4),
                "avg_psnr": round(np.mean([i["psnr"] for i in items]), 2),
                "avg_ssim": round(np.mean([i["ssim"] for i in items]), 4),
            }
            summary_rows.append(row)

        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

        print(f"  汇总结果已保存: {path}")
        return path

    def save_comparison_csv(self, results):
        path = os.path.join(self.output_dir, f"comparison_{self.timestamp}.csv")
        if not results:
            return path

        comparisons = []
        baseline_map = {}
        for r in results:
            key = (r.get("dataset", ""), r.get("model", ""), r.get("filename", ""))
            if r.get("enhancement", "none") == "none":
                baseline_map[key] = r

        for r in results:
            if r.get("enhancement", "none") == "none":
                continue
            key = (r.get("dataset", ""), r.get("model", ""), r.get("filename", ""))
            baseline = baseline_map.get(key)
            if not baseline:
                continue

            comparisons.append({
                "dataset": r["dataset"],
                "model": r["model"],
                "filename": r["filename"],
                "enhancement": r["enhancement"],
                "baseline_mAP50": baseline["mAP50"],
                "enhanced_mAP50": r["mAP50"],
                "mAP50_diff": round(r["mAP50"] - baseline["mAP50"], 4),
                "mAP50_change_pct": round((r["mAP50"] - baseline["mAP50"]) / max(baseline["mAP50"], 1e-6) * 100, 2),
                "baseline_mAP50_95": baseline["mAP50_95"],
                "enhanced_mAP50_95": r["mAP50_95"],
                "mAP50_95_diff": round(r["mAP50_95"] - baseline["mAP50_95"], 4),
                "baseline_vehicles": baseline["vehicle_count"],
                "enhanced_vehicles": r["vehicle_count"],
                "vehicle_diff": r["vehicle_count"] - baseline["vehicle_count"],
            })

        if comparisons:
            fieldnames = list(comparisons[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparisons)

        print(f"  对比结果已保存: {path}")
        return path

    def print_summary(self, results):
        if not results:
            print("  无结果")
            return

        print("\n" + "=" * 70)
        print("  基准测试汇总")
        print("=" * 70)

        groups = {}
        for r in results:
            key = (r.get("dataset", ""), r.get("enhancement", "none"), r.get("model", ""))
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        for (dataset, enhancement, model), items in sorted(groups.items()):
            avg_map50 = np.mean([i["mAP50"] for i in items])
            avg_map50_95 = np.mean([i["mAP50_95"] for i in items])
            avg_enh_time = np.mean([i["enhancement_time_ms"] for i in items])
            avg_det_time = np.mean([i["detect_time_ms"] for i in items])
            total_vehicles = sum([i["vehicle_count"] for i in items])

            print(f"\n  [{dataset}] 增强={enhancement} 模型={model} ({len(items)}张)")
            print(f"    mAP50:    {avg_map50:.4f}")
            print(f"    mAP50-95: {avg_map50_95:.4f}")
            print(f"    增强耗时: {avg_enh_time:.1f}ms")
            print(f"    检测耗时: {avg_det_time:.1f}ms")
            print(f"    检测车辆: {total_vehicles}")

        print("\n" + "=" * 70)

    def generate_all(self, results):
        self.save_detailed_csv(results)
        self.save_summary_csv(results)
        self.save_comparison_csv(results)
        self.print_summary(results)
