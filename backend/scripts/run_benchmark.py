import os
import sys
import time
import json
import contextlib
import io

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from backend.services.benchmark.config import load_test_config
from backend.services.benchmark.runner import BenchmarkRunner, ENHANCE_FUNC_MAP
from backend.services.benchmark.metrics import (
    load_yolo_labels, yolo_to_xyxy,
    calculate_map, calculate_map_range,
    calculate_precision_recall, calculate_psnr_ssim,
)
from backend.services.benchmark.report import ReportGenerator

MAX_IMAGES_PER_DATASET = 50


class SampledBenchmarkRunner(BenchmarkRunner):
    def run_all(self, progress_callback=None):
        all_results = []
        datasets = self.config["datasets"]
        models = self.config["models"]

        total_ops = 0
        for ds_cfg in datasets.values():
            n = min(len(ds_cfg.get_images()), MAX_IMAGES_PER_DATASET)
            total_ops += n * (len(ds_cfg.enhancements) + 1) * len(models)

        done = 0
        for ds_name, ds_cfg in datasets.items():
            images = ds_cfg.get_images()[:MAX_IMAGES_PER_DATASET]
            if not images:
                print(f"  [{ds_name}] 无图片，跳过")
                continue

            print(f"  [{ds_name}] {len(images)} 张图片 (采样自全量 {len(ds_cfg.get_images())})")

            enh_list = ds_cfg.enhancements + ["none"]

            for model_name in models:
                self._load_model(model_name)

                for img_name in images:
                    img_path = os.path.join(ds_cfg.image_dir, img_name)
                    lbl_path = ds_cfg.get_label_path(img_name)

                    for enh in enh_list:
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = self.run_single(
                                ds_name, img_path, lbl_path, enh, model_name
                            )
                        if result:
                            all_results.append(result)
                        done += 1
                        if progress_callback:
                            progress_callback(done, total_ops, ds_name, img_name, enh)

        return all_results


def main():
    print("=== 多源数据测试 ===\n")

    config = load_test_config()
    runner = SampledBenchmarkRunner(config)

    total_ops = 0
    for ds in config["datasets"].values():
        n = min(len(ds.get_images()), MAX_IMAGES_PER_DATASET)
        total_ops += n * (1 + len(ds.enhancements))

    print(f"数据集: {len(config['datasets'])} 个, 每个最多 {MAX_IMAGES_PER_DATASET} 张")
    print(f"总操作数: {total_ops}")
    print(f"检测模型: {config['models']}")
    print(f"置信度阈值: {config['conf_threshold']}\n")

    start_time = time.time()

    def progress(done, total, ds_name, img_name, enhancement):
        if done % 5 == 0 or done == total:
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            print(f"  [{done}/{total}] {ds_name} ({enhancement}) "
                  f"速度: {speed:.2f} img/s, 预计剩余: {eta/60:.1f} 分钟")

    results = runner.run_all(progress_callback=progress)

    elapsed = time.time() - start_time
    print(f"\n=== 测试完成 ===")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"总结果数: {len(results)}")

    report_gen = ReportGenerator()
    report_gen.generate_all(results)

    json_path = os.path.join(report_gen.output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {report_gen.output_dir}")


if __name__ == "__main__":
    main()
