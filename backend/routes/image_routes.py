import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from services.image_enhancer import ImageEnhancer
from services.auto_enhancer import AutoEnhancer
from services.vehicle_detector import VehicleDetector
from utils.database import get_db_connection

image_bp = Blueprint("image", __name__)


@image_bp.errorhandler(Exception)
def handle_exception(e):
    """捕获所有未处理的异常"""
    import traceback

    error_trace = traceback.format_exc()
    print("\n=== 未捕获的异常 ===")
    print(f"异常类型: {type(e).__name__}")
    print(f"异常信息: {str(e)}")
    print(f"堆栈跟踪:\n{error_trace}")
    print("=== 异常结束 ===\n")
    return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@image_bp.route("/enhance", methods=["POST"])
def enhance_image():
    import time

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    enhancement_type = request.form.get("enhancement_type", "dehaze")
    strength = float(request.form.get("strength", "1.5"))

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, current_app.config["ALLOWED_EXTENSIONS_IMAGE"]):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"

    upload_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "images")
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, unique_name)
    file.save(image_path)

    enhanced_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], "enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)

    start_time = time.time()

    try:
        enhancer = ImageEnhancer()
        enhanced_name = f"enhanced_{enhancement_type}_{unique_name}"
        enhanced_path = os.path.join(enhanced_dir, enhanced_name)
        enhancer.enhance_image(image_path, enhanced_path, enhancement_type, strength)
    except Exception as e:
        print(f"图像增强失败: {e}")
        return jsonify({"error": f"图像增强失败: {str(e)}"}), 500

    enhance_time = round(time.time() - start_time, 2)

    return jsonify({
        "success": True,
        "unique_name": unique_name,
        "enhanced_image": f"/output/enhanced/{enhanced_name}",
        "enhanced_path": enhanced_path,
        "enhance_time": enhance_time,
    })


@image_bp.route("/detect", methods=["POST"])
def detect_vehicles():
    import time
    import json

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    enhanced_path = data.get("enhanced_path", "")
    unique_name = data.get("unique_name", "")
    model_type = data.get("model_type", "yolov8s")
    image_path = data.get("image_path", "")
    original_filename = data.get("original_filename", "")

    if not enhanced_path or not os.path.exists(enhanced_path):
        return jsonify({"error": "增强后的图片不存在"}), 400

    detected_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], "detected")
    os.makedirs(detected_dir, exist_ok=True)

    detected_name = f"detected_{unique_name}"
    detected_path = os.path.join(detected_dir, detected_name)

    start_time = time.time()

    detector = VehicleDetector()
    counts = detector.detect_vehicles(enhanced_path, detected_path)

    total_vehicles = sum(counts.values())
    detect_time = round(time.time() - start_time, 2)

    enhanced_blob = None
    detected_blob = None
    try:
        with open(enhanced_path, "rb") as f:
            enhanced_blob = f.read()
        with open(detected_path, "rb") as f:
            detected_blob = f.read()
    except Exception as e:
        print(f"读取图片失败: {e}")

    record_id = None
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            sql = """
                INSERT INTO detection_records
                (file_type, file_name, file_path, total_vehicles, car_count,
                 truck_count, bus_count, motorcycle_count, bicycle_count,
                 enhanced_image_path, detected_image_path,
                 enhanced_image, detected_image, processing_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                "image", original_filename, image_path, total_vehicles,
                counts.get("car", 0), counts.get("truck", 0),
                counts.get("bus", 0), counts.get("motorcycle", 0),
                counts.get("bicycle", 0), enhanced_path, detected_path,
                enhanced_blob, detected_blob, detect_time,
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            ))
            conn.commit()
            record_id = cursor.lastrowid
    except Exception as e:
        print(f"保存到数据库失败: {e}")

    return jsonify({
        "success": True,
        "record_id": record_id,
        "counts": counts,
        "total_vehicles": total_vehicles,
        "detected_image": f"/output/detected/{detected_name}",
        "detect_time": detect_time,
    })
