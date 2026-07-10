import time
import gc
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import uuid
import logging
from datetime import datetime
from utils.database import get_db_connection
from config import PROJECT_DIR
from services.image_enhancer import ImageEnhancer

logger = logging.getLogger(__name__)

video_bp = Blueprint("video", __name__)

UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads", "videos")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "detected")
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_model_cache = {}
_video_tracked = {}


def load_model(name):
    if name not in _model_cache:
        from services.vehicle_detector import VehicleDetector
        try:
            model, actual_name = VehicleDetector.load_model_safe(name)
            _model_cache[name] = model
            logger.info(f"视频检测加载模型: {name} (实际: {actual_name})")
        except Exception as e:
            logger.error(f"模型 {name} 加载失败: {e}")
            raise
    return _model_cache[name]


def calc_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def do_enhance(frame, etype, enhancer, auto_enhancer, strength=1.5):
    if etype == "auto":
        return auto_enhancer.enhance_frame(frame, auto_enhancer.detect_condition_from_frame(frame))
    elif etype == "dehaze" or etype == "nafnet_dehaze":
        return enhancer.nafnet_dehaze_frame(frame)
    elif etype == "ffa_dehaze":
        return enhancer.ffa_dehaze_frame(frame, strength)
    elif etype == "ffa_dehaze_ots":
        return enhancer.ffa_dehaze_ots_frame(frame, strength)
    elif etype == "low_light":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    elif etype == "mbllen":
        m = enhancer.mbllen
        return m.enhance_frame(frame) if m else frame
    elif etype == "mbllen_onnx":
        m = enhancer.onnx_mbllen
        return m.enhance_frame(frame) if m else frame
    elif etype == "zero_dce":
        d = enhancer.dce
        return d.enhance_frame(frame) if d else frame
    elif etype == "derain" or etype == "nafnet_derain":
        return enhancer.nafnet_derain_frame(frame)
    elif etype == "legacy_dehaze":
        return ImageEnhancer._clahe_dehaze_frame(frame)
    elif etype == "legacy_derain":
        return ImageEnhancer._morph_derain_frame(frame)
    return frame


@video_bp.route("/enhance", methods=["POST"])
def enhance_video():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file"}), 400

        etype = request.form.get("enhancement_type", "auto")
        strength = float(request.form.get("strength", "1.5"))

        fname = secure_filename(file.filename)
        ext = fname.rsplit(".", 1)[-1]
        uname = f"{uuid.uuid4().hex}.{ext}"

        upath = os.path.join(UPLOAD_DIR, uname)
        file.save(upath)

        enhanced_name = f"enhanced_{uname}"
        epath = os.path.join(OUTPUT_DIR, enhanced_name)

        start = time.time()

        from services.auto_enhancer import AutoEnhancer
        auto_enhancer = AutoEnhancer()
        enhancer = ImageEnhancer()

        cap = cv2.VideoCapture(upath)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video"}), 500

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        for codec_tag in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec_tag)
            out = cv2.VideoWriter(epath, fourcc, fps, (w, h))
            if out.isOpened():
                break

        if out is None or not out.isOpened():
            cap.release()
            return jsonify({"error": "Cannot create output video - codec not supported"}), 500

        frame_num = 0
        last_enhanced = None
        while cap.isOpened():
            ret, frm = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % 2 != 0:
                try:
                    frm = do_enhance(frm, etype, enhancer, auto_enhancer, strength)
                except Exception as e:
                    logger.warning(f"Frame {frame_num} enhance failed: {e}, using original")
                last_enhanced = frm
            elif last_enhanced is not None:
                frm = last_enhanced
            out.write(frm)

        cap.release()
        out.release()

        enhance_time = round(time.time() - start, 2)
        size_mb = round(os.path.getsize(epath) / (1024 * 1024), 2)

        return jsonify({
            "success": True,
            "unique_name": uname,
            "enhanced_video": f"/output/detected/{enhanced_name}",
            "enhanced_path": epath,
            "original_path": upath,
            "enhance_time": enhance_time,
            "size_mb": size_mb,
        })

    except Exception as e:
        import traceback
        logger.error(f"Video enhance error: {traceback.format_exc()}")
        return jsonify({"error": f"视频增强失败: {str(e)}"}), 500


@video_bp.route("/detect", methods=["POST"])
def detect_video():
    global _video_tracked

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400

        enhanced_path = data.get("enhanced_path", "")
        unique_name = data.get("unique_name", "")
        model_type = data.get("model_type", "yolov8n")
        original_filename = data.get("original_filename", "")

        if not enhanced_path or not os.path.exists(enhanced_path):
            return jsonify({"error": f"增强后的视频不存在: {enhanced_path}"}), 400

        start = time.time()

        opath = os.path.join(OUTPUT_DIR, f"detected_{unique_name}")

        cap = cv2.VideoCapture(enhanced_path)
        if not cap.isOpened():
            return jsonify({"error": f"Cannot open video: {enhanced_path}"}), 500

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        for codec_tag in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec_tag)
            out = cv2.VideoWriter(opath, fourcc, fps, (w, h))
            if out.isOpened():
                break

        if out is None or not out.isOpened():
            cap.release()
            return jsonify({"error": "Cannot create output video - codec not supported"}), 500

        frame_num = 0
        maxcnt = 0
        _video_tracked = {}
        gc.collect()
        try:
            model = load_model(model_type)
        except Exception as e:
            cap.release()
            if out:
                out.release()
            return jsonify({"error": f"模型加载失败（可能内存不足）: {str(e)}"}), 500

        while cap.isOpened():
            ret, frm = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % 6 != 0:
                out.write(frm)
                continue

            res = model(frm, conf=0.35, imgsz=640, verbose=False)
            det = []
            for r in res:
                for b in r.boxes:
                    c = int(b.cls[0])
                    if c in [1, 2, 3, 5, 7]:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        vt = {2: "car", 7: "truck", 5: "bus", 3: "motorcycle", 1: "bicycle"}.get(c)
                        det.append({"b": (x1, y1, x2, y2), "t": vt})
                        cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frm, vt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            nt = {}
            for d in det:
                m = False
                for tid, ti in _video_tracked.items():
                    if calc_iou(d["b"], ti["b"]) > 0.3:
                        nt[tid] = d
                        m = True
                        break
                if not m:
                    nt[str(uuid.uuid4())[:8]] = d
            _video_tracked = nt
            if len(_video_tracked) > maxcnt:
                maxcnt = len(_video_tracked)
            out.write(frm)

        cap.release()
        out.release()

        cnt = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
        for t in _video_tracked.values():
            if t["t"] in cnt:
                cnt[t["t"]] += 1

        detect_time = round(time.time() - start, 2)

        created_at = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        record_id = 0
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO detection_records
                        (file_type,file_name,file_path,total_vehicles,car_count,truck_count,bus_count,motorcycle_count,bicycle_count,detected_image_path,created_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        ("video", original_filename, enhanced_path, maxcnt,
                         cnt["car"], cnt["truck"], cnt["bus"],
                         cnt["motorcycle"], cnt["bicycle"], opath, created_at),
                    )
                    conn.commit()
                    record_id = cursor.lastrowid
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
            record_id = 0

        return jsonify({
            "success": True,
            "record_id": record_id,
            "total_vehicles": maxcnt,
            "counts": cnt,
            "output_video": f"/output/detected/detected_{unique_name}",
            "detect_time": detect_time,
        })

    except Exception as e:
        import traceback
        logger.error(f"Video detect error: {traceback.format_exc()}")
        return jsonify({"error": f"视频检测失败: {str(e)}"}), 500
