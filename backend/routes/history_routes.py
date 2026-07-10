import os
import glob
import csv
from config import PROJECT_DIR
from flask import Blueprint, request, jsonify, Response
from utils.database import get_db_connection

history_bp = Blueprint("history", __name__)

COLUMNS_EXCEPT_BLOB = """
    id, file_type, file_name, file_path, total_vehicles,
    car_count, truck_count, bus_count, motorcycle_count, bicycle_count,
    enhanced_image_path, detected_image_path, processing_time, created_at
"""


def _safe_remove(path):
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _fetch_file_paths(record_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_path, enhanced_image_path, detected_image_path "
            "FROM detection_records WHERE id = %s",
            (record_id,),
        )
        return cursor.fetchone()


def _delete_record_files(record):
    if not record:
        return
    if isinstance(record, dict):
        _safe_remove(record.get("file_path"))
        _safe_remove(record.get("enhanced_image_path"))
        _safe_remove(record.get("detected_image_path"))
    else:
        _safe_remove(record[0])
        _safe_remove(record[1])
        _safe_remove(record[2])


@history_bp.route("/api/history", methods=["GET"])
def get_history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    offset = (page - 1) * per_page

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM detection_records")
            total_result = cursor.fetchone()
            total = (
                total_result[0]
                if isinstance(total_result, tuple)
                else total_result["total"]
            )

            cursor.execute(
                f"SELECT {COLUMNS_EXCEPT_BLOB} FROM detection_records "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (per_page, offset),
            )
            records = cursor.fetchall()

            records_dict = []
            for record in records:
                if isinstance(record, dict):
                    records_dict.append(record)
                else:
                    columns = [desc[0] for desc in cursor.description]
                    records_dict.append(dict(zip(columns, record)))

            return jsonify({
                "success": True,
                "total": total,
                "page": page,
                "per_page": per_page,
                "records": records_dict,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/<int:record_id>", methods=["GET"])
def get_history_record(record_id):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {COLUMNS_EXCEPT_BLOB} FROM detection_records WHERE id = %s",
                (record_id,),
            )
            record = cursor.fetchone()

            if not record:
                return jsonify({"error": "Record not found"}), 404

            if isinstance(record, dict):
                record_dict = record
            else:
                columns = [desc[0] for desc in cursor.description]
                record_dict = dict(zip(columns, record))

            return jsonify({"success": True, "record": record_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/<int:record_id>/image/<image_type>", methods=["GET"])
def get_record_image(record_id, image_type):
    if image_type not in ("enhanced", "detected"):
        return jsonify({"error": "Invalid image type"}), 400

    blob_col = f"{image_type}_image"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {blob_col} FROM detection_records WHERE id = %s",
                (record_id,),
            )
            record = cursor.fetchone()

            if not record:
                return jsonify({"error": "Record not found"}), 404

            blob_data = record[blob_col] if isinstance(record, dict) else record[0]
            if not blob_data:
                return jsonify({"error": "Image not found"}), 404

            return Response(blob_data, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history_record(record_id):
    try:
        record = _fetch_file_paths(record_id)
        _delete_record_files(record)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detection_records WHERE id = %s", (record_id,))
            conn.commit()

            return jsonify({"success": True, "message": "Record deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/delete", methods=["POST"])
def delete_multiple_records():
    try:
        data = request.get_json()
        ids = data.get("ids", [])

        if not ids:
            return jsonify({"error": "No IDs provided"}), 400

        for rid in ids:
            record = _fetch_file_paths(rid)
            _delete_record_files(record)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["%s"] * len(ids))
            cursor.execute(
                f"DELETE FROM detection_records WHERE id IN ({placeholders})", ids
            )
            conn.commit()

            return jsonify({
                "success": True,
                "message": f"Deleted {cursor.rowcount} records successfully",
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@history_bp.route("/api/history/delete_all", methods=["POST"])
def delete_all_records():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT file_path, enhanced_image_path, detected_image_path "
                "FROM detection_records"
            )
            all_records = cursor.fetchall()
            for record in all_records:
                _delete_record_files(record)

            cursor.execute("DELETE FROM detection_records")
            conn.commit()

            return jsonify(
                {"success": True, "message": "All records deleted successfully"}
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _parse_benchmark_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _latest_benchmark_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


@history_bp.route("/api/benchmark/results", methods=["GET"])
def get_benchmark_results():
    benchmark_dir = os.path.join(PROJECT_DIR, "output", "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    summary_path = _latest_benchmark_file(os.path.join(benchmark_dir, "summary_*.csv"))
    detailed_path = _latest_benchmark_file(os.path.join(benchmark_dir, "detailed_*.csv"))

    summary = []
    detailed = []

    if summary_path:
        try:
            summary = _parse_benchmark_csv(summary_path)
        except Exception:
            pass

    if detailed_path:
        try:
            detailed = _parse_benchmark_csv(detailed_path)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "summary": summary,
        "detailed": detailed,
        "empty": len(summary) == 0 and len(detailed) == 0,
    })
