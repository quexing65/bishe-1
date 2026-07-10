import os
import sys
import traceback
import re
from flask import Flask, send_from_directory, jsonify, request, Response
from flask_cors import CORS

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.database import init_database

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 启用CORS
    CORS(app)

    # 创建必要的目录
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(app.config["OUTPUT_FOLDER"], "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(app.config["OUTPUT_FOLDER"], "detected"), exist_ok=True)

    # 初始化数据库
    init_database()

    # 注册蓝图
    from routes.image_routes import image_bp
    from routes.video_routes import video_bp
    from routes.camera_routes import camera_bp
    from routes.history_routes import history_bp

    app.register_blueprint(image_bp, url_prefix="/api/image")
    app.register_blueprint(video_bp, url_prefix="/api/video")
    app.register_blueprint(camera_bp, url_prefix="/api/camera")
    app.register_blueprint(history_bp, url_prefix="")

    # 全局错误处理器
    @app.errorhandler(Exception)
    def handle_exception(e):
        error_trace = traceback.format_exc()
        print(f"\n{'=' * 50}")
        print(f"未捕获的异常: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print(f"堆栈跟踪:\n{error_trace}")
        print(f"{'=' * 50}\n")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

    # 前端页面路由
    @app.route("/")
    def index():
        return send_from_directory(os.path.join(PROJECT_DIR, "frontend"), "index.html")

    @app.route("/pages/<path:path>")
    def serve_pages(path):
        return send_from_directory(os.path.join(PROJECT_DIR, "frontend", "pages"), path)

    @app.route("/css/<path:path>")
    def serve_css(path):
        return send_from_directory(os.path.join(PROJECT_DIR, "frontend", "css"), path)

    @app.route("/js/<path:path>")
    def serve_js(path):
        return send_from_directory(os.path.join(PROJECT_DIR, "frontend", "js"), path)

    @app.route("/output/<path:path>")
    def serve_output(path):
        full_path = os.path.join(PROJECT_DIR, "output", path)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return _serve_video_range(full_path, "video/mp4")
        return send_from_directory(os.path.join(PROJECT_DIR, "output"), path)

    # 处理Chrome DevTools的请求
    @app.route("/.well-known/appspecific/com.chrome.devtools.json")
    def chrome_devtools():
        return jsonify({}), 204

    return app


def _serve_video_range(file_path, content_type):
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")

    if not range_header:
        with open(file_path, "rb") as f:
            data = f.read()
        resp = Response(data, 200, mimetype=content_type,
                        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"})
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    m = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not m:
        return Response("Invalid range", 416)

    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else min(start + 2 * 1024 * 1024, file_size - 1)
    if start >= file_size:
        return Response("Range not satisfiable", 416, headers={"Content-Range": f"bytes */{file_size}"})

    length = end - start + 1
    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    resp = Response(data, 206, mimetype=content_type,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Content-Length": str(length),
                        "Accept-Ranges": "bytes",
                    })
    resp.headers["Cache-Control"] = "no-cache"
    return resp


if __name__ == "__main__":
    app = create_app()
    print(f"FRONTEND_FOLDER: {os.path.join(PROJECT_DIR, 'frontend')}")
    print(
        f"Index.html exists: {os.path.exists(os.path.join(PROJECT_DIR, 'frontend', 'index.html'))}"
    )
    app.run(host="0.0.0.0", port=5000, debug=False)
