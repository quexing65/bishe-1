import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


class Config:
    SECRET_KEY = "bishe-vehicle-detection-2024"
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "123456"
    MYSQL_DATABASE = "bishe_vehicle"

    UPLOAD_FOLDER = os.path.join(PROJECT_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "output")
    WEIGHTS_FOLDER = os.path.join(PROJECT_DIR, "weights")

    ALLOWED_EXTENSIONS_IMAGE = {"png", "jpg", "jpeg", "bmp", "webp"}
    ALLOWED_EXTENSIONS_VIDEO = {"mp4", "avi", "mov", "mkv"}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
