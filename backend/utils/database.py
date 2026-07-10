import os
import pymysql
from pymysql import cursors
from contextlib import contextmanager

DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", 3306)),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "123456"),
    "database": os.environ.get("MYSQL_DATABASE", "bishe_vehicle"),
    "cursorclass": cursors.DictCursor,
}


@contextmanager
def get_db_connection():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    config_without_db = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    db_name = DB_CONFIG["database"]

    conn = pymysql.connect(**config_without_db)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        conn.commit()
    finally:
        conn.close()

    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    file_type ENUM('image', 'video', 'camera') NOT NULL,
                    file_name VARCHAR(255),
                    file_path VARCHAR(500),
                    total_vehicles INT DEFAULT 0,
                    car_count INT DEFAULT 0,
                    truck_count INT DEFAULT 0,
                    bus_count INT DEFAULT 0,
                    motorcycle_count INT DEFAULT 0,
                    bicycle_count INT DEFAULT 0,
                    enhanced_image_path VARCHAR(500),
                    detected_image_path VARCHAR(500),
                    processing_time FLOAT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            try:
                cursor.execute("""
                    ALTER TABLE detection_records 
                    ADD COLUMN processing_time FLOAT DEFAULT NULL AFTER detected_image_path
                """)
                conn.commit()
                print("数据库表已更新：添加processing_time字段")
            except Exception:
                pass

            try:
                cursor.execute("""
                    ALTER TABLE detection_records 
                    ADD COLUMN enhanced_image LONGBLOB AFTER enhanced_image_path
                """)
                conn.commit()
                print("数据库表已更新：添加enhanced_image字段")
            except Exception:
                pass

            try:
                cursor.execute("""
                    ALTER TABLE detection_records 
                    ADD COLUMN detected_image LONGBLOB AFTER detected_image_path
                """)
                conn.commit()
                print("数据库表已更新：添加detected_image字段")
            except Exception:
                pass

            try:
                cursor.execute("""
                    ALTER TABLE detection_records 
                    ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)
                conn.commit()
                print("数据库表已更新：添加created_at字段")
            except Exception:
                pass

            try:
                cursor.execute("""
                    UPDATE detection_records 
                    SET created_at = NOW() 
                    WHERE created_at IS NULL
                """)
                conn.commit()
                affected = cursor.rowcount
                if affected > 0:
                    print(f"数据库表已更新：回填{affected}条记录的created_at")
            except Exception:
                pass

        conn.commit()
    finally:
        conn.close()
