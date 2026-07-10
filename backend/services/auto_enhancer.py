import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AutoEnhancer:
    _instance = None
    _initialized = False

    DEFAULT_THRESHOLDS = {
        "low_light_brightness": 80,
        "low_light_contrast": 50,
        "dehaze_sharpness": 300,
        "dehaze_contrast": 45,
        "derain_saturation": 30,
    }

    def __new__(cls, thresholds=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, thresholds=None):
        if AutoEnhancer._initialized:
            return
        AutoEnhancer._initialized = True
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._image_enhancer = None

    def _get_enhancer(self):
        if self._image_enhancer is None:
            from services.image_enhancer import ImageEnhancer
            self._image_enhancer = ImageEnhancer()
        return self._image_enhancer

    def detect_condition(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return "normal"
        return self._detect_from_frame(img)

    def detect_condition_from_frame(self, frame):
        return self._detect_from_frame(frame)

    def _detect_from_frame(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mean_brightness = np.mean(gray)
        contrast = gray.std()
        mean_saturation = np.mean(hsv[:, :, 1])
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        print(
            f"图像分析 - 亮度:{mean_brightness:.1f}, 对比度:{contrast:.1f}, 饱和度:{mean_saturation:.1f}, 清晰度:{sharpness:.1f}"
        )

        t = self.thresholds
        if mean_brightness < t["low_light_brightness"] or contrast < t["low_light_contrast"]:
            return "low_light"
        elif sharpness < t["dehaze_sharpness"] or contrast < t["dehaze_contrast"]:
            return "dehaze"
        elif mean_saturation < t["derain_saturation"]:
            return "derain"
        else:
            return "normal"

    def enhance_frame(self, frame, condition):
        if condition == "low_light":
            return self._enhance_low_light_frame(frame)
        elif condition == "dehaze":
            return self._dehaze_frame(frame)
        elif condition == "derain":
            return self._derain_frame(frame)
        return frame

    def _dehaze_frame(self, img):
        enhancer = self._get_enhancer()
        return enhancer.nafnet_dehaze_frame(img)

    def _enhance_low_light_frame(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _derain_frame(self, img):
        enhancer = self._get_enhancer()
        return enhancer.nafnet_derain_frame(img)

    def auto_enhance(self, image_path, output_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        condition = self.detect_condition(image_path)
        print(f"自动识别场景: {condition}")

        if condition == "low_light":
            print("使用低光照增强...")
            return self.enhance_low_light(image_path, output_path)
        elif condition == "dehaze":
            print("使用去雾增强...")
            return self.dehaze(image_path, output_path)
        elif condition == "derain":
            print("使用去雨增强...")
            return self.derain(image_path, output_path)
        else:
            return self.dehaze(image_path, output_path)

    def dehaze(self, image_path, output_path):
        enhancer = self._get_enhancer()
        return enhancer.nafnet_dehaze(image_path, output_path)

    def enhance_low_light(self, image_path, output_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, enhanced)
        return output_path

    def derain(self, image_path, output_path):
        enhancer = self._get_enhancer()
        return enhancer.nafnet_derain(image_path, output_path)

    def enhance_image(self, image_path, output_path, condition):
        if condition == "low_light":
            return self.enhance_low_light(image_path, output_path)
        elif condition == "dehaze":
            return self.dehaze(image_path, output_path)
        elif condition == "derain":
            return self.derain(image_path, output_path)
        else:
            return self.dehaze(image_path, output_path)
