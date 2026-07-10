import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_DIR


def get_weights_path(filename):
    return os.path.join(PROJECT_DIR, "weights", "enhancement", filename)


WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")


def build_mbllen():
    """构建MBLLEN网络"""
    from tensorflow.keras import layers, Model  # type: ignore[import-untyped]

    def EM(input, kernal_size, channel):
        conv_1 = layers.Conv2D(channel, (3, 3), activation="relu", padding="same")(
            input
        )
        conv_2 = layers.Conv2D(
            channel, (kernal_size, kernal_size), activation="relu", padding="valid"
        )(conv_1)
        conv_3 = layers.Conv2D(
            channel * 2, (kernal_size, kernal_size), activation="relu", padding="valid"
        )(conv_2)
        conv_4 = layers.Conv2D(
            channel * 4, (kernal_size, kernal_size), activation="relu", padding="valid"
        )(conv_3)
        conv_5 = layers.Conv2DTranspose(
            channel * 2, (kernal_size, kernal_size), activation="relu", padding="valid"
        )(conv_4)
        conv_6 = layers.Conv2DTranspose(
            channel, (kernal_size, kernal_size), activation="relu", padding="valid"
        )(conv_5)
        res = layers.Conv2DTranspose(
            3, (kernal_size, kernal_size), activation="relu", padding="valid"
        )(conv_6)
        return res

    inputs = layers.Input(shape=(None, None, 3))
    FEM = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    EM_com = EM(FEM, 5, 8)

    for j in range(3):
        for i in range(0, 3):
            FEM = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(FEM)
            EM1 = EM(FEM, 5, 8)
            EM_com = layers.Concatenate(axis=3)([EM_com, EM1])

    outputs = layers.Conv2D(3, (1, 1), activation="relu", padding="same")(EM_com)
    return Model(inputs, outputs)


class MBLLENEnhancer:
    """MBLLEN 低光照图像增强"""

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path

        if model_path is None:
            model_path = get_weights_path("LOL_img_lowlight.h5")

        if os.path.exists(model_path):
            try:
                print("正在构建MBLLEN模型...")
                self.model = build_mbllen()

                print("正在加载权重...")
                self.model.load_weights(model_path)
                print("MBLLEN模型加载成功!")
            except Exception as e:
                print(f"MBLLEN模型加载失败: {e}")
                self.model = None
        else:
            print(f"MBLLEN模型文件不存在: {model_path}")

    def postprocess(
        self, output, lowpercent=5, highpercent=95, maxrange=0.8, hsvgamma=0.8
    ):
        """官方后处理：线性放大 + 百分位重缩放 + HSV伽马校正

        参考MBLLEN官方实现的后处理方法
        """
        fake_B = output[0, :, :, :3].copy()

        # 第1步：线性放大（基于95百分位）
        gray_fake_B = (
            fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 2] * 0.114
        )
        percent_max = np.sum(gray_fake_B >= maxrange) / np.sum(gray_fake_B <= 1.0)
        max_value = np.percentile(gray_fake_B[:], highpercent)
        if percent_max < (100 - highpercent) / 100.0:
            scale = maxrange / max_value
            fake_B = fake_B * scale
            fake_B = np.minimum(fake_B, 1.0)

        # 第2步：减去5百分位并重缩放（仅当值<0.5时执行）
        gray_fake_B = (
            fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 2] * 0.114
        )
        sub_value = np.percentile(gray_fake_B[:], lowpercent)
        if sub_value < 0.5:
            fake_B = (fake_B - sub_value) * (1.0 / (1 - sub_value))

        # 第3步：HSV伽马校正
        imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(imgHSV)
        S = np.power(S, hsvgamma)
        imgHSV = cv2.merge([H, S, V])
        fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
        fake_B = np.minimum(fake_B, 1.0)

        return fake_B

    def enhance(self, image_path, output_path):
        """增强低光照图像

        流程：预处理 → 模型预测 → 官方后处理 → 保存
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        if self.model is None:
            raise RuntimeError("MBLLEN模型未加载")

        try:
            # 预处理：BGR→RGB，归一化到[0,1]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype("float32") / 255.0
            img_rgb = np.expand_dims(img_rgb, axis=0)

            # 模型预测
            try:
                output = self.model.predict(img_rgb, verbose=0)
            except TypeError:
                # 某些TensorFlow版本不支持verbose参数
                output = self.model.predict(img_rgb)

            # 应用官方后处理
            enhanced = self.postprocess(output)
            enhanced = np.clip(enhanced, 0, 1)
            enhanced = enhanced * 255
            enhanced = enhanced.astype("uint8")
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, enhanced)
            return output_path
        except Exception as e:
            import traceback

            print(f"MBLLEN增强详细错误:\n{traceback.format_exc()}")
            raise RuntimeError(f"MBLLEN增强失败: {e}")

    def enhance_frame(self, img):
        if self.model is None:
            raise RuntimeError("MBLLEN模型未加载")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype("float32") / 255.0
            img_rgb = np.expand_dims(img_rgb, axis=0)
            try:
                output = self.model.predict(img_rgb, verbose=0)
            except TypeError:
                output = self.model.predict(img_rgb)
            enhanced = self.postprocess(output)
            enhanced = np.clip(enhanced, 0, 1)
            enhanced = (enhanced * 255).astype("uint8")
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            return enhanced
        except Exception as e:
            raise RuntimeError(f"MBLLEN帧增强失败: {e}")


class OnnxMBLLENEnhancer:
    """MBLLEN ONNX加速版 - 使用ONNX Runtime推理"""

    _session_cache = {}

    def __init__(self, model_path=None):
        self.session = None
        self.input_name = None
        self.output_name = None

        if model_path is None:
            model_path = os.path.join(WEIGHTS_DIR, "enhancement", "mbllen.onnx")

        if model_path in OnnxMBLLENEnhancer._session_cache:
            self.session = OnnxMBLLENEnhancer._session_cache[model_path]
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            import onnxruntime as ort

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX模型不存在: {model_path}")

            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                session_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )

            OnnxMBLLENEnhancer._session_cache[model_path] = self.session
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def postprocess(
        self, output, lowpercent=5, highpercent=95, maxrange=0.8, hsvgamma=0.8
    ):
        # 检查output的形状
        fake_B = output[0]
        print(f"Output shape: {fake_B.shape}")

        # 根据形状进行适当的转换
        if len(fake_B.shape) == 3:
            if fake_B.shape[0] == 3:
                # 形状为 (3, height, width)，需要转换为 (height, width, 3)
                fake_B = fake_B.transpose(1, 2, 0).copy()
            elif fake_B.shape[2] == 3:
                # 形状已经是 (height, width, 3)，不需要转换
                fake_B = fake_B.copy()

        gray_fake_B = (
            fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 2] * 0.114
        )
        percent_max = np.sum(gray_fake_B >= maxrange) / np.sum(gray_fake_B <= 1.0)
        max_value = np.percentile(gray_fake_B[:], highpercent)
        if percent_max < (100 - highpercent) / 100.0:
            scale = maxrange / max_value
            fake_B = fake_B * scale
            fake_B = np.minimum(fake_B, 1.0)

        gray_fake_B = (
            fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 2] * 0.114
        )
        sub_value = np.percentile(gray_fake_B[:], lowpercent)
        if sub_value < 0.5:
            fake_B = (fake_B - sub_value) * (1.0 / (1 - sub_value))

        imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(imgHSV)
        S = np.power(S, hsvgamma)
        imgHSV = cv2.merge([H, S, V])
        fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
        fake_B = np.minimum(fake_B, 1.0)

        return fake_B

    def enhance(self, image_path, output_path):
        t_start = time.time()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        try:
            # 保存原始大小
            orig_h, orig_w = img.shape[:2]

            # 调整图像大小到模型期望的大小 (400, 600)
            img_resized = cv2.resize(img, (600, 400))

            # 预处理
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype("float32") / 255.0
            img_rgb = np.expand_dims(img_rgb, axis=0)  # 形状变为 (1, 400, 600, 3)

            # 推理
            output = self.session.run([self.output_name], {self.input_name: img_rgb})[0]

            # 后处理
            enhanced = self.postprocess(output)
            enhanced = np.clip(enhanced, 0, 1)
            enhanced = (enhanced * 255).astype("uint8")

            # 调整回原始大小
            enhanced = cv2.resize(enhanced, (orig_w, orig_h))

            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, enhanced)

            infer_time = (time.time() - t_start) * 1000
            return {
                "output_path": output_path,
                "inference_time_ms": round(infer_time, 2),
                "engine": "ONNX Runtime",
            }
        except Exception as e:
            raise RuntimeError(f"ONNX MBLLEN增强失败: {e}")

    def enhance_frame(self, img):
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (600, 400))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        img_rgb = np.expand_dims(img_rgb, axis=0)
        output = self.session.run([self.output_name], {self.input_name: img_rgb})[0]
        enhanced = self.postprocess(output)
        enhanced = np.clip(enhanced, 0, 1)
        enhanced = (enhanced * 255).astype("uint8")
        enhanced = cv2.resize(enhanced, (orig_w, orig_h))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        return enhanced


class ImageEnhancer:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ImageEnhancer._initialized:
            return
        ImageEnhancer._initialized = True
        self._mbllen = None
        self._onnx_mbllen = None
        self._onnx_nafnet_derain = None
        self._onnx_nafnet_dehaze = None
        self._dce = None
        self._ffa_dehaze = None
        self._ffa_dehaze_ots = None

    @property
    def dce(self):
        if self._dce is None:
            dce_path = os.path.join(WEIGHTS_DIR, "enhancement", "zero_dce.onnx")
            if os.path.exists(dce_path):
                self._dce = ZeroDCEEnhancer(dce_path)
                print("Zero-DCE 低光增强模型加载成功!")
        return self._dce

    @property
    def onnx_nafnet_dehaze(self):
        if self._onnx_nafnet_dehaze is None:
            dehaze_path = os.path.join(WEIGHTS_DIR, "enhancement", "nafnet_dehaze.onnx")
            if os.path.exists(dehaze_path):
                self._onnx_nafnet_dehaze = NAFNetONNXEnhancer(dehaze_path)
                print("NAFNet去雾模型加载成功!")
        return self._onnx_nafnet_dehaze

    @property
    def onnx_nafnet_derain(self):
        if self._onnx_nafnet_derain is None:
            derain_path = os.path.join(WEIGHTS_DIR, "enhancement", "nafnet_derain.onnx")
            if os.path.exists(derain_path):
                self._onnx_nafnet_derain = NAFNetONNXEnhancer(derain_path)
                print("NAFNet去雨模型加载成功!")
        return self._onnx_nafnet_derain

    @property
    def onnx_ffa_dehaze(self):
        if self._ffa_dehaze is None:
            ffa_path = os.path.join(WEIGHTS_DIR, "enhancement", "ffa_net_dehaze.onnx")
            if os.path.exists(ffa_path):
                self._ffa_dehaze = FFAONNXEnhancer(ffa_path)
                print("FFA-Net(ITS)去雾模型加载成功!")
        return self._ffa_dehaze

    @property
    def onnx_ffa_dehaze_ots(self):
        if self._ffa_dehaze_ots is None:
            ffa_ots_path = os.path.join(WEIGHTS_DIR, "enhancement", "ffa_net_dehaze_ots.onnx")
            if os.path.exists(ffa_ots_path):
                self._ffa_dehaze_ots = FFAONNXEnhancer(ffa_ots_path)
                print("FFA-Net(OTS)去雾模型加载成功!")
        return self._ffa_dehaze_ots

    @property
    def mbllen(self):
        if self._mbllen is None:
            model_path = get_weights_path("LOL_img_lowlight.h5")
            if os.path.exists(model_path):
                self._mbllen = MBLLENEnhancer(model_path)
                if self._mbllen.model is None:
                    self._mbllen = None
        return self._mbllen

    @property
    def onnx_mbllen(self):
        if self._onnx_mbllen is None:
            onnx_path = os.path.join(WEIGHTS_DIR, "enhancement", "mbllen.onnx")
            if os.path.exists(onnx_path):
                self._onnx_mbllen = OnnxMBLLENEnhancer(onnx_path)
        return self._onnx_mbllen

    def dehaze(self, image_path, output_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = img.copy()
        for i in range(3):
            enhanced[:, :, i] = clahe.apply(img[:, :, i])
        cv2.imwrite(output_path, enhanced)
        return output_path

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
        img = cv2.imread(image_path)
        if img is None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        denoised = cv2.fastNlMeansDenoisingColored(opened, None, 10, 10, 7, 21)
        cv2.imwrite(output_path, denoised)
        return output_path

    def enhance_image(self, image_path, output_path, enhancement_type="dehaze", strength=1.5):
        dispatch = {
            "dehaze": lambda: self.nafnet_dehaze(image_path, output_path),
            "derain": lambda: self.nafnet_derain(image_path, output_path),
            "nafnet_dehaze": lambda: self.nafnet_dehaze(image_path, output_path),
            "nafnet_derain": lambda: self.nafnet_derain(image_path, output_path),
            "ffa_dehaze": lambda: self.ffa_dehaze(image_path, output_path, strength),
            "ffa_dehaze_ots": lambda: self.ffa_dehaze_ots(image_path, output_path, strength),
            "legacy_dehaze": lambda: self.dehaze(image_path, output_path),
            "legacy_derain": lambda: self.derain(image_path, output_path),
            "low_light": lambda: self.enhance_low_light(image_path, output_path),
            "mbllen": lambda: self.mbllen_enhance(image_path, output_path),
            "mbllen_onnx": lambda: self.mbllen_onnx_enhance(image_path, output_path),
            "zero_dce": lambda: self.zero_dce_enhance(image_path, output_path),
        }
        handler = dispatch.get(enhancement_type)
        if handler:
            return handler()
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imwrite(output_path, img)
        return output_path

    def mbllen_enhance(self, image_path, output_path):
        if self.mbllen is None:
            raise RuntimeError("MBLLEN(TensorFlow)模型未初始化，请确保LOL_img_lowlight.h5文件存在")
        print(f"[MBLLEN] 开始增强: {image_path}")
        result = self.mbllen.enhance(image_path, output_path)
        print(f"[MBLLEN] 增强完成: {result}")
        return result

    def mbllen_onnx_enhance(self, image_path, output_path):
        if self.onnx_mbllen is None:
            raise RuntimeError("MBLLEN(ONNX)模型未初始化，请确保mbllen.onnx文件存在")
        print(f"[ONNX MBLLEN] 开始增强: {image_path}")
        result = self.onnx_mbllen.enhance(image_path, output_path)
        print(f"[ONNX MBLLEN] 增强完成: {result}")
        return result

    def zero_dce_enhance(self, image_path, output_path):
        if self.dce is None:
            raise RuntimeError("Zero-DCE模型未初始化")
        print(f"[Zero-DCE] 开始增强: {image_path}")
        result = self.dce.enhance(image_path, output_path)
        print(f"[Zero-DCE] 增强完成: {result}")
        return result

    def nafnet_derain(self, image_path, output_path):
        """使用 NAFNet 模型进行去雨增强"""
        if self.onnx_nafnet_derain is None:
            print("NAFNet去雨模型不可用，回退到传统方法")
            return self.derain(image_path, output_path)
        
        print(f"[NAFNet] 开始去雨增强: {image_path}")
        try:
            result = self.onnx_nafnet_derain.enhance(image_path, output_path)
            print(f"[NAFNet] 去雨增强完成: {result}")
            return result
        except Exception as e:
            print(f"[NAFNet] 去雨失败: {e}，回退到传统方法")
            return self.derain(image_path, output_path)

    def nafnet_dehaze(self, image_path, output_path):
        """使用 NAFNet 模型进行去雾增强，不支持回退"""
        if self.onnx_nafnet_dehaze is None:
            raise RuntimeError("NAFNet去雾模型不可用，请检查 weights/enhancement/nafnet_dehaze.onnx 是否存在")
        
        print(f"[NAFNet] 开始去雾增强: {image_path}")
        result = self.onnx_nafnet_dehaze.enhance(image_path, output_path)
        print(f"[NAFNet] 去雾增强完成: {result}")
        return result

    def ffa_dehaze(self, image_path, output_path, strength=1.5):
        """使用 FFA-Net(ITS) 模型进行去雾增强"""
        if self.onnx_ffa_dehaze is None:
            print("FFA-Net(ITS)去雾模型不可用，回退到传统方法")
            return self.dehaze(image_path, output_path)
        
        print(f"[FFA-Net ITS] 开始去雾增强: {image_path}, 强度: {strength}")
        try:
            result = self.onnx_ffa_dehaze.enhance(image_path, output_path, strength)
            print(f"[FFA-Net ITS] 去雾增强完成: {result}")
            return result
        except Exception as e:
            print(f"[FFA-Net ITS] 去雾失败: {e}，回退到传统方法")
            return self.dehaze(image_path, output_path)

    def ffa_dehaze_ots(self, image_path, output_path, strength=1.5):
        """使用 FFA-Net(OTS) 模型进行去雾增强"""
        if self.onnx_ffa_dehaze_ots is None:
            print("FFA-Net(OTS)去雾模型不可用，回退到传统方法")
            return self.dehaze(image_path, output_path)
        
        print(f"[FFA-Net OTS] 开始去雾增强: {image_path}, 强度: {strength}")
        try:
            result = self.onnx_ffa_dehaze_ots.enhance(image_path, output_path, strength)
            print(f"[FFA-Net OTS] 去雾增强完成: {result}")
            return result
        except Exception as e:
            print(f"[FFA-Net OTS] 去雾失败: {e}，回退到传统方法")
            return self.dehaze(image_path, output_path)

    def ffa_dehaze_frame(self, img, strength=1.5):
        """FFA-Net(ITS) 帧级去雾，输入输出均为 BGR numpy 数组"""
        if self.onnx_ffa_dehaze is None:
            return self._clahe_dehaze_frame(img)
        try:
            return self.onnx_ffa_dehaze.enhance_frame(img, strength)
        except Exception as e:
            print(f"[FFA-Net ITS] 帧去雾失败: {e}，回退到传统方法")
            return self._clahe_dehaze_frame(img)

    def ffa_dehaze_ots_frame(self, img, strength=1.5):
        """FFA-Net(OTS) 帧级去雾，输入输出均为 BGR numpy 数组"""
        if self.onnx_ffa_dehaze_ots is None:
            return self._clahe_dehaze_frame(img)
        try:
            return self.onnx_ffa_dehaze_ots.enhance_frame(img, strength)
        except Exception as e:
            print(f"[FFA-Net OTS] 帧去雾失败: {e}，回退到传统方法")
            return self._clahe_dehaze_frame(img)

    def nafnet_dehaze_frame(self, img):
        """NAFNet 帧级去雾，输入输出均为 BGR numpy 数组，不支持回退"""
        if self.onnx_nafnet_dehaze is None:
            raise RuntimeError("NAFNet去雾模型不可用，请检查 weights/enhancement/nafnet_dehaze.onnx 是否存在")
        return self.onnx_nafnet_dehaze.enhance_frame(img)

    def nafnet_derain_frame(self, img):
        """NAFNet 帧级去雨，输入输出均为 BGR numpy 数组"""
        if self.onnx_nafnet_derain is None:
            return self._morph_derain_frame(img)
        try:
            return self.onnx_nafnet_derain.enhance_frame(img)
        except Exception as e:
            print(f"[NAFNet] 帧去雨失败: {e}，回退到传统方法")
            return self._morph_derain_frame(img)

    @staticmethod
    def _clahe_dehaze_frame(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = img.copy()
        for i in range(3):
            enhanced[:, :, i] = clahe.apply(img[:, :, i])
        return enhanced

    @staticmethod
    def _morph_derain_frame(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return cv2.fastNlMeansDenoisingColored(opened, None, 10, 10, 7, 21)


class ZeroDCEEnhancer:
    _session_cache = {}

    def __init__(self, model_path):
        if model_path in ZeroDCEEnhancer._session_cache:
            self.session = ZeroDCEEnhancer._session_cache[model_path]
            self.input_name = self.session.get_inputs()[0].name
        else:
            import onnxruntime as ort

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Zero-DCE ONNX 模型不存在: {model_path}")

            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

            ZeroDCEEnhancer._session_cache[model_path] = self.session
            self.input_name = self.session.get_inputs()[0].name

    def enhance_frame(self, img):
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)
        img_input = np.expand_dims(img_chw, axis=0)
        output = self.session.run(None, {self.input_name: img_input})[0]
        enhanced = output[0].transpose(1, 2, 0)
        enhanced = np.clip(enhanced, 0, 1)
        enhanced = (enhanced * 255).astype("uint8")
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        if enhanced.shape[:2] != (orig_h, orig_w):
            enhanced = cv2.resize(enhanced, (orig_w, orig_h))
        return enhanced

    def enhance(self, image_path, output_path):
        t_start = time.time()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        enhanced = self.enhance_frame(img)
        cv2.imwrite(output_path, enhanced)
        infer_time = (time.time() - t_start) * 1000
        return {
            "output_path": output_path,
            "inference_time_ms": round(infer_time, 2),
            "engine": "Zero-DCE ONNX Runtime",
        }


class NAFNetONNXEnhancer:
    """NAFNet ONNX 增强器 - 使用 ONNX Runtime 推理"""

    _session_cache = {}

    def __init__(self, model_path):
        self.session = None
        self.input_name = None
        self.output_name = None

        if model_path in NAFNetONNXEnhancer._session_cache:
            self.session = NAFNetONNXEnhancer._session_cache[model_path]
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            import onnxruntime as ort

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"NAFNet ONNX 模型不存在: {model_path}")

            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )

            NAFNetONNXEnhancer._session_cache[model_path] = self.session
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img):
        orig_h, orig_w = img.shape[:2]
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)
        img_input = np.expand_dims(img_chw, axis=0)
        return img_input, orig_h, orig_w, pad_h, pad_w

    def _postprocess(self, output, orig_h, orig_w, pad_h, pad_w):
        enhanced = output[0].transpose(1, 2, 0)
        enhanced = np.clip(enhanced, 0, 1)
        enhanced = (enhanced * 255).astype("uint8")
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        if pad_h > 0 or pad_w > 0:
            enhanced = enhanced[:orig_h, :orig_w]
        if enhanced.shape[:2] != (orig_h, orig_w):
            enhanced = cv2.resize(enhanced, (orig_w, orig_h))
        return enhanced

    def enhance_frame(self, img):
        """直接处理 BGR numpy 数组，返回增强后的 BGR 数组"""
        img_input, orig_h, orig_w, pad_h, pad_w = self._preprocess(img)
        output = self.session.run([self.output_name], {self.input_name: img_input})[0]
        return self._postprocess(output, orig_h, orig_w, pad_h, pad_w)

    def enhance(self, image_path, output_path):
        t_start = time.time()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        try:
            enhanced = self.enhance_frame(img)
            cv2.imwrite(output_path, enhanced)

            infer_time = (time.time() - t_start) * 1000
            return {
                "output_path": output_path,
                "inference_time_ms": round(infer_time, 2),
                "engine": "NAFNet ONNX Runtime",
            }
        except Exception as e:
            raise RuntimeError(f"NAFNet ONNX 增强失败: {e}")


class FFAONNXEnhancer:
    """FFA-Net ONNX 增强器 - 用于图像去雾"""

    _session_cache = {}

    def __init__(self, model_path):
        self.session = None
        self.input_name = None
        self.output_name = None

        if model_path in FFAONNXEnhancer._session_cache:
            self.session = FFAONNXEnhancer._session_cache[model_path]
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        else:
            import onnxruntime as ort

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"FFA-Net ONNX 模型不存在: {model_path}")

            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.intra_op_num_threads = 4
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )

            FFAONNXEnhancer._session_cache[model_path] = self.session
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, img):
        orig_h, orig_w = img.shape[:2]
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype("float32") / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)
        img_input = np.expand_dims(img_chw, axis=0)
        return img_input, orig_h, orig_w, pad_h, pad_w

    def _postprocess(self, output, input_tensor, orig_h, orig_w, pad_h, pad_w, strength=1.0):
        output_chw = output[0]
        input_chw = input_tensor[0]
        residual = output_chw - input_chw
        enhanced_chw = input_chw + strength * residual
        enhanced = enhanced_chw.transpose(1, 2, 0)
        enhanced = np.clip(enhanced, 0, 1)
        enhanced = (enhanced * 255).astype("uint8")
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        if pad_h > 0 or pad_w > 0:
            enhanced = enhanced[:orig_h, :orig_w]
        if enhanced.shape[:2] != (orig_h, orig_w):
            enhanced = cv2.resize(enhanced, (orig_w, orig_h))
        return enhanced

    @staticmethod
    def _apply_post_enhance(img, gamma=0.8, clahe_clip=2.0):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        return enhanced

    def enhance_frame(self, img, strength=1.5):
        """直接处理 BGR numpy 数组，返回增强后的 BGR 数组"""
        img_input, orig_h, orig_w, pad_h, pad_w = self._preprocess(img)
        output = self.session.run([self.output_name], {self.input_name: img_input})[0]
        enhanced = self._postprocess(output, img_input, orig_h, orig_w, pad_h, pad_w, strength)
        enhanced = self._apply_post_enhance(enhanced, gamma=0.8, clahe_clip=2.0)
        return enhanced

    def enhance(self, image_path, output_path, strength=1.5):
        t_start = time.time()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        try:
            enhanced = self.enhance_frame(img, strength)
            cv2.imwrite(output_path, enhanced)
            infer_time = (time.time() - t_start) * 1000
            return {
                "output_path": output_path,
                "inference_time_ms": round(infer_time, 2),
                "engine": "FFA-Net ONNX Runtime",
                "strength": strength,
            }
        except Exception as e:
            raise RuntimeError(f"FFA-Net ONNX 增强失败: {e}")
