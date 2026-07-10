# 基于图像增强的恶劣天气车辆感知系统

## 项目介绍

本毕设项目实现了基于图像增强的恶劣天气车辆感知系统，支持图片识别、视频识别和摄像头实时识别。系统通过多种图像增强技术提升恶劣天气条件下的车辆检测准确性，为智能交通系统提供可靠的车辆感知能力。

## 技术栈

- **前端**: HTML5 + CSS3 + JavaScript
- **后端**: Python Flask
- **数据库**: MySQL 8.0
- **图像增强**: 
  - OpenCV (去雾/去雨/低光照增强)
  - MBLLEN (深度学习低光照增强)
  - ONNX Runtime (模型推理加速)
- **车辆检测**: YOLOv8 (Ultralytics)

## 功能特性

1. **图像增强**: 支持去雾、去雨、低光照增强，包含传统算法和深度学习模型
2. **车辆检测**: 识别轿车、卡车、公交车、摩托车、自行车等多种车型
3. **图片识别**: 上传图片进行识别，支持多种增强类型
4. **视频识别**: 上传视频进行批量帧识别，可设置检测间隔
5. **摄像头识别**: 使用电脑摄像头实时捕捉画面并进行车辆检测
6. **历史记录**: 查看所有历史检测记录，支持分页和删除操作
7. **测试结果**: 查看多源数据测试结果与分析报告
8. **主题切换**: 支持白色、灰色、黑色三种主题
9. **ONNX加速**: 支持MBLLEN和YOLOv8模型的ONNX加速，提升推理性能

## 项目结构

```
bishe/
├── backend/                      # Python后端
│   ├── app.py                   # Flask主应用
│   ├── config.py                # 配置文件
│   ├── requirements.txt         # 依赖
│   ├── routes/                  # 路由
│   ├── services/                # 业务逻辑
│   ├── utils/                   # 工具类
│   └── weights/                 # 模型权重
├── frontend/                    # HTML前端
│   ├── index.html               # 主页
│   ├── pages/                   # 子页面
│   └── css/                     # 样式
├── output/                      # 输出结果
│   └── benchmark/               # 测试结果
└── weights/                     # 模型权重
    └── test_data/               # 测试数据
```

## 环境配置

### 1. 安装依赖

使用uv管理Python环境与依赖：

```bash
uv venv
.venv\Scripts\activate
cd backend
uv pip install -r requirements.txt
```

### 2. MySQL配置

确保MySQL正在运行，并创建数据库：

```sql
CREATE DATABASE bishe_vehicle;
```

### 3. 配置数据库连接

在项目根目录创建 `.env` 文件，设置数据库环境变量：

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=bishe_vehicle
```

> 配置机密信息请通过环境变量设置，不要硬编码在代码中。

## 启动项目

### 方式1: 使用启动脚本

```bash
cd backend
run.bat
```

### 方式2: 手动启动

```bash
cd backend
python app.py
```

## 使用说明

1. 启动后端服务后，在浏览器打开 `http://localhost:5000`
2. 首页可以进入各个功能模块:
   - **图片识别**: 上传图片，选择增强类型进行识别
   - **视频识别**: 上传视频进行批量帧识别，可设置检测间隔
   - **摄像头识别**: 开启摄像头拍照识别，支持实时预览
   - **历史记录**: 查看所有历史检测数据，支持分页和删除操作
   - **测试结果**: 查看多源数据测试结果与分析报告

## 注意事项

- 需要安装MySQL并创建数据库
- 首次运行时会自动下载YOLOv8模型
- 确保有稳定的网络连接以下载模型
- 摄像头功能需要浏览器授权摄像头权限
- 视频处理可能需要较长时间，取决于视频长度和计算机性能
- 对于低光照增强，系统会优先使用ONNX加速版本的MBLLEN模型以提高性能

## 性能优化

- 使用ONNX Runtime加速模型推理
- 对视频处理使用帧间隔采样，减少处理量
- 对大图片进行resize处理，减少计算量
- 模型权重缓存，避免重复加载

## 测试数据

项目提供了测试数据，位于 `weights/test_data/` 目录，包含多种场景下的车辆图片，可用于测试系统性能。

## ONNX模型转换

### MBLLEN模型转ONNX

项目已包含转换好的MBLLEN ONNX模型 `weights/mbllen.onnx`。如需重新转换，可使用以下命令：

```bash
cd backend
python services/onnx_converter.py --mbllen
```

### YOLOv8模型转ONNX

项目已包含转换好的YOLOv8 ONNX模型。如需重新转换，可使用以下命令：

```bash
cd backend
python services/onnx_converter.py --yolo yolov8x
```

## 故障排除

### 数据库连接失败

1. 确保MySQL服务正在运行
2. 检查 `backend/config.py` 中的数据库配置是否正确
3. 确保数据库 `bishe_vehicle` 已创建

### ONNX Runtime导入失败

1. 确保已安装onnxruntime库：`pip install onnxruntime`
2. 检查Python版本是否与onnxruntime兼容

### 模型加载失败

1. 确保模型文件存在于 `weights/` 目录
2. 检查模型文件是否损坏
3. 尝试重新下载模型

## 致谢

- YOLOv8: https://github.com/ultralytics/ultralytics
- MBLLEN: https://github.com/ZZUTK/MBLLEN
- ONNX Runtime: https://github.com/microsoft/onnxruntime
