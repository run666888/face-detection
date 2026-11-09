# Face Detection · 实时多人脸检测与识别

基于 OpenCV YuNet 的实时多人脸检测，以及基于 InsightFace 的人脸注册/识别（本地 `faces_db.json`）。

## 功能

- 摄像头实时检测多张人脸，输出人脸框、置信度和 5 个关键点
- 按键保存截图；可调置信度阈值、无窗口测试模式
- `register.py` 注册人脸，`face_recognizer.py` 实时识别

## 运行

```bash
pip install opencv-python insightface numpy
python detect_faces.py
```

模型需自行下载后放到 `models/`（本仓库不包含任何模型与任何人脸数据）：

- YuNet：OpenCV Zoo `face_detection_yunet_2023mar.onnx`
- InsightFace：`buffalo_s`（**仅限非商业研究用途，遵循 InsightFace 官方许可**）

## 隐私与合规

- 仓库不包含 `faces_db.json`、截图或任何真实人脸数据，请勿把采集到的人脸数据上传到公开仓库。
- 人脸信息属于敏感个人信息，用于他人前应取得明确同意，并遵守当地法律（如《个人信息保护法》）。
- InsightFace 预训练模型为非商业许可；商用请替换为获得商业授权的模型。

## License

代码 MIT；第三方模型与库遵循各自许可。
