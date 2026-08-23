# 视觉分拣监控矩阵

运行：

```powershell
E:\robot_software\envs\robotgame\python.exe D:\robot_projects\robotgame\vision_sorting\main.py
```

主要文件：

## 环境与模型

```powershell
pip install -r requirements.txt
python main.py
```

运行识别前，需要在界面选择兼容 Ultralytics YOLO 的 `.pt` 权重，并准备对应类别名称、手眼标定矩阵和 `feature_data` 配置。仓库不包含训练权重、相机标定数据或现场机器人配置。需要 Intel RealSense D435；在线抓取还需要可访问的 DOBOT TCP 接口。

- `main.py`：程序入口
- `main_window.py`：主界面与抓取流程
- `camera.py`：RealSense 彩色与深度双画面
- `recognition.py`：YOLO 检测与轮廓中心提取
- `robot_online.py`：机器人在线模式 TCP 客户端
- `calibration_utils.py`：手眼标定矩阵加载与坐标转换
- `feature_dialog.py`：特征示教与掩膜编辑器

特征配置保存为：

`D:\robot_projects\robotgame\vision_sorting\feature_data\config.json`

同目录下会生成 `template.png` 和 `mask.png`。
