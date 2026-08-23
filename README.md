# 视觉分拣监控矩阵

运行：

```powershell
E:\robot_software\envs\robotgame\python.exe D:\robot_projects\robotgame\vision_sorting\main.py
```

主要文件：

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
