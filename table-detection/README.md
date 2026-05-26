# Table Detection

一个基于 YOLO 的桌面物体检测练手项目。

目前支持检测 11 类桌面物体：

- book, bottle, earphone, glass, headphone
- keyboard, laptop, mobile, mouse, pen, penstand

主要用于学习：

- 数据集制作
- YOLO 训练流程
- 数据增强
- 调参
- ONNX 导出
- 实时推理

---

# Environment

```bash
python 3.10
pytorch 2.x
ultralytics
opencv-python
```

安装：

```bash
pip install ultralytics opencv-python
```

---

# Dataset

数据集来源于 Roboflow：

https://universe.roboflow.com/celebalworkspace-bqx5k/table-03wsy/dataset/1

流程：

1. 从 Roboflow 下载数据集
2. 选择 YOLO 格式

当前数据量：

- train: 1142 images
- val: 148 images

---

# Project Structure

```
table-detection/
├── train.py          # 训练脚本（含数据集抽样）
├── test_train.py     # 学习率对比实验（踩坑点见下方）
├── predict.py        # 推理脚本
├── check.py          # 数据集分布检查
├── colab.ipynb       # Colab 版本（与 .py 对应）
└── datasets/         # 数据集（Roboflow YOLO 格式）
```

---

# Train

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.01,
    mosaic=1.0,
    fliplr=0.5
)
```

---

# Predict

图片推理：

```python
model.predict("test.jpg")
```

摄像头实时检测：

```python
model.predict(source=0)
```

---

# Export ONNX

```python
model.export(format="onnx")
```

---

# Some Notes / Problems

## 1. lr0 没生效 — optimizer="auto" 的陷阱

### 现象

做学习率对比实验时，`lr0=0.001, 0.005, 0.01, 0.02` 的结果完全一样：
- loss 一模一样
- mAP 一模一样
- precision 一模一样

本质上其实是同一次训练。

### 原因

YOLO 新版本默认 `optimizer="auto"`，它会自动选择优化器并**覆盖**你手动设置的 `lr0`。

关键日志证据：

```text
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.005'
optimizer: AdamW(lr=0.000667, momentum=0.9)
```

可见实际生效的是 `AdamW(lr=0.000667)`，你设的 `lr0` 全被忽略了。

### 解决

必须**显式指定优化器**，不能依赖 auto：

```python
# 方法1: SGD + 手动学习率
model.train(optimizer="SGD", lr0=0.01)

# 方法2: AdamW + 手动学习率
model.train(optimizer="AdamW", lr0=0.001)
```

详见 [test_train.py](table-detection/test_train.py) 中的实验代码（踩坑版）与 [train.py](table-detection/train.py)（正确版）。

---

## 2. Mosaic 会影响收敛

开启 Mosaic 后：

- 小目标效果更好
- 泛化更强

但：

- loss 波动更大
- 后期收敛变慢

YOLO 默认会在最后几个 epoch 自动关闭 Mosaic：

```text
Closing dataloader mosaic
```

---

## 3. 数据量太小

目前数据集只有 200+ 图片。

问题：

- 容易过拟合
- 泛化一般
- 实际摄像头效果不稳定

后续准备继续扩充：

- 不同光照
- 不同角度
- 遮挡
- 夜间环境

---

## 4. mAP 不代表真实效果

有时候：

- val mAP 看起来不错

但：

- 实时摄像头漏检严重

所以后面会更关注：

- 实际推理效果
- 漏检
- 误检
- FPS

---

# TODO

- [ ] 增加数据集
- [ ] 完成 ONNX 部署
- [ ] TensorRT 推理
- [ ] 尝试 YOLO Seg
- [ ] 实时视频检测优化