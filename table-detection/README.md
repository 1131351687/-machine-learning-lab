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

# Label Studio

数据标注工具，用于自制数据集或补充标注。

需在虚拟环境中安装：

```bash
pip install label-studio
```

新建脚本文件让 Label Studio 能访问本地图片，二选一：

**① start_label_studio.bat**

```bat
@echo off
call conda activate yolo_tutorial
set LOCAL_FILES_SERVING_ENABLED=true
label-studio
```

**② start_label_studio.ps1（右键PowerShell启动）**

```powershell
conda activate yolo_tutorial
$env:LOCAL_FILES_SERVING_ENABLED="true"
label-studio
```

将以上内容分别保存为对应文件，启动时双击或运行：

```cmd
:: CMD
start_label_studio.bat
```

```powershell
# PowerShell
.\start_label_studio.ps1
```

---

# Project Structure

```
table-detection/
├── train.py                # 训练脚本（含数据集抽样）
├── test_train.py           # 学习率对比实验（踩坑点见下方）
├── predict.py              # 推理脚本
├── check.py                # 数据集分布检查
├── colab.ipynb             # Colab 版本（与 .py 对应）
├── start_label_studio.ps1  # Label Studio 启动脚本
└── datasets/               # 数据集（Roboflow YOLO 格式）
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

## 1. lr0 未生效 — optimizer="auto" 的陷阱

做学习率对比实验时，设置了 `lr0=0.001, 0.005, 0.01, 0.02`，但 loss、mAP、precision 结果完全一致，相当于只跑了一次。

**原因**：YOLO 默认 `optimizer="auto"` 会自行选择优化器并覆盖手动设置的 `lr0`。训练日志可见：

```text
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.005'
optimizer: AdamW(lr=0.000667, momentum=0.9)
```

实际生效的是 `AdamW(lr=0.000667)`，手动传入的 `lr0` 参数被静默忽略。

**解决**：显式指定优化器：

```python
model.train(optimizer="SGD", lr0=0.01)    # SGD 生效
model.train(optimizer="AdamW", lr0=0.001) # AdamW 生效
```

详见 [test_train.py](test_train.py)（踩坑版）与 [train.py](train.py)（正确版）。

---

## 2. 本地部署容易炸显存

本地 GPU 显存有限（尤其是笔记本），训练时很容易 OOM（out of memory）。

**解决方式**：

- **降低 batch**：把 `batch=16` 降到 `batch=8` 或 `batch=4`，显存占用直线下降
- **减小 imgsz**：`imgsz=640` 降到 `imgsz=320`
- **换 Colab**：免费 GPU（T4/K100）足够跑 YOLO，完全不存在显存问题

### Colab 部署要注意改路径

本地脚本和 Colab 笔记本的文件结构不同，直接跑会找不到数据集。需要改两处：

**① 数据路径**：从相对路径改为 Google Drive 绝对路径

```python
# 本地（相对路径）
SRC_ROOT = Path('datasets')

# Colab（挂载 Drive 后的绝对路径），此处为google云端硬盘上传后的路径，具体根据实际情况修改
SRC_ROOT = Path('/content/drive/MyDrive/datasets')
```

**② data.yaml**：必须重新生成，因为里面写死了 `path` 字段

Colab 中手动指定 yaml 内容：

```python
data_yaml = {
    "path": str(DST_ROOT),            # 指向 Drive 上的路径
    "train": "train/images",
    "val": "valid/images",
    "nc": 11,
    "names": ['book', 'bottole', 'earphone', 'glass', 'headphone',
              'keyboard', 'laptop', 'mobile', 'mouse', 'pen', 'penstand']
}

with open(DST_ROOT / "data.yaml", 'w') as f:
    yaml.dump(data_yaml, f)
```

> 详细对比见 [colab.ipynb](colab.ipynb) 中的写法与 [train.py](train.py) 的区别。


---

# TODO

- [ ] 增加数据集
- [ ] 完成 ONNX 部署
- [ ] TensorRT 推理
- [ ] 尝试 YOLO Seg
- [ ] 实时视频检测优化