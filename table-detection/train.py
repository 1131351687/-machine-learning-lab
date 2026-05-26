from tabnanny import verbose

from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import random
import shutil
import yaml

SRC_ROOT=Path('datasets') # 原始数据集目录
DST_ROOT=Path('datasets_subset') # 处理的数据集
TRAIN_RATIO=0.2
VAL_RATIO=0.3
seed=42 # 随机种子

NAMES=['book', 'bottole', 'earphone', 'glass', 'headphone',
         'keyboard', 'laptop', 'mobile', 'mouse', 'pen', 'penstand']
NC=len(NAMES)

def sample_split(split:str,ratio:float,seed:int):
    """
    随机抽样图片，复制图片和标签到新目录
    Args:
        split: 数据集划分，'train'、'val'或'test'
        ratio: 抽样比例，0-1之间
        seed: 随机种子，确保可重复抽样
    """
    random.seed(seed)
    src_img_dir=SRC_ROOT/split/'images'     # 原始图像目录
    src_label_dir=SRC_ROOT/split/'labels'

    dst_img_dir=DST_ROOT/split/'images'     # 目标图像目录
    dst_label_dir=DST_ROOT/split/'labels'

    dst_img_dir.mkdir(parents=True, exist_ok=True)      # 创建目标图像目录
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    images=list(src_img_dir.glob('*.*'))     # 获取所有图像文件
    if not images:
        print(f'{split} 目录下没有找到图像文件')
        return 0,0
    k=max(1,int(len(images)*ratio))     # 计算抽样数量，至少抽取1张
    sampled=random.sample(images,k)     # 随机抽样图像文件

    copied=0
    for img_path in sampled:
        lbl_path=src_label_dir /f'{img_path.stem}.txt'     # 对应的标签文件路径
        shutil.copy2(img_path,dst_img_dir/img_path.name)     # 复制图像文件
        if lbl_path.exists():
            shutil.copy2(lbl_path,dst_label_dir/lbl_path.name)     # 复制标签文件
        copied+=1

    print(f'[{split}]抽样{copied}/{len(images)}')
    return copied,len(images)

def build_subset_dataset():
    """
    构建小数据集目录，并生成 data.yaml。
    """
    data_yaml = {
        "path": str(DST_ROOT.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "nc": NC,
        "names": NAMES,
    }

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    with open(DST_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"已生成小数据集：{DST_ROOT}")




if __name__=="__main__":
    if DST_ROOT.exists():
        print(f"删除旧数据集：{DST_ROOT}")
        shutil.rmtree(DST_ROOT)  # 递归删除整个文件夹
    sample_split("train", TRAIN_RATIO, seed)
    sample_split("valid", VAL_RATIO, seed)
    build_subset_dataset()
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(DST_ROOT / "data.yaml"),
        epochs=50,
        imgsz=640,       # 输入图像大小
        verbose=False,       # 训练过程不输出详细日志
    )

