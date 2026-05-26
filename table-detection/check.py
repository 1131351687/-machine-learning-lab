# 1. 检查各类别数据分布
from collections import Counter
from pathlib import Path

NAMES=['book', 'bottole', 'earphone', 'glass', 'headphone',
         'keyboard', 'laptop', 'mobile', 'mouse', 'pen', 'penstand']


def analyze_dataset(data_root):
    """分析数据集各类别数量"""
    label_dir = Path(data_root) / 'train' / 'labels'
    class_counts = Counter()

    for label_file in label_dir.glob('*.txt'):
        with open(label_file) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    print("各类别样本数量：")
    for i in range(11):  # 你的11个类别
        print(f"  {NAMES[i]}: {class_counts.get(i, 0)} 个")
    return class_counts

# 运行分析
class_counts = analyze_dataset('datasets_subset')