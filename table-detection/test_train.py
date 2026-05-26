from tabnanny import verbose

from ultralytics import YOLO
from pathlib import Path
import pandas as pd

DST_ROOT=Path('datasets_subset') # 处理的数据集


if __name__=="__main__":
    # 不同学习率
    learning_rates=[0.001,0.005,0.01,0.02,0.05]
    results_log = []

    for lr in learning_rates:
        model = YOLO("yolo26n.pt")
        model.train(
            data=str(DST_ROOT / "data.yaml"),
            epochs=20,
            imgsz=640,       # 输入图像大小
            batch=16,        # 批量大小
            lr0=lr,         # 初始学习率
            verbose=False,       # 训练过程不输出详细日志
            name=f"exp_lr_{lr}", # 保存结果的文件夹名称
            exist_ok=True       # 如果文件夹已存在，允许覆盖
        )

        val_results=model.val(verbose=False)
        results_log.append({
            '实验':f'lr={lr}',
            'lr0': lr,
            'epochs': 50,
            'imgsz': 640,
            'batch': 16,
            'mAP50':val_results.box.map50,   # mAP50是一个常用的性能指标，表示在IoU阈值为0.5时的平均精度
            'mAP50-95':val_results.box.map, # mAP50-95是另一个常用的性能指标，表示在IoU阈值从0.5到0.95之间的平均精度
            'P':val_results.box.p[0] if len(val_results.box.p)>0 else 0,       # 精确率
            'R':val_results.box.r[0] if len(val_results.box.r)>0 else 0,       # 召回率
        })
        print(f"结果：mAP50={val_results.box.map50:.3f},mAP50-95={val_results.box.map:.3f}")
    df=pd.DataFrame(results_log)
    df.to_csv('exp_result_lr.csv',index=False)  # 将结果保存到CSV文件
    print("实验结果已保存到 exp_result_lr.csv")
    print(df[['实验','mAP50','mAP50-95']])

