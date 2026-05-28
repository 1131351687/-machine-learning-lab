from tabnanny import verbose

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import sys

DST_ROOT=Path('datasets_subset') # 处理的数据集

# 定义不同的实验配置
EXPERIMENTS = {
    'lr': {  # 学习率实验
        'param_name': 'lr0',
        'param_values': [0.001, 0.005, 0.01, 0.02, 0.05],
        'fixed_params': {
            'optimizer': "AdamW",
            'epochs': 20,
            'imgsz': 640,
            'batch': 16
        }
    },
    'optimizer': {  # 优化器实验
        'param_name': 'optimizer',
        'param_values': ['AdamW', 'SGD', 'Adam', 'RMSprop'],
        'fixed_params': {
            'lr0': 0.01,
            'epochs': 20,
            'imgsz': 640,
            'batch': 16
        }
    },
    'epochs': {  # 训练轮数实验
        'param_name': 'epochs',
        'param_values': [10, 20, 50, 100],
        'fixed_params': {
            'optimizer': "AdamW",
            'lr0': 0.01,
            'imgsz': 640,
            'batch': 16
        }
    },
    'batch': {  # 批量大小实验
        'param_name': 'batch',
        'param_values': [8, 16, 32, 64],
        'fixed_params': {
            'optimizer': "AdamW",
            'lr0': 0.01,
            'epochs': 20,
            'imgsz': 640
        }
    },
    'weight_decay': {  # 权重衰减实验
        'param_name': 'weight_decay',
        'param_values': [0.0001, 0.0005, 0.001, 0.005],
        'fixed_params': {
            'optimizer': "AdamW",
            'lr0': 0.01,
            'epochs': 20,
            'imgsz': 640,
            'batch': 16
        }
    }
}

def train_with_params(param_name,param_values,fixed_params,experiment_name):
    """通用训练函数"""
    result_log=[]
    for value in param_values:
        print(f"\n开始训练:{param_name}={value}")
        model=YOLO("yolo26n.pt")
        train_params={
            'data':str(DST_ROOT / "data.yaml"),
            'verbose':False,
            'name':f"exp_{experiment_name}_{param_name}_{value}",
            'exist_ok':True,
            **fixed_params      # 解包固定参数，**fixed_params会将字典中的键值对作为关键字参数传递给函数
        }

        train_params[param_name]=value  # 设置当前实验参数
        model.train(**train_params)     # 使用解包的参数进行训练
        val_results=model.val(verbose=False)     # 验证模型性能
        result_entry={
            '实验':f'{param_name}={value}',
            **fixed_params,
            'mAP50':val_results.box.map50,
            'mAP50-95':val_results.box.map,
            'P':val_results.box.p[0] if len(val_results.box.p)>0 else 0,
            'R':val_results.box.r[0] if len(val_results.box.r)>0 else 0,
        }
        result_log.append(result_entry)
        print(f"结果：mAP50={val_results.box.map50:.3f},mAP50-95={val_results.box.map:.3f}")

    df=pd.DataFrame(result_log)
    output_file=f'exp_result_{experiment_name}.csv'
    df.to_csv(output_file,index=False)
    print(f"实验结果已保存到 {output_file}")
    print(df[['实验','mAP50','mAP50-95']])

    return df

if __name__=="__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_TO_RUN = sys.argv[1]  # 从命令行获取参数
        # 可用的参数（实验名称）
        # lr           学习率实验
        # optimizer    优化器实验
        # epochs       训练轮数实验
        # batch        批量大小实验
        # weight_decay  权重衰减实验
    else:
        EXPERIMENT_TO_RUN = 'lr'  # 默认值
    exp_confit=EXPERIMENTS[EXPERIMENT_TO_RUN]
    results_df=train_with_params(
        param_name=exp_confit['param_name'],
        param_values=exp_confit['param_values'],
        fixed_params=exp_confit['fixed_params'],
        experiment_name=EXPERIMENT_TO_RUN
    )




