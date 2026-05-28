from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import multiprocessing
import shutil


DST_ROOT = Path('datasets_subset')  # 处理的数据集
RESULT_ROOT = Path('图像增强实验')  # 实验结果目录
PROJECT_ROOT = (Path('runs') / 'detect').resolve()  # 训练输出目录（绝对路径，避免重复拼接）

experiments = [
    # baseline
    {
        'name': 'baseline',
        'params': {
            'augment': False,
            'mosaic': 0.0,
            'fliplr': 0.0,
        }
    },
    # mosaic
    {
        'name': 'mosaic',
        'params': {
            'mosaic': 1.0,
        }
    },
    # hsv weak
    {
        'name': 'hsv_weak',
        'params': {
            'hsv_h': 0.01,
            'hsv_s': 0.02,
            'hsv_v': 0.01,
        }
    },
    # hsv mid
    {
        'name': 'hsv_mid',
        'params': {
            'hsv_h': 0.015,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
        }
    },
    # hsv strong
    {
        'name': 'hsv_strong',
        'params': {
            'hsv_h': 0.03,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
        }
    },
    # flip
    {
        'name': 'flip',
        'params': {
            'fliplr': 0.5,
        }
    }
]

all_results = []
_ALLOWED_AUG_KEYS = {'augment', 'mosaic', 'hsv_h', 'hsv_s', 'hsv_v', 'fliplr', 'flipud', 'auto_augment'}


def _summary_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    last = df.iloc[-1]
    return {
        'Precision': last.get('metrics/precision(B)', None),
        'Recall': last.get('metrics/recall(B)', None),
        'mAP50': last.get('metrics/mAP50(B)', None),
        'mAP50-95': last.get('metrics/mAP50-95(B)', None),
        'Train Box Loss': last.get('train/box_loss', None),
        'Val Box Loss': last.get('val/box_loss', None),
    }


if __name__ == '__main__':
    multiprocessing.freeze_support()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        exp_name = exp['name']
        run_dir = PROJECT_ROOT / ('exp_' + exp_name)
        csv_path = run_dir / 'results.csv'
        save_csv_path = RESULT_ROOT / (exp_name + '_results.csv')
        done_marker = RESULT_ROOT / (exp_name + '_done.txt')
        last_ckpt = run_dir / 'weights' / 'last.pt'

        # 1) 已经完整保存过：直接跳过
        if done_marker.exists() and save_csv_path.exists() and save_csv_path.stat().st_size > 0:
            print('\n========== Skip: %s (已有结果) ==========' % exp_name)
            summary = _summary_from_csv(save_csv_path)
            all_results.append((exp_name, summary))
            continue

        # 2) 有 last.pt：从断点恢复
        if last_ckpt.exists():
            print('\n========== Resume: %s (从断点继续) ==========' % exp_name)
            model = YOLO(str(last_ckpt))
            train_kwargs = {
                'data': str(DST_ROOT / 'data.yaml'),
                'epochs': 20,
                'imgsz': 640,
                'batch': 16,
                'workers': 0,
                'seed': 42,
                'lr0': 0.001,
                'optimizer': 'AdamW',
                'verbose': False,
                'project': str(PROJECT_ROOT),
                'name': 'exp_' + exp_name,
                'exist_ok': True,
                'resume': True,
            }
        else:
            print('\n========== Running: %s ==========' % exp_name)
            model = YOLO('yolo26n.pt')
            train_kwargs = {
                'data': str(DST_ROOT / 'data.yaml'),
                'epochs': 20,
                'imgsz': 640,
                'batch': 16,
                'workers': 0,
                'seed': 42,
                'lr0': 0.001,
                'optimizer': 'AdamW',
                'verbose': False,
                'project': str(PROJECT_ROOT),
                'name': 'exp_' + exp_name,
                'exist_ok': True,
            }

        # 只把 YOLO 支持的增强参数传进去，避免无效参数报错
        for k, v in exp.get('params', {}).items():
            if k in _ALLOWED_AUG_KEYS:
                train_kwargs[k] = v
            else:
                print("警告: 参数 '%s' 不是有效的 YOLO 训练参数，已忽略。" % k)

        train_result = model.train(**train_kwargs)

        # 使用 Ultralytics 返回的真实保存目录，避免 exp_xxx-2 或双层 runs/detect 导致路径错读
        run_dir = Path(str(train_result.save_dir))
        csv_path = run_dir / 'results.csv'

        # 3) 每次训练结束后，复制本次结果 CSV 到“图像增强实验”文件夹
        if csv_path.exists():
            shutil.copy2(csv_path, save_csv_path)
            done_marker.write_text('done\n', encoding='utf-8')
            summary = _summary_from_csv(csv_path)
            print('已保存结果到: %s' % save_csv_path)
        else:
            summary = {
                'Precision': None,
                'Recall': None,
                'mAP50': None,
                'mAP50-95': None,
                'Train Box Loss': None,
                'Val Box Loss': None,
            }
            print('警告: 未找到 %s' % csv_path)

        all_results.append((exp_name, summary))

    # 汇总结果
    summary_df = pd.DataFrame([
        dict([('experiment', name)] + list(result.items()))
        for name, result in all_results
    ])

    print('\n========== Final Summary ==========' + '\n')
    print(summary_df)

    summary_csv = RESULT_ROOT / 'experiment_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print('\n结果已保存到 %s' % summary_csv)
