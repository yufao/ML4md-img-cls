# ml4img: 医学影像难例筛选（UNet + K 折）

目标：
- 支持二/多分类分割数据集，使用 UNet 做 K 折训练与验证。
- 基于验证集单样本指标（IoU/mIoU 或 Dice/mDice）与不确定度（MC Dropout 像素熵）筛选“困难样本”。
- 统一通过 manifest 连接不同数据集；原始数据放 rawig/ 下，训练产物写入 outputs/。

目录规划：
- rawig/<DATASET>/ ... 原始数据（只读）
- manifests/<DATASET>.csv ... 清单（id,image_path[,mask_path][,fold]）
- outputs/<RUN_NAME>/ ... 训练与筛选产物（metrics、hard_sample_ids.txt 等）

快速开始：
1) 准备数据并生成清单
```
python -m src.scripts.build_manifest_from_dirs \
  --images ~/ml4img/rawig/ORD5K/images \
  --masks  ~/ml4img/rawig/ORD5K/masks \
  --out    ~/ml4img/manifests/ORD5K.csv
```

2) 安装依赖（建议在 venv 中）
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3) 训练与筛选（示例：二分类，以 IoU 排序；多分类改 --num-classes 与 --metric）
```
python -m src.train.kfold_unet \
  --manifest ~/ml4img/manifests/ORD5K.csv \
  --outdir   ~/ml4img/outputs/ord5k_unet_k5 \
  --num-classes 1 \
  --metric iou \
  --epochs 10 --kfold 5 --mc-steps 8
```

分类（ORD5K）快速开始：
1) 生成分类 manifest
```
python -m src.scripts.build_ord5k_manifest_cls \
  --csv ~/ml4img/rawig/ORD5K/full_df.csv \
  --images_dir ~/ml4img/rawig/ORD5K/preprocessed_images \
  --out ~/ml4img/manifests/ord5k_cls.csv
```

2) 运行 K 折多标签分类训练与困难样本挖掘（快速验证 2 epoch）
```
python -m src.train.kfold_classifier \
  --csv ~/ml4img/rawig/ORD5K/full_df.csv \
  --image-root ~/ml4img/rawig/ORD5K/preprocessed_images \
  --out-dir ~/ml4img/outputs/ord5k_cls \
  --epochs 2 --kfold 5 --model resnet18 --batch-size 32 --lr 1e-3 \
  --hard-threshold 0.6 --hard-percentile 10 --mc-samples 10
```

产物：
- 分割（UNet）示例运行：`outputs/<run_name>/metrics_fold_*.csv`、`hard_samples.csv`、`hard_sample_ids.txt`（取决于训练脚本配置）。
- 分类（ORD5K）本仓默认输出：
  - 每折：`outputs/ord5k_cls/fold_*/metrics.json`、`outputs/ord5k_cls/fold_*/hard_samples.csv`
  - 汇总：`outputs/ord5k_cls/summary_metrics.json`、`outputs/ord5k_cls/all_hard_samples.csv`

注意：
- 多分类掩膜需为索引图（0..C-1）。若存在忽略值（如 255），使用 --ignore-index 指定（分割管线）。
- ORD5K 无掩膜：请走“分类管线”，通过 MC Dropout 熵 + 置信度/错误筛选困难样本。
