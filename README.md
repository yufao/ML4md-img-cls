# ml4img: 医学影像分类 & 分割困难样本筛选 (ResNet + K 折 + 不确定性)

## 目标
统一在多来源、多格式 CSV 数据集上进行：
- 图像分类 (二分类/多分类) ResNet K 折训练，收集样本级预测置信度与熵，标记困难样本。
- 图像分割 (UNet 等) K 折训练，按单图指标 (Dice/mIoU) 与 MC Dropout 像素熵筛选困难区域/样本。
- 支持灰度/彩色混合输入与患者级分组拆分，避免数据泄露。
- 提供 manifest 归一化脚本与列名自动匹配，降低不同数据集 CSV 格式差异影响。

## 目录结构（统一约定）
```
rawig/                # 原始数据集根目录（只读，不做改动）
  NIH/                # NIH ChestX-ray14 原始结构（images_001..images_012）
  ORD5K/              # ORD5K 原始数据
manifests/            # 标准化后的清单 CSV（统一放这里）
                      # 例如：manifests/nih14_manifest.csv, manifests/ord5k_cls.csv
outputs/              # 训练与筛选产物
src/                  # 代码
configs/              # YAML 配置
```

## 安装依赖
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 使用配置文件一键运行（推荐）
```bash
python -m src.scripts.run_from_config --config configs/train_config_example.yaml
```

## Manifest 标准化（多格式 CSV 兼容）
统一将清单 CSV 放在 `manifests/` 下；NIH 与 ORD5K 均遵循该约定。
NIH14 可使用 `build_nih_manifest_multilabel.py` 从原始 `Data_Entry_2017.csv` 生成：
```bash
python -m src.scripts.build_nih_manifest_multilabel \
  --nih_csv rawig/NIH/Data_Entry_2017.csv \
  --images_root rawig/NIH \
  --out manifests/nih14_manifest.csv \
  --include_json_vec
```
亦可使用 `normalize_manifest.py` 或数据集类的列名自动识别能力：
- 支持常见别名：image_id/id/uid、image_path/path/file、label/target/class、patient_id/patient/pid 等。
- 可通过 `column_alias` 参数显式指定映射。

示例：
```python
from src.data.cls_dataset import MedicalImageDataset
ds = MedicalImageDataset(csv_path='raw.csv', images_root='images/', column_alias={'label':['gt_label']})
```

## 分类 K 折训练 (ResNet)
```bash
python -m src.scripts.train_kfold \
  --meta_csv rawig/ORD5K/full_df.csv \
  --img_root rawig/ORD5K/preprocessed_images \
  --out_dir outputs/ord5k_cls \
  --folds 5 --model resnet50 --pretrained --use_amp \
  --epochs 20 --batch_size 32 --lr 1e-4 --weight_decay 1e-4 \
  --aug_strategy fundus
```
**数据增强策略 (`--aug_strategy`)：**
- `fundus`: 垂直翻转 + 旋转（适用于眼底图，避免水平翻转破坏左右眼特征）。
- `cxr`: 仅旋转（适用于胸片，避免翻转破坏心脏位置与重力特征）。
- `mri`: 水平翻转 + 旋转（适用于脑部 MRI，增加左右脑病灶多样性）。
- `default`: 水平翻转 + 旋转（适用于通用自然图像）。

若数据纯灰度：添加 `--pure_gray`；若启用 CT 窗宽窗位：添加 `--apply_ct_window`。

产物：
- `foldX_best.pth` 每折最佳权重
- `foldX_val_preds.csv` 验证样本预测与不确定度 (max_prob, entropy)

## 集成预测与不一致度 (单标签，可选)
```bash
python -m src.scripts.predict_ensemble \
  --meta_csv rawig/ORD5K/full_df.csv \
  --img_root rawig/ORD5K/preprocessed_images \
  --ck_dir outputs/ord5k_cls \
  --out_csv outputs/ord5k_cls/ensemble_preds.csv --pretrained
```
输出每张图的平均概率、熵与模型间预测差异计数。

## 聚合困难样本（单标签）
```bash
python -m src.scripts.aggregate_difficult \
  --pred_dir outputs/ord5k_cls \
  --out_csv outputs/ord5k_cls/difficult.csv \
  --max_prob_thresh 0.6 --err_rate_thresh 0.5 --use_quantile_entropy 0.8 --unique_pred_thresh 2 \
  --ensemble_csv outputs/ord5k_cls/ensemble_preds.csv
```
输出字段：image_id, patient_id, err_rate, avg_max_prob, avg_entropy, unique_preds, is_difficult。

## NIH14 多标签流程（新增）
1) Manifest：`manifests/nih14_manifest.csv`（上文生成脚本）
2) 训练与后处理（示例配置 `configs/nih14_wkversion.yaml`）：
```bash
python -m src.scripts.run_from_config --config configs/nih14_wkversion.yaml

# 仅后处理（调参不重训，便于调比例）
python -m src.scripts.run_from_config --config configs/nih14_wkversion.yaml --post_only
```
3) 可调困难样本比例：在配置中设置
```yaml
post:
  ensemble: true
  aggregate: true
  skip_if_exists: false
  difficult_ratio: 0.1   # 10% 困难样本
```
4) 导出困难样本原图（便于人工复核）：
```bash
python -m src.scripts.export_difficult_images \
  --difficult_csv outputs/nih14_resnet50/difficult.csv \
  --meta_csv manifests/nih14_manifest.csv \
  --out_dir exports/nih14_difficult_images \
  --img_root rawig/NIH
```

## 分割困难样本（UNet 管线说明）
分割训练脚本示例（若已存在）：
```bash
python -m src.train.kfold_unet \
  --manifest manifests/DATASET_A.csv \
  --outdir outputs/datasetA_unet_k5 \
  --num-classes 1 --epochs 30 --kfold 5 --mc-steps 8
```
困难样本判定可基于：低 Dice/mIoU、MC Dropout 均值像素熵高、预测波动大。

## ResNet 与 SimpleCNN 说明
- `src/models/classifier.py` 中 `build_classifier` 支持 `resnet18`, `resnet50`, `simple_cnn`。
- 更丰富的输入通道修改与冻结策略在 `src/utils/model_utils.build_resnet` 中（分类 K 折脚本使用）。
- 若要扩展 EfficientNet/DenseNet，在 MODEL_REGISTRY 中添加并对齐权重 API。

## 单通道 vs 三通道策略
- 默认：将灰度重复到 RGB 三通道（利用预训练特征）。
- 纯灰度：使用 `--pure_gray` 改第一层 conv 权重 (平均化) 并减少不必要的通道冗余。

## 不确定性与困难样本判定策略（概要）
1. 错误样本 (pred != true_label)
2. 低置信度 (max_prob < 阈值或下分位)
3. 高熵 (entropy 上分位数)
4. 模型集成不一致 (unique_preds ≥ 2)
5. 多折出错率 (err_rate > 阈值)

单标签：默认“或”组合，可在 `aggregate_difficult.py` 中改为“与”或权重打分。
多标签（NIH14）：支持分位阈值模式与固定比例模式（推荐，`post.difficult_ratio`）。

## 置信度校准 (温度缩放)
使用 `src/utils/calibration/temperature_scaling.py`：
```python
from src.utils.calibration.temperature_scaling import fit_temperature
import torch, pandas as pd
df = pd.read_csv('outputs/ord5k_cls/fold0_val_preds.csv')
# 假设已缓存 logits 张量 val_logits, labels 张量
scaler = fit_temperature(val_logits, labels)
calibrated = torch.softmax(scaler(val_logits), dim=1)
```

## 数据列名自动识别
`MedicalImageDataset` 支持别名自动匹配或传入 `column_alias` 显式指定；不匹配时报错列出映射结果，便于调试。

## 后续扩展建议
- Grad-CAM 可视化困难样本区域
- 主动学习循环：人工复核后增量重训
- 多模型集成与 MC Dropout 结合的方差分析
- 半监督自训练：困难样本高熵伪标签剔除或低熵样本强化

## 常见问题
Q: 为什么找不到 ResNet 实现？
A: ResNet 构建在 `src/utils/model_utils.py` (函数 `build_resnet`)，分类脚本直接调用；`classifier.py` 另提供简化接口与 SimpleCNN 快速验证。

Q: CSV 列名不统一怎么办？
A: 使用数据集的别名自动识别，或先用 `build_nih_manifest_multilabel.py / normalize_manifest.py` 生成统一清单。

Q: 如何只筛选错误但置信度高的潜在标注噪声样本？
A: 在单标签聚合产物中过滤 `is_error==1` 且 `max_prob>0.9`；多标签可结合高熵/高分歧样本人工复核。

## 许可证
见 `LICENSE`。
