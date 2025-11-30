# 困难样本筛选 (ResNet + K 折 CV)

更新日期：2025-11-23（合并多标签与专家标注流程）

本文说明使用 ResNet 与 K 折交叉验证在 `ml4img` 项目中筛选困难样本的完整工作流：
1. 训练并保存每折验证集预测
2. 集成预测生成 `ensemble_preds.csv`
3. 聚合困难样本生成 `difficult.csv`（单标签 / 多标签两套逻辑）
4. 导出困难样本原图到独立文件夹供人工复核
5. 专家标注后与原始 manifest 合并生成“黄金标准”清单

## 数据 CSV 要求
至少列：`image_id,image_path,label` ；可选：`patient_id,window_center,window_width`。

## 训练 k 折
```bash
python -m src.scripts.train_kfold \
  --meta_csv data/meta.csv \
  --img_root data/images \
  --out_dir outputs_cls \
  --folds 5 \
  --pretrained \
  --use_amp
```
纯单通道：加 `--pure_gray`。

## 集成不一致度 (可选)
```bash
python -m src.scripts.predict_ensemble \
  --meta_csv data/meta.csv \
  --img_root data/images \
  --ck_dir outputs_cls \
  --out_csv outputs_cls/ensemble_preds.csv \
  --pretrained
```

## 聚合困难样本（单标签）
```bash
python -m src.scripts.aggregate_difficult \
  --pred_dir outputs_cls \
  --out_csv outputs_cls/difficult.csv \
  --max_prob_thresh 0.6 \
  --err_rate_thresh 0.5 \
  --use_quantile_entropy 0.8 \
  --unique_pred_thresh 2 \
  --ensemble_csv outputs_cls/ensemble_preds.csv
```

## 输出指标
`difficult.csv`（单标签版本）字段：
`image_id, patient_id, err_rate, avg_max_prob, avg_entropy, unique_preds, is_difficult`

## 聚合困难样本（多标签）
```bash
python -m src.scripts.aggregate_difficult_multilabel \
  --ensemble_csv outputs/ord5k_cls_multilabel/ensemble_preds.csv \
  --out_csv outputs/ord5k_cls_multilabel/difficult.csv \
  --entropy_q 0.8 --disagree_q 0.8
```
或使用按比例：`--top_ratio 0.1`（前 10% 难度得分）。

多标签 `difficult.csv` 字段：
`image_id, patient_id, avg_probs, mean_entropy, disagreement_jaccard, diff_score, is_difficult`

难度得分融合：默认 `0.7 * 熵排名 + 0.3 * Jaccard 分歧排名`（分歧取 1-平均 Jaccard）。

## 调参建议
- 初始学习率：1e-4 (AdamW)
- 预训练微调阶段：`--freeze_warmup_epochs 2` 后全量解冻
- Epoch: 20~50 小中规模数据

## 导出困难样本原图
推荐使用新版脚本（支持多级路径回退与按类分目录）：

**核心参数:**
- `--csv`: `difficult.csv` 文件路径。
- `--img_root`: 图片的根目录。如果 CSV 中的路径已经是完整的相对路径（相对于项目根目录），可以设为 `.`。
- `--out_dir`: 导出图片的目录。
- `--path_col`: CSV 文件中包含图片路径的列名。
- `--is_difficult_col`: (可选) CSV 文件中标记是否为困难样本的列名，默认为 `is_difficult`。
- `--with_classes`: (可选) 如果提供，会根据类别列创建子目录。
- `--manifest_csv`: (可选) 当 `difficult.csv` 中只有 `image_id` 时，提供此参数以从完整的 manifest 文件中查找 `image_path`。

**示例 1: NIH 数据集 (使用 manifest 补全路径)**
```bash
python -m src.scripts.extract_difficult_images \
  --csv outputs/nih14_resnet50/difficult.csv \
  --img_root . \
  --path_col image_path \
  --is_difficult_col is_difficult \
  --out_dir f4dficimg/nih14_difficult \
  --manifest_csv manifests/nih-chest-xrays-14/manifest.csv \
  --manifest_id_col image_id \
  --manifest_path_col image_path
```

**示例 2: ORD5K 多标签数据集**
```bash
python -m src.scripts.extract_difficult_images \
  --csv outputs/ord5k_cls_multilabel/difficult.csv \
  --img_root rawig/ORD5K/preprocessed_images \
  --out_dir exports/ord5k_difficult_images
```

**示例 3: MRI 数据集 (按类别分子目录)**
```bash
python -m src.scripts.extract_difficult_images \
  --csv outputs/brain_tumor_resnet50/difficult.csv \
  --img_root rawig/MRItumor \
  --out_dir exports/mri_difficult_images \
  --with_classes
```

脚本内部路径解析策略：
1. 检查 CSV 提供的路径是否直接有效（例如，从项目根目录出发的相对路径）。
2. 检查路径是否为绝对路径。
3. 尝试拼接 `img_root` 和 CSV 中的路径。
4. 若上述均失败，则在 `img_root` 下递归搜索文件名。

旧脚本 `export_difficult_images.py` 已标记弃用，仅保留兼容需求时使用。

## 专家标注合并（黄金标准清单）
当专家对导出的困难样本重新打标签后，使用：
```bash
python -m src.scripts.merge_expert_annotations \
  --original_manifest manifests/ord5k_cls_multilabel.csv \
  --expert_annotations expert_data/ord5k_expert.xlsx \
  --output_manifest manifests/ord5k_cls_multilabel_golden.csv \
  --join_key image_path \
  --label_cols N D G C A H M O
```
单标签示例：
```bash
python -m src.scripts.merge_expert_annotations \
  --original_manifest manifests/mri_manifest.csv \
  --expert_annotations expert_data/mri_expert.xlsx \
  --output_manifest manifests/mri_manifest_golden.csv \
  --join_key image_id \
  --label_cols class_name class_index
```
合并规则：
- 匹配 `join_key` 后用专家列覆盖原始 manifest 对应标签列
- 未出现于专家文件的样本保持原标签
- 可新增标签列（原始不存在时自动添加）

最佳实践：使用 `image_path`（含相对子目录+扩展名）作为 join_key，避免仅用裸 ID 造成冲突或匹配失败。

## 置信度校准 (温度缩放)
取验证集 logits 调用 `utils.calibration.temperature_scaling.fit_temperature` 获得温度 T；推理时 logits/T 再 softmax。

## 下一步扩展
- Grad-CAM 可视化困难区域定位
- MC Dropout / 深度集成不确定性增强（多标签脚本已支持熵与分歧）
- 主动学习闭环：困难 + 专家标注合并后增量重训
- 难度得分权重调参：可暴露到配置（例如 0.5/0.5 或自适应）

## 常见问题 FAQ
Q: 为什么有些文件无法复制？\nA: 检查 manifest 是否存储了完整相对路径（含类别子目录与扩展名）。仅存裸 ID 时依赖回退搜索，可能遗漏。\n
Q: `is_difficult` 全 0 怎么办？\nA: 调整聚合参数（降低分位数或提高比例），或确认先前 ensemble 步骤成功生成概率与熵。\n
Q: 专家标注文件列名不一致？\nA: 统一列名再执行合并，或先用 pandas 重命名列后再调用脚本。

## 数据增强策略更新 (2023-10-27)

### 保守增强策略 (Conservative Strategy)
在困难样本筛选任务中，我们发现传统的几何增强（如水平翻转、旋转）可能会破坏医学图像的关键解剖学特征，导致模型产生错误的“高不确定性”预测，从而污染困难样本集。

例如：
- **眼底图 (Fundus)**: 视盘和黄斑的相对位置决定了左/右眼。水平翻转会将左眼变为解剖学上不可能的“右眼”（视盘位置错误），导致模型困惑。
- **胸片 (CXR)**: 心脏通常位于左侧。水平翻转会制造“右位心”假象；大幅度旋转会改变胸腔积液的重力分布特征。
- **脑部 MRI**: 左右脑半球的功能区和病灶位置具有侧向性，翻转可能混淆病灶定位。

### 解决方案
我们将所有数据集（ORD5K, NIH, MRI）的增强策略统一调整为 **`aug_strategy: none`**。
- **禁用**: 所有几何变换（翻转、旋转、剪切）。
- **保留**: 仅保留轻微的色彩抖动 (`ColorJitter(brightness=0.1, contrast=0.1)`) 以模拟成像差异，同时不改变解剖结构。

此更改已在 `src/data/cls_dataset.py` 和 `src/data/multilabel_dataset.py` 中实现，并更新了相关配置文件。
