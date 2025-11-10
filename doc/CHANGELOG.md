# CHANGELOG

## 2025-11-09
- 新建独立项目 `~/ml4img`，用于医学影像分割的 K 折训练与“困难样本”筛选。
- 实现通用 UNet（支持二/多分类）、K 折训练、指标（IoU/Dice/mIoU/mDice）与 MC Dropout 不确定度。
- 统一数据接入为 manifest(id,image_path[,mask_path][,fold])；新增目录 rawig/ 与 manifests/。
- 提供清单生成脚本 `src/scripts/build_manifest_from_dirs.py`。
- 初始化文档目录 `doc/` 与 README、CHANGELOG。

### 同日增量：ORD5K（ODIR-5K）无掩膜 -> 引入分类管线与难例挖掘
- 现状确认：`rawig/ORD5K/` 仅包含 `preprocessed_images/` 与 `full_df.csv`，无分割掩膜，无法直接做 UNet 分割训练。
- 新增多标签分类管线（K 折 + 困难样本）：
	- 模型与训练：`src/models/classifier.py`（ResNet18/50 + Dropout），`src/train/kfold_classifier.py`。
	- Manifest 构建脚本：`src/scripts/build_ord5k_manifest_cls.py`（从 `full_df.csv` + `preprocessed_images/` 生成分类清单）。
	- 困难样本定义：预测错误 ∪ 低置信度（真实正类 prob<thr）∪ 高不确定度（MC Dropout 熵 Top 百分位）。
	- 产物：
		- 每折：`outputs/ord5k_cls/fold_*/metrics.json`、`fold_*/hard_samples.csv`
		- 汇总：`outputs/ord5k_cls/summary_metrics.json`、`outputs/ord5k_cls/all_hard_samples.csv`
- 文档更新：
	- `doc/STATUS_2025-11-09.md` 增补“分类策略采纳、执行计划”。
	- `README.md` 增加“分类（ORD5K）快速开始”与命令示例。
