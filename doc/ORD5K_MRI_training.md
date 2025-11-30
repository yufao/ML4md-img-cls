# ORD5K 与 MRI 训练说明

## 任务类型判定
- NIH14: 多标签 (14 种病灶可共存) → BCEWithLogitsLoss + sigmoid + `label_cols`。
- ORD5K: 统计结果显示行内 N..O 八列求和>1 的比例≈15.66%，属于多标签 → BCEWithLogitsLoss + `label_cols: [N,D,G,C,A,H,M,O]`。
- MRI Brain Tumor: 单标签多类 (glioma / meningioma / notumor / pituitary) → CrossEntropyLoss + softmax + `label_col: class_index`。

## YAML 关键字段
| 字段 | 单标签多类 | 多标签 |
|------|-----------|--------|
| `multilabel` | false | true |
| `label_col` | 类别索引或名称 | 不使用（留空） |
| `label_cols` | 不使用 | 多个 0/1 列名列表 |
| `prob_th` | 可忽略 | F1 阈值，建议 0.2~0.5 网格搜索 |
| `print_debug` | 可选 | 建议早期设为 true 观察分布 |

## 多标签指标解释
- AUC：阈值无关，类别极端不均衡时初期正常增加。
- micro-F1：受阈值影响，默认 0.5 容易全负 → 0。降低 `prob_th` 可提前看到非零 F1。
- macro AUC NaN：全部有效类别在验证集中为常量（全 0 或全 1）时跳过 → 列表为空 → NaN。

## MRI 验证集 acc=0 的根因与修复
### 根因
患者分组折分导致某折验证集仅包含一个类别，训练集缺该类别 → 模型验证时无法学习该类别分布，acc 维持 0。
### 修复策略
1. 在 `train_kfold.py` 增加折分检查，若某折验证集缺任意类别则回退到 `StratifiedKFold`。
2. 暂时禁用 `patient_col`（或指向不存在列名）以验证可学性。
3. 切换为 RGB 三通道 (`pure_gray: false`) 保留预训练权重的早期表示能力；调低或去除 warmup 冻结。

## 阈值选择建议 (多标签)
1. 运行验证后收集概率：`fold*_val_preds.csv`。
2. 网格：`t in np.arange(0.05,0.55,0.05)` 计算 micro/macro F1，选择最优。
3. 若各类不均衡严重，可独立为每类搜索阈值，或依据 Precision-Recall 曲线的 F1 最大点。保持一套全局阈值便于部署，可在文档中注明。

## 调试清单
- 折分日志：是否每折 train/val label 分布都覆盖所有类。
- 训练/验证损失：多标签初期 loss 快速下降但 F1=0 正常；观察 AUC 提升是否正常。
- 检查 CSV 列：`id_col / path_col / patient_col / label_cols` 是否与数据一致。
- 图像模式：灰度复制为 RGB（3ch） → 预训练收敛更快；若使用单通道，要确保第一层已替换并正确初始化。

## 常见问题与处理
| 问题 | 现象 | 处理 |
|------|------|------|
| 验证集单类 | acc=0 或 AUC NaN | 回退普通 StratifiedKFold |
| micro-F1 一直 0 | 全负预测 | 降低 `prob_th`; 继续训练; 调整 pos_weight |
| AUC NaN | 所有有效类在验证集常量 | 检查折分; 增加样本或调整分组策略 |
| 训练慢 | GPU 利用高, IO 成瓶颈 | 降 folds / 降 img_size / resnet18 / 修 torchvision 扩展 |

## 运行示例
多标签 ORD5K：
```bash
python -m src.scripts.run_from_config --config configs/ord5k_wkversion.yaml
```
单标签 MRI：
```bash
python -m src.scripts.run_from_config --config configs/brain_tumor_wkversion.yaml
```

## 后续改进建议
- 为多标签添加每类独立阈值自动搜索脚本。
- 记录阈值选择过程到 JSONL（增加 `best_prob_th` 字段）。
- 增加预测概率分布直方图导出，辅助阈值确定。

(更新日期: 2025-11-21)
