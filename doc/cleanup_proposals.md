# 脚本重复 / 过时清理建议

日期: 2025-11-21

## 分类概览

- 训练相关: `train_kfold.py`, `train_kfold_multilabel.py`, `src/train/kfold_classifier.py`
- 集成预测: `predict_ensemble.py`, `predict_ensemble_multilabel.py`
- 困难样本聚合: `aggregate_difficult.py`, `aggregate_difficult_multilabel.py`
- 困难样本导出: `export_difficult_images.py`, `extract_difficult_images.py`
- Manifest 构建: `build_ord5k_manifest_cls.py`, `build_ord5k_manifest_multilabel.py`, `build_mri_manifest.py`, `build_nih_manifest_multilabel.py`, `build_manifest_from_dirs.py`, `normalize_manifest.py`
- 清理与辅助: `clean_outputs.py`, `run_from_config.py`

## 详细分析与建议

### 1. 训练脚本
| 脚本 | 功能 | 状态 | 建议 |
|------|------|------|------|
| `train_kfold.py` | 单标签分类 K 折 | 活跃 / 已中文化 | 保留，作为单标签主脚本 |
| `train_kfold_multilabel.py` | 多标签分类 K 折 | 活跃 / 已中文化 | 保留，作为多标签主脚本 |
| `src/train/kfold_classifier.py` | 单标签 K 折 + MC Dropout 不确定性挖掘 | 功能与 `train_kfold.py` 部分重合 | 若仅需不确定性功能，可将其逻辑迁入 `train_kfold.py` 并删除该文件；否则标注为“高级实验脚本” |

迁移建议：在 `train_kfold.py` 中新增可选参数 `--mc_dropout_steps` 与对应记录逻辑后，删除 `src/train/kfold_classifier.py`。

### 2. 集成预测脚本
| 脚本 | 功能 | 状态 | 建议 |
|------|------|------|------|
| `predict_ensemble.py` | 单标签多折集成 + 概率/熵 | 活跃 | 可与多标签版本统一封装 |
| `predict_ensemble_multilabel.py` | 多标签多折集成 + Jaccard 分歧 | 活跃 | 与上合并：新增 `--multilabel` 分支 |

统一方案：创建 `predict_ensemble_unified.py`（或改造现有单标签脚本），根据 `--multilabel` 走不同聚合逻辑，完成后删除旧多标签脚本。

### 3. 困难样本聚合
| 脚本 | 功能 | 状态 | 建议 |
|------|------|------|------|
| `aggregate_difficult.py` | 单标签错误率 + 熵 + 置信度规则 | 活跃 | 保留或统一 |
| `aggregate_difficult_multilabel.py` | 多标签熵 + Jaccard 分歧 + 排序融合 | 已增强（is_difficult, diff_score） | 建议统一 |

统一方案：将单标签与多标签的聚合策略整合到 `aggregate_difficult_unified.py`：
- 公共输入：`--ensemble_csv` / 或目录。
- 模式选择：`--multilabel`。
- 输出统一包含：`image_id`, `patient_id`, `avg_probs`, `mean_entropy`, `disagreement_jaccard`(多标签时), `diff_score`, `is_difficult`。
完成后删除旧两个脚本。

### 4. 困难样本导出
| 脚本 | 功能 | 状态 | 建议 |
|------|------|------|------|
| `export_difficult_images.py` | 旧版困难样本导出（不含 is_difficult 支持） | 过时 | 标记弃用，阶段性删除 |
| `extract_difficult_images.py` | 新版，支持 `is_difficult` 与类子目录 | 活跃 | 设为唯一保留版本 |

行动：确认无其它流程引用 `export_difficult_images.py` 后删除，并在 README 中指向新版脚本。

### 5. Manifest 构建/规范化
| 脚本 | 功能 | 状态 | 建议 |
|------|------|------|------|
| `build_ord5k_manifest_cls.py` | ORD5K 单标签 Manifest | 若不再训练单标签则冗余 | 若后续只保留多标签，可删除 |
| `build_ord5k_manifest_multilabel.py` | ORD5K 多标签 Manifest | 主用 | 保留 |
| `build_mri_manifest.py` | MRI 文件夹结构扫描生成 Manifest | 主用 | 保留 |
| `build_nih_manifest_multilabel.py` | NIH14 CSV/文本标签解析生成多标签 Manifest | 主用 | 保留 |
| `build_manifest_from_dirs.py` | 泛化目录转 CSV（多任务兼容） | 功能与上部分重叠 | 评估是否被调用；若 seldom 使用，可合并到 README 示例或归档 |
| `normalize_manifest.py` | 规范现有 Manifest（路径标准化） | 独立工具 | 保留 |

统一策略：保留“专用 + 规范化”两个层级：
- 专用：数据集格式多样（MRI/NIH/ORD5K）保留独立脚本便于维护文档。
- 泛化：若 `build_manifest_from_dirs.py` 使用率低，可在文档中迁移其示例后删除。

### 6. 其它辅助脚本
| 脚本 | 功能 | 建议 |
|------|------|------|
| `clean_outputs.py` | 输出目录列举 / 删除 | 保留；可扩展 dry-run 模式 |
| `run_from_config.py` | 单入口训练+后处理 | 保留；后续若统一 ensemble/aggregate 接口仅需微调调用部分 |

## 推荐执行顺序（分阶段）
1. 标记废弃：`export_difficult_images.py`、（若单标签训练不再需要）`build_ord5k_manifest_cls.py`。
2. 合并脚本族：ensemble 与 difficult 聚合统一（新增 unified 脚本）。
3. 迁移不确定性逻辑：将 `kfold_classifier.py` 的 MC Dropout 选样能力融入 `train_kfold.py`。
4. 清理：删除废弃脚本并更新 README / QUICK_START 指向新路径。
5. 审核：运行一次 `run_from_config.py` 确认统一后流程无回归。

## 风险与缓解
- 风险：删除脚本导致已有自动化调用失败。缓解：先打“deprecated”标记（文件头部注释），观察 1-2 周后再删。
- 风险：统一脚本参数膨胀难维护。缓解：按模式分组验证，保留最小公共子集，复杂选项仅在对应模式可见。
- 风险：多标签与单标签聚合指标含义不同。缓解：统一输出字段但允许空列（如单标签时空置 `disagreement_jaccard`）。

## 后续需要人工确认的点
- 是否仍有单标签 ORD5K 训练场景？
- `build_manifest_from_dirs.py` 的实际使用频次与场景。
- 是否需要保留独立 MC Dropout 脚本以供研究对比。

## 建议的 README 更新要点
- 更新“困难样本导出”章节指向 `extract_difficult_images.py`。
- 增加“脚本统一策略”简述：ensemble 与 difficult 聚合已统一。
- 标注已弃用脚本与迁移说明。

## 即将执行的行动（待确认）
- 给废弃脚本头部添加 `# DEPRECATED` 注释与替代脚本引用。
- 新建 unified 脚本原型（待用户确认再实施）。

---
如需直接开始第 1 阶段（添加弃用标记），请确认或调整上述清单。
