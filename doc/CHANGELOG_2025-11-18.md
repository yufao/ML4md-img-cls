# 变更记录（2025-11-18）

本次增量更新聚焦于 NIH 多标签困难样本筛选的可控性与可用性提升：

- 新增：多标签聚合支持按比例筛选
  - 脚本：`src/scripts/aggregate_difficult_multilabel.py`
  - 新参数：`--top_ratio`（0~1]，按“难度得分”选取 Top 比例
  - 难度得分：0.7×熵排名 + 0.3×分歧排名（可扩展）
  - 输出：仅写出被选中的困难样本行到 `out_csv`

- 串接配置：一键运行支持在 YAML 中设置比例
  - 文件：`src/scripts/run_from_config.py`
  - 新增读取键：`post.difficult_ratio`，自动传给聚合脚本
  - 示例：`configs/nih14_wkversion.yaml` 中已加入 `post.difficult_ratio: 0.1`

- 新工具：导出困难样本图片
  - 脚本：`src/scripts/export_difficult_images.py`
  - 功能：根据困难样本 CSV（或结合 manifest）将图片复制到指定目录
  - 适配：单/多标签产物；支持 `--limit` 抽样

- 文档与注释（中文化）
  - 为以下核心脚本与模块补充了中文说明/注释：
    - `aggregate_difficult_multilabel.py`
    - `predict_ensemble_multilabel.py`
    - `train_kfold_multilabel.py`
    - `build_nih_manifest_multilabel.py`
    - `run_from_config.py`
    - `export_difficult_images.py`
    - `src/data/multilabel_dataset.py`
    - `src/utils/splits.py`
    - `src/utils/model_utils.py`

## 兼容性
- 已验证与 NIH 烟雾测试兼容（2 折、1 轮）。
- 若未设置 `post.difficult_ratio`，聚合器回退到分位阈值模式（`--entropy_q/--disagree_q`）。

## Bug 修复
- 修复多标签组分层折分在极端情况下出现空验证折的问题：
  - 由于分组分折缺乏约束，出现把所有组塞进少数折，导致yvl为空，训练启动失败
  - `src/utils/splits.py` 中 `multilabel_stratified_group_kfold` 先对前 n_splits 个组做“每折至少一个组”的预分配，避免空折；
  - 若组数量少于折数，自动缩减有效折数并补齐空折（仅兼容用途），避免 `evaluate` 阶段 `np.concatenate` 报错。

## 后续建议
- 将难度得分权重作为配置项开放（例如 `post.score_weights: {entropy:0.7, disagree:0.3}`）。
- 导出工具支持按子目录分组（例如按类或病人）。
