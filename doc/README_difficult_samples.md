# 困难样本筛选 (ResNet + k 折 CV)

本说明文档补充了使用 ResNet 与 k 折交叉验证在 `ml4img` 项目中筛选图像分类困难样本的最小实现路径。

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

## 聚合困难样本
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
`difficult.csv` 包含：`image_id, patient_id, err_rate, avg_max_prob, avg_entropy, unique_preds, is_difficult`。

## 调参建议
- 初始学习率：1e-4 (AdamW)
- 预训练微调阶段：`--freeze_warmup_epochs 2` 后全量解冻
- Epoch: 20~50 小中规模数据

## 置信度校准 (温度缩放)
取验证集 logits 调用 `utils.calibration.temperature_scaling.fit_temperature` 获得温度 T；推理时 logits/T 再 softmax。

## 下一步扩展
- Grad-CAM 可视化困难样本区域
- MC Dropout 或深度集成进一步量化不确定性
- 人工复核闭环 (active learning 重训练)
