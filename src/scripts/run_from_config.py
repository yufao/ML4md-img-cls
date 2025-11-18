"""从 YAML 配置文件一键运行训练与后处理（可选）。

支持旧版“平铺”配置（键名与训练脚本参数一致）与“增强版”配置：

Example YAML (flat):
    experiment: ord5k_resnet50
    meta_csv: rawig/ORD5K/full_df.csv
    img_root: rawig/ORD5K/preprocessed_images
    out_dir: outputs/ord5k_cls
    folds: 10
    model: resnet50
    pretrained: true
    pure_gray: false
    img_size: 224
    batch_size: 32
    epochs: 20
    lr: 1e-4
    weight_decay: 1e-4
    num_workers: 4
    use_amp: true
    label_col: label
    id_col: image_id
    path_col: image_path
    patient_col: patient_id
    apply_ct_window: false
    freeze_warmup_epochs: 2
    log_jsonl: outputs/ord5k_cls/metrics.jsonl

增强版（带 post 段）：
    post:
        ensemble: true     # default true
        aggregate: true    # default true
        skip_if_exists: false

多标签模式（NIH14 等）：
    multilabel: true
    # 可显式指定 label_cols（列表），或依赖 NIH14 默认列自动识别
    label_cols: [Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia]

新增：post.difficult_ratio（0~1）用于按比例筛选困难样本，优先于分位阈值。
"""
import argparse
import subprocess
import sys
import os
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config')
    ap.add_argument('--post_only', action='store_true', help='Only run post-processing (ensemble + aggregate)')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    multilabel = bool(cfg.get('multilabel', False))
    # Build CLI args for train scripts
    cmd = [sys.executable, '-m', 'src.scripts.train_kfold_multilabel' if multilabel else 'src.scripts.train_kfold']
    def add_flag(k, v):
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k}')
        else:
            cmd.extend([f'--{k}', str(v)])

    keys = [
        'meta_csv','img_root','out_dir','folds','seed','model','pretrained','pure_gray','img_size','batch_size',
        'epochs','lr','weight_decay','num_workers','use_amp','label_col','id_col','path_col','patient_col',
        'apply_ct_window','freeze_warmup_epochs','log_jsonl'
    ]
    for k in keys:
        if k in cfg:
            add_flag(k, cfg[k])

    # Multilabel-specific args
    if multilabel:
        if 'label_cols' in cfg and isinstance(cfg['label_cols'], (list, tuple)):
            add_flag('label_cols', ','.join([str(x) for x in cfg['label_cols']]))
        if 'labels_json_col' in cfg and cfg['labels_json_col']:
            add_flag('labels_json_col', cfg['labels_json_col'])

    if not args.post_only:
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print('Skip training (post_only=True)')

    # Post-processing: ensemble prediction and difficult sample aggregation
    post_cfg = cfg.get('post', {}) if isinstance(cfg, dict) else {}
    do_ensemble = post_cfg.get('ensemble', True)
    do_aggregate = post_cfg.get('aggregate', True)
    skip_if_exists = post_cfg.get('skip_if_exists', False)
    difficult_ratio = post_cfg.get('difficult_ratio', None)

    # Gather common args
    out_dir = cfg.get('out_dir')
    meta_csv = cfg.get('meta_csv')
    img_root = cfg.get('img_root', '')
    model = cfg.get('model', 'resnet50')
    pure_gray = bool(cfg.get('pure_gray', False))
    img_size = int(cfg.get('img_size', 224))
    num_workers = int(cfg.get('num_workers', 4))
    pretrained = bool(cfg.get('pretrained', False))

    if do_ensemble:
        ens_out = os.path.join(out_dir, 'ensemble_preds.csv')
        if skip_if_exists and os.path.exists(ens_out):
            print(f'Skip ensemble: exists {ens_out}')
        else:
            cmd2 = [
                sys.executable, '-m', 'src.scripts.predict_ensemble_multilabel' if multilabel else 'src.scripts.predict_ensemble',
                '--meta_csv', meta_csv,
                '--img_root', img_root,
                '--ck_dir', out_dir,
                '--out_csv', ens_out,
                '--model', model,
                '--img_size', str(img_size),
                '--num_workers', str(num_workers),
            ]
            # Pass column names if present in config (predict_ensemble will still alias-fallback)
            for key in ['label_col', 'id_col', 'path_col', 'patient_col']:
                if key in cfg and cfg.get(key):
                    cmd2 += [f'--{key}', str(cfg.get(key))]
            if multilabel and 'label_cols' in cfg and isinstance(cfg['label_cols'], (list, tuple)):
                cmd2 += ['--label_cols', ','.join([str(x) for x in cfg['label_cols']])]
            if multilabel and 'labels_json_col' in cfg and cfg['labels_json_col']:
                cmd2 += ['--labels_json_col', str(cfg['labels_json_col'])]
            if pure_gray:
                cmd2.append('--pure_gray')
            if pretrained:
                cmd2.append('--pretrained')
            print('Running:', ' '.join(cmd2))
            subprocess.run(cmd2, check=True)

    if do_aggregate:
        diff_out = os.path.join(out_dir, 'difficult.csv')
        if skip_if_exists and os.path.exists(diff_out):
            print(f'Skip aggregate: exists {diff_out}')
        else:
            if multilabel:
                cmd3 = [
                    sys.executable, '-m', 'src.scripts.aggregate_difficult_multilabel',
                    '--ensemble_csv', os.path.join(out_dir, 'ensemble_preds.csv'),
                    '--out_csv', diff_out,
                    '--entropy_q', '0.8',
                    '--disagree_q', '0.8',
                ]
                if difficult_ratio is not None:
                    cmd3 += ['--top_ratio', str(difficult_ratio)]
            else:
                cmd3 = [
                    sys.executable, '-m', 'src.scripts.aggregate_difficult',
                    '--pred_dir', out_dir,
                    '--out_csv', diff_out,
                    '--max_prob_thresh', '0.6',
                    '--err_rate_thresh', '0.5',
                    '--use_quantile_entropy', '0.8',
                    '--unique_pred_thresh', '2',
                    '--ensemble_csv', os.path.join(out_dir, 'ensemble_preds.csv')
                ]
            print('Running:', ' '.join(cmd3))
            subprocess.run(cmd3, check=True)


if __name__ == '__main__':
    main()
