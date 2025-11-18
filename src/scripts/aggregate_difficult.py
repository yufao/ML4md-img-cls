import os
import glob
import json
import argparse
import numpy as np
import pandas as pd


def parse_probs(cell):
    """Parse a serialized list of probabilities from CSV (JSON or repr)."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(x) for x in cell]
    if isinstance(cell, str):
        try:
            return [float(x) for x in json.loads(cell)]
        except Exception:
            try:
                # Fallback for repr-like strings
                s = cell.strip().lstrip('[').rstrip(']')
                return [float(x) for x in s.split(',') if x.strip()]
            except Exception:
                return []
    return []


def aggregate_val(pred_dir: str) -> pd.DataFrame:
    """Aggregate fold validation predictions into per-image statistics.

    Expects per-fold CSVs like '<pred_dir>/fold{K}_val_preds.csv' with columns:
      - image_id, patient_id (optional), true_label, pred_label, max_prob, entropy
    """
    files = sorted(glob.glob(os.path.join(pred_dir, 'fold*_val_preds.csv')))
    if not files:
        raise RuntimeError(f'No fold*_val_preds.csv in {pred_dir}')

    agg = {}
    for f in files:
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            iid = r['image_id']
            rec = agg.setdefault(iid, {
                'image_id': iid,
                'patient_id': r.get('patient_id', None),
                'true_label': int(r['true_label']),
                'folds': 0,
                'errors': 0,
                'max_probs': [],
                'entropies': [],
                'preds': []
            })
            rec['folds'] += 1
            rec['errors'] += int(r.get('is_error', int(r['true_label'] != r['pred_label'])))
            rec['max_probs'].append(float(r['max_prob']))
            rec['entropies'].append(float(r.get('entropy', 0.0)))
            rec['preds'].append(int(r['pred_label']))

    rows = []
    for iid, v in agg.items():
        rows.append({
            'image_id': iid,
            'patient_id': v['patient_id'],
            'true_label': v['true_label'],
            'folds': v['folds'],
            'errors': v['errors'],
            'err_rate': v['errors'] / max(1, v['folds']),
            'avg_max_prob': float(np.mean(v['max_probs'])) if v['max_probs'] else np.nan,
            'avg_entropy': float(np.mean(v['entropies'])) if v['entropies'] else np.nan,
            'unique_preds_val': len(set(v['preds']))
        })
    return pd.DataFrame(rows)


def mark_difficult(df: pd.DataFrame, max_prob_thresh: float, err_rate_thresh: float, ent_thresh: float, unique_pred_thresh: int):
    mask = (
        (df['avg_max_prob'] < max_prob_thresh) |
        (df['err_rate'] > err_rate_thresh) |
        (df['avg_entropy'] > ent_thresh) |
        (df.get('unique_preds', df.get('unique_preds_val', 1)) >= unique_pred_thresh)
    )
    df['is_difficult'] = mask.astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--max_prob_thresh', type=float, default=0.6)
    parser.add_argument('--err_rate_thresh', type=float, default=0.5)
    parser.add_argument('--unique_pred_thresh', type=int, default=2)
    parser.add_argument('--use_quantile_entropy', type=float, default=0.8)
    parser.add_argument('--entropy_thresh', type=float, default=-1.0)
    parser.add_argument('--ensemble_csv', default='')
    args = parser.parse_args()

    df = aggregate_val(args.pred_dir)
    if args.ensemble_csv and os.path.exists(args.ensemble_csv):
        ens = pd.read_csv(args.ensemble_csv)
        df = df.merge(ens[['image_id', 'avg_max_prob', 'avg_entropy', 'unique_preds']], on='image_id', how='left', suffixes=('_val', ''))
        df['avg_max_prob'] = df['avg_max_prob'].fillna(df['avg_max_prob_val'])
        df['avg_entropy'] = df['avg_entropy'].fillna(df['avg_entropy_val'])
        df['unique_preds'] = df['unique_preds'].fillna(df['unique_preds_val'])

    if args.entropy_thresh >= 0:
        ent_thresh = args.entropy_thresh
    else:
        ent_thresh = df['avg_entropy'].quantile(args.use_quantile_entropy)

    df = mark_difficult(df, args.max_prob_thresh, args.err_rate_thresh, ent_thresh, args.unique_pred_thresh)
    df.to_csv(args.out_csv, index=False)
    print(f'Wrote {args.out_csv}. Difficult: {int(df.is_difficult.sum())}/{len(df)}')

if __name__ == '__main__':
    main()
