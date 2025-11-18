"""多标签困难样本聚合脚本。

两种模式：
1) 比例模式（推荐）：根据难度得分选取前 R（R=top_ratio×N）。
    难度得分默认为 0.7×熵排名 + 0.3×分歧排名，熵越大/分歧越大越难。
2) 分位数模式（兜底）：满足 mean_entropy ≥ entropy_q 或 disagreement_jaccard ≥ disagree_q 的样本。

仅输出被选中的困难样本到 out_csv（不再整表打 is_difficult 标记）。
"""
import os
import argparse
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--entropy_q', type=float, default=0.8)
    ap.add_argument('--disagree_q', type=float, default=0.8)
    ap.add_argument('--top_ratio', type=float, default=None, help='Select top ratio (0~1] by difficulty score; overrides quantiles if provided')
    args = ap.parse_args()

    ens = pd.read_csv(args.ensemble_csv)
    n = len(ens)
    if n == 0:
        pd.DataFrame([]).to_csv(args.out_csv, index=False)
        print(f"Difficult samples saved to {args.out_csv}; count=0 of 0")
        return

    has_ent = 'mean_entropy' in ens.columns
    has_dis = 'disagreement_jaccard' in ens.columns

    # Ratio mode: rank-based fusion for robustness
    if args.top_ratio is not None and args.top_ratio > 0:
        k = max(1, int(n * min(1.0, float(args.top_ratio))))
        ent_rank = ens['mean_entropy'].rank(pct=True) if has_ent else 0.5
        dis_rank = ens['disagreement_jaccard'].rank(pct=True) if has_dis else 0.5
        # Difficulty: entropy weighs more than disagreement by default
        diff_score = 0.7 * ent_rank + 0.3 * dis_rank
        ens['_diff_score'] = diff_score
        out = ens.sort_values('_diff_score', ascending=False).head(k).drop(columns=['_diff_score'])
    else:
        # Quantile mode
        parts = []
        if has_ent:
            ent_th = ens['mean_entropy'].quantile(args.entropy_q)
            parts.append(ens[ens['mean_entropy'] >= ent_th])
        if has_dis:
            dis_th = ens['disagreement_jaccard'].quantile(args.disagree_q)
            parts.append(ens[ens['disagreement_jaccard'] >= dis_th])
        if parts:
            out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['image_id'])
        else:
            out = ens.copy()

    out.to_csv(args.out_csv, index=False)
    print(f"Difficult samples saved to {args.out_csv}; count={len(out)} of {n}")


if __name__ == '__main__':
    main()
