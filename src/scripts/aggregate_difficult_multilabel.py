"""多标签困难样本聚合脚本。

支持两种选取模式：
1) 比例模式（推荐）：根据综合难度得分选取前 R（R=top_ratio×N）。
   综合难度得分 = 0.7×熵(分位排名) + 0.3×分歧(分位排名)，分数越高越难。
2) 分位数模式（兜底）：满足 (mean_entropy ≥ entropy_q) 或 (disagreement_jaccard ≥ disagree_q) 的样本并集合去重。

输出行为：
- 默认：仅输出被选中的困难样本；增加列 is_difficult=1。
- 若传 --full_with_flag：输出整表，同时为非困难样本补 is_difficult=0，便于后续二次筛选或统计。

新增：--score_export 可保存每个样本的综合难度得分（比例模式），列名 diff_score。
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
    ap.add_argument('--top_ratio', type=float, default=None, help='按难度得分选择前比例 (0~1]，提供则覆盖分位模式')
    ap.add_argument('--full_with_flag', action='store_true', help='输出整表并带 is_difficult 标记')
    ap.add_argument('--score_export', action='store_true', help='在比例模式下导出 diff_score 列')
    args = ap.parse_args()

    ens = pd.read_csv(args.ensemble_csv)
    n = len(ens)
    if n == 0:
        pd.DataFrame([]).to_csv(args.out_csv, index=False)
        print(f"Difficult samples saved to {args.out_csv}; count=0 of 0")
        return

    has_ent = 'mean_entropy' in ens.columns
    has_dis = 'disagreement_jaccard' in ens.columns

    # 比例模式：基于排名的分数融合，提高不同尺度特征的稳健性
    if args.top_ratio is not None and args.top_ratio > 0:
        k = max(1, int(n * min(1.0, float(args.top_ratio))))
        ent_rank = ens['mean_entropy'].rank(pct=True) if has_ent else 0.5
        dis_rank = ens['disagreement_jaccard'].rank(pct=True) if has_dis else 0.5
        # 难度加权：默认熵权重较高（信息不确定性优先），可视需要调整
        diff_score = 0.7 * ent_rank + 0.3 * dis_rank
        ens['diff_score'] = diff_score
        selected = ens.sort_values('diff_score', ascending=False).head(k)
        selected = selected.copy()
        selected['is_difficult'] = 1
        if args.full_with_flag:
            out = ens.copy()
            # 补充标记
            out['is_difficult'] = 0
            out.loc[selected.index, 'is_difficult'] = 1
        else:
            out = selected
        if not args.score_export and 'diff_score' in out.columns:
            # 若不导出得分且非整表，仅保留困难样本但去掉 diff_score
            if not args.full_with_flag:
                out = out.drop(columns=['diff_score'])
    else:
        # 分位数模式：分别按熵/分歧阈值挑选，再集合去重
        parts = []
        if has_ent:
            ent_th = ens['mean_entropy'].quantile(args.entropy_q)
            parts.append(ens[ens['mean_entropy'] >= ent_th])
        if has_dis:
            dis_th = ens['disagreement_jaccard'].quantile(args.disagree_q)
            parts.append(ens[ens['disagreement_jaccard'] >= dis_th])
        if parts:
            sel = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['image_id'])
        else:
            sel = ens.copy()
        sel = sel.copy()
        sel['is_difficult'] = 1
        if args.full_with_flag:
            out = ens.copy()
            out['is_difficult'] = 0
            out.loc[sel.index, 'is_difficult'] = 1
        else:
            out = sel

    out.to_csv(args.out_csv, index=False)
    print(f"Difficult samples saved to {args.out_csv}; difficult_count={(out['is_difficult']==1).sum()} total={n} full_mode={args.full_with_flag}")


if __name__ == '__main__':
    main()
