"""折分工具：支持患者级分组的分层 K 折。

主要函数：
- stratified_group_kfold(df, y_col, group_col, n_splits, seed)
    若不可用 StratifiedGroupKFold 则回退为 GroupKFold。

- multilabel_stratified_group_kfold(df, y_cols, group_col, n_splits, seed)
    基于组的贪心分配策略，使各折在每个类别上的正例计数尽量均衡。
"""
from typing import Iterator, Tuple, List
import numpy as np
import pandas as pd

def stratified_group_kfold(df: pd.DataFrame, y_col: str, group_col: str, n_splits: int, seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df[y_col].values
        g = df[group_col].values
        for tr_idx, va_idx in sgkf.split(df, y, g):
            yield tr_idx, va_idx
    except Exception:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        groups = df[group_col].values
        for tr_idx, va_idx in gkf.split(df, groups=groups):
            yield tr_idx, va_idx


def multilabel_stratified_group_kfold(
    df: pd.DataFrame,
    y_cols: List[str],
    group_col: str,
    n_splits: int,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """多标签、组感知的贪心折分。

    算法：
    - 先按组对标签向量求和，得到每个组的标签计数
    - 按“总正例数”对组做降序排序
    - 依次将每个组分配到“当前与期望计数偏差最小”的折中（平方误差）
    """
    rng = np.random.RandomState(seed)
    groups = df[group_col].astype(str).values
    uniq_groups = np.array(sorted(pd.unique(groups)))
    # Map group -> indices
    group_to_indices = {g: np.where(groups == g)[0] for g in uniq_groups}
    # Group label vectors
    Y = df[y_cols].values.astype(np.float32)
    group_label_sum = {g: Y[group_to_indices[g]].sum(axis=0) for g in uniq_groups}
    total_label_sum = np.stack(list(group_label_sum.values()), axis=0).sum(axis=0)
    expected_per_fold = total_label_sum / float(n_splits)

    # Order groups by total positives
    order = np.argsort([-group_label_sum[g].sum() for g in uniq_groups])
    ordered_groups = uniq_groups[order]

    # Initialize folds
    folds = [set() for _ in range(n_splits)]
    fold_label_sum = [np.zeros_like(expected_per_fold) for _ in range(n_splits)]

    for g in ordered_groups:
        costs = []
        gvec = group_label_sum[g]
        for k in range(n_splits):
            new_sum = fold_label_sum[k] + gvec
            cost = ((new_sum - expected_per_fold) ** 2).sum()
            costs.append(cost)
        # Random tie-break
        min_cost = np.min(costs)
        candidate = [i for i, c in enumerate(costs) if np.isclose(c, min_cost)]
        k_best = rng.choice(candidate)
        folds[k_best].add(g)
        fold_label_sum[k_best] = fold_label_sum[k_best] + gvec

    # Convert to indices
    group_to_fold = {}
    for k, gset in enumerate(folds):
        for g in gset:
            group_to_fold[g] = k

    idx = np.arange(len(df))
    for k in range(n_splits):
        va_groups = {g for g, f in group_to_fold.items() if f == k}
        va_idx = np.concatenate([group_to_indices[g] for g in va_groups]) if va_groups else np.array([], dtype=int)
        tr_idx = np.setdiff1d(idx, va_idx)
        yield tr_idx, va_idx
