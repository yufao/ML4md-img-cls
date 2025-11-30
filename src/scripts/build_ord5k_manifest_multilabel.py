import argparse
import pandas as pd
from pathlib import Path

"""
从原始 ORD5K full_df.csv 生成多标签分类 manifest：
输出列：sample_id, image_path, N,D,G,C,A,H,M,O
- 自动从常见列推断出图像相对路径（支持 filename/filepath/image_path）
- 若存在 ID 列，用作 sample_id；否则用行号
- 标签列按 [N,D,G,C,A,H,M,O] 查找，缺失则置 0
"""

LABELS = ['N','D','G','C','A','H','M','O']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='raw full_df.csv')
    ap.add_argument('--images_dir', required=True, help='root dir for images (relative paths allowed)')
    ap.add_argument('--out', required=True, help='output manifest csv path')
    return ap.parse_args()


def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    cols = list(df.columns)
    id_col = find_col(cols, ['sample_id','ID','id','uid','image_id'])
    path_col = find_col(cols, ['image_path','filepath','path','file','filename','image','img_path'])
    if path_col is None:
        raise ValueError('Cannot find path column in full_df.csv')

    out = pd.DataFrame()
    out['sample_id'] = df[id_col] if id_col is not None else df.index.astype(str)
    out['image_path'] = df[path_col].astype(str)

    for l in LABELS:
        if l in df.columns:
            out[l] = (df[l].fillna(0).astype(float) > 0).astype(int)
        else:
            out[l] = 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Wrote {len(out)} rows to {args.out}')


if __name__ == '__main__':
    main()
