"""将困难样本的原图复制到指定目录，便于人工复核。

适配单标签/多标签管线产出的困难样本 CSV。
若困难 CSV 中没有 `image_path` 列，可提供 `--meta_csv` 以通过 `image_id` 进行路径映射。

示例：
python -m src.scripts.export_difficult_images \
    --difficult_csv outputs/nih14_smoke/difficult.csv \
    --meta_csv rawig/NIH/nih_manifest.csv \
    --out_dir exports/nih14_smoke_difficult \
    --img_root rawig/NIH \
    --id_col image_id --path_col image_path

可选：`--limit 100` 仅拷贝前 100 张做快速核验。
"""
import os
import argparse
import shutil
import pandas as pd


def resolve_path(p: str, img_root: str | None) -> str:
    if not isinstance(p, str):
        return ''
    if os.path.isabs(p):
        return p
    if img_root:
        return os.path.join(img_root, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--difficult_csv', required=True, help='CSV with difficult samples (must contain image_id; image_path optional)')
    ap.add_argument('--out_dir', required=True, help='Directory to copy images into')
    ap.add_argument('--meta_csv', default='', help='Manifest CSV to map image_id -> image_path when difficult CSV lacks image_path')
    ap.add_argument('--img_root', default='', help='Optional root to join with relative image_path')
    ap.add_argument('--id_col', default='image_id')
    ap.add_argument('--path_col', default='image_path')
    ap.add_argument('--limit', type=int, default=0, help='Copy at most N images for quick verification (0 means all)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.difficult_csv)

    # Keep only unique ids
    id_col = args.id_col
    path_col = args.path_col

    if path_col in df.columns and df[path_col].notna().any():
        work = df[[id_col, path_col]].dropna().drop_duplicates(id_col)
    else:
        assert args.meta_csv, 'meta_csv is required when difficult CSV has no image_path'
        meta = pd.read_csv(args.meta_csv)
        assert id_col in meta.columns, f"{id_col} not in meta"
        assert path_col in meta.columns, f"{path_col} not in meta"
        work = df[[id_col]].drop_duplicates(id_col).merge(meta[[id_col, path_col]], on=id_col, how='left')

    # Resolve to absolute/usable paths
    work['__src'] = work[path_col].apply(lambda p: resolve_path(p, args.img_root))
    work = work[work['__src'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]

    total = len(work)
    if args.limit and args.limit > 0:
        work = work.head(args.limit)

    copied = 0
    for _, row in work.iterrows():
        src = row['__src']
        base = os.path.basename(src)
        dst = os.path.join(args.out_dir, base)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            pass

    print(f"Copied {copied}/{total} images to {args.out_dir}")


if __name__ == '__main__':
    main()
