import os
import csv
import re
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


def derive_patient_id(fname: str) -> str:
    stem = Path(fname).stem
    base = re.sub(r'[_\- ]?\d+$', '', stem) or stem
    return hashlib.md5(base.encode('utf-8')).hexdigest()[:12]


def collect_class_dirs(root: Path) -> List[Path]:
    return [d for d in root.iterdir() if d.is_dir()]


def scan_folder(img_root: Path, split_name: str = "", class_to_idx: Optional[dict] = None):
    rows = []
    if class_to_idx is None:
        class_names = sorted([d.name for d in collect_class_dirs(img_root)])
        class_to_idx = {c: i for i, c in enumerate(class_names)}
    else:
        class_names = sorted(class_to_idx.keys())
    for cls in class_names:
        cls_dir = img_root / cls
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                image_id = p.stem
                class_name = cls
                class_index = class_to_idx[cls]
                patient_id = derive_patient_id(p.name)
                rows.append([
                    image_id,
                    str(p),
                    class_name,
                    class_index,
                    patient_id,
                    split_name,
                ])
    return rows, class_to_idx


def make_relative(path: str, relative_to: Optional[Path]) -> str:
    if not relative_to:
        return path
    try:
        return str(Path(path).resolve().relative_to(relative_to.resolve()))
    except Exception:
        return path


def build_manifest(img_root: str, out_csv: str, relative_to: Optional[str]):
    root = Path(img_root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {img_root}")

    # 支持两种布局：
    # 1) 顶层直接是类别目录
    # 2) 顶层含 Training/Validation/Testing 等，再下一层是类别
    split_dirs = {}
    for d in root.iterdir():
        if d.is_dir() and d.name.lower() in {"train", "training", "val", "valid", "validation", "test", "testing"}:
            split_dirs[d.name.lower()] = d

    rows = []
    if split_dirs:
        # 先收集所有 split 下的类名做全局映射，避免各子集类索引不一致
        all_class_names = set()
        for d in split_dirs.values():
            for cd in collect_class_dirs(d):
                all_class_names.add(cd.name)
        class_to_idx = {c: i for i, c in enumerate(sorted(all_class_names))}
        for name, d in split_dirs.items():
            part_rows, _ = scan_folder(d, name, class_to_idx=class_to_idx)
            rows.extend(part_rows)
    else:
        part_rows, class_to_idx = scan_folder(root, "")
        rows.extend(part_rows)

    if not rows:
        raise SystemExit(f"No images found under {img_root}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rel_base = Path(relative_to) if relative_to else None

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'image_path', 'class_name', 'class_index', 'patient_id', 'split'])
        for r in rows:
            r[1] = make_relative(r[1], rel_base)
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    # 打印类映射以便核对
    try:
        print('Class mapping:', {k: v for k, v in class_to_idx.items()})
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_root', required=True, help='数据根目录，如 rawig/MRItumor')
    ap.add_argument('--out_csv', default='manifests/mri_manifest.csv')
    ap.add_argument('--relative_to', default='.', help='转为相对该目录的相对路径；空则保持原样')
    args = ap.parse_args()

    build_manifest(args.img_root, args.out_csv, args.relative_to or None)


if __name__ == '__main__':
    main()
import os 
import csv
import re
import hashlib
import argparse
from pathlib import Path
import logging
from typing import List,Tuple,Dict,Optional


import pandas as pd

# 别名
SPLIT_DIR_NAMES = {
    'train': {'train', 'training'},
    'val': {'val', 'valid', 'validation'},
    'test': {'test', 'testing'},
}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def guess_split_name(name: str) -> Optional[str]:
    n = name.lower()
    for split, aliases in SPLIT_DIR_NAMES.items():
        if n in aliases:
            return split
    return None



def derive_patient_id(fname:str) -> str:
    """
    根据文件名生成一个相对稳定的“患者/序列”ID:
    - 去掉常见的尾部数字
    - 再对剩余部分做 md5，避免过长/冲突
    """
    stem=Path(fname).stem
    base=re.sub(r'[_\- ]?\d+$', '', stem) or stem
    return hashlib.md5(base.encode('utf-8')).hexdigest()[:12]


def detect_layout(root:Path) -> Tuple[bool,Dict[str,Path]]:
    """
    返回二元组 (has_split_level, split_dir_map)
    - 若顶层含有 train/val/test 之类目录，返回(True ，映射)
    - 否则返回 (False ,空映射)
    """
    split_map={}
    for d in root.iterdir():
        if  d.is_dir():
            s=guess_split_name(d.name)
            if s:
                split_map[s] = d
    return (len(split_map)>0), split_map

def collect_class_dirs(base: Path) -> List[Path]:
    return [d for d in base.iterdir() if d.is_dir()]


def scan_one_split(split_root: Path, split_name: str, class_to_idx: Optional[Dict[str, int]] = None):
    rows=[]


def scan_one_split(split_root: Path, split_name: str, class_to_idx: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, int], List[Dict]]:
    rows = []
    # 改进1：提前验证 split_root
    if not split_root.is_dir():
        raise ValueError(f"split_root 不是有效目录: {split_root}")
    
    if class_to_idx is None:
        # 改进2：记录自动发现的类别数
        class_names = sorted([d.name for d in collect_class_dirs(split_root)])
        logging.info(f"自动发现 {len(class_names)} 个类别: {class_names}")
        class_to_idx = {c: i for i, c in enumerate(class_names)}
    else:
        class_names = sorted(class_to_idx.keys())

    # 改进3：使用 rglob 的过滤功能减少迭代
    for cls in class_names:
        cls_dir = split_root / cls
        if not cls_dir.is_dir():
            logging.warning(f"类别目录不存在: {cls_dir}")
            continue
        
        # 只匹配图片扩展名，大幅减少文件检查
        for ext in IMG_EXTS:
            for p in cls_dir.rglob(f'*{ext}'):
                # 改进4：处理 .nii.gz 等多后缀情况
                if ext == '.gz' and p.name.endswith('.nii.gz'):
                    ext = '.nii.gz'
                image_id = p.stem.rsplit('.', 1)[0] if ext == '.nii.gz' else p.stem
                # ... 其余字段
                rows.append({
                    'image_id': image_id,
                    'image_path': str(p.resolve()),  # 改进5：使用绝对路径
                    'class_name': cls,
                    'class_index': class_to_idx[cls],
                    'patient_id': derive_patient_id(p.name),
                    'split': split_name
                })
    return class_to_idx, rows

def make_relative(paths: List[str], relative_to: Optional[Path]) -> List[str]:
    '''
    将一组绝对路径转换为目标基准目录的相对路径，失败时回退到原路径。
    '''
    if not relative_to:
        return paths
    rels = []
    for s in paths:
        try:
            rels.append(str(Path(s).resolve().relative_to(relative_to.resolve())))
        except Exception:
            rels.append(s)
    return rels


def build_manifest(img_root: str, out_csv: str, relative_to: Optional[str], with_folds: bool, n_splits: int, seed: int, folds_on: str):
    root = Path(img_root)
    assert root.is_dir(), f'Not a directory: {img_root}'
    has_split, split_map = detect_layout(root)

    rows = []
    class_to_idx = None
    if has_split:
        # 先根据 train（若存在）建立类索引，保持一致性
        if 'train' in split_map:
            class_to_idx, train_rows = scan_one_split(split_map['train'], 'train', class_to_idx=None)
            rows.extend(train_rows)
        # 其他 split 复用映射
        for s in ['val', 'test']:
            if s in split_map:
                _, r = scan_one_split(split_map[s], s, class_to_idx=class_to_idx)
                rows.extend(r)
        # 若只有 val/test 没有 train 的场景，再各自独立扫描
        for s, p in split_map.items():
            if s not in {'train', 'val', 'test'}:
                _, r = scan_one_split(p, s, class_to_idx=class_to_idx)
                rows.extend(r)
    else:
        # 顶层即为类别目录
        class_names = sorted([d.name for d in collect_class_dirs(root)])
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        _, r = scan_one_split(root, 'unknown', class_to_idx=class_to_idx)
        rows.extend(r)

    if not rows:
        raise RuntimeError(f'No images found under {img_root}')

    df = pd.DataFrame(rows, columns=['image_id', 'image_path', 'class_name', 'class_index', 'patient_id', 'split'])

    # 去重（少见但稳妥）
    df = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)

    # 相对路径
    rel_base = Path(relative_to) if relative_to else None
    if rel_base:
        df['image_path'] = make_relative(df['image_path'].tolist(), rel_base)

    # 可选：添加 K 折
    if with_folds:
        if not HAVE_SPLITS:
            raise RuntimeError('src.utils.splits.stratified_group_kfold not available. Set --with_folds=0 or ensure imports work.')
        df['fold'] = -1
        if folds_on not in {'all', 'train'}:
            raise ValueError('--folds_on must be "all" or "train"')
        mask = df['split'].isin(['train']) if (has_split and folds_on == 'train') else pd.Series([True] * len(df))
        sub = df[mask].reset_index()
        # 使用患者分组 + 类别分层
        fold_ids = [-1] * len(sub)
        for k, (tr_idx, va_idx) in enumerate(stratified_group_kfold(sub, y_col='class_index', group_col='patient_id', n_splits=n_splits, seed=seed)):
            sub.loc[va_idx, 'fold'] = k
        df.loc[mask, 'fold'] = sub['fold'].astype(int).values
    
     # 写出
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows to {out_csv}')
    print('Class mapping:', {k: v for k, v in class_to_idx.items()} if class_to_idx else 'unknown')
    if with_folds:
        print('Fold counts:', df['fold'].value_counts(dropna=False).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_root', required=True, help='数据根目录，如 rawig/MRItumor')
    ap.add_argument('--out_csv', default='manifests/mri_manifest.csv')
    ap.add_argument('--relative_to', default='.', help='将 image_path 转成相对该目录的相对路径；设为空则保持绝对路径')
    ap.add_argument('--with_folds', type=int, default=0, help='是否直接在 manifest 中生成 fold 列（0/1）')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds_on', type=str, default='all', help='all 或 train（仅对 train 划分折）')
    args = ap.parse_args()

    rel = args.relative_to if args.relative_to else None
    build_manifest(args.img_root, args.out_csv, rel, bool(args.with_folds), args.folds, args.seed, args.folds_on)


if __name__ == '__main__':
    main()
