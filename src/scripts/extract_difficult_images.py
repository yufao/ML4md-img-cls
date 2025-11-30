import os
import csv
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

"""
从 difficult.csv 中筛选 is_difficult==1 的样本，将对应图像复制到目标目录。
- 兼容单标签与多标签 difficult.csv（要求包含 image_path 或 image_id / filename 之一）。
- 若存在 class_name / class_index 列且传入 --with_classes，则按类名/索引分子目录输出。
- 默认保持原文件名，不改扩展名。
- 针对部分 CSV 仅提供“裸 id”(无扩展名、无子目录) 的情况，提供多级回退：
    1) 直接按 path_col + img_root 拼接；
    2) 若失败，在 img_root 下 rglob 精确名称；
    3) 若仍失败且无扩展名，rglob 搜索 stem.* （匹配任意扩展名）

示例：
python -m src.scripts.extract_difficult_images \
    --csv outputs/brain_tumor_resnet50/difficult.csv \
    --img_root rawig/MRItumor \
    --out_dir outputs/brain_tumor_resnet50/difficult_images \
    --with_classes
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='difficult.csv 路径')
    ap.add_argument('--img_root', default='', help='图片根目录（若 CSV 提供相对路径或仅文件名时需要）')
    ap.add_argument('--out_dir', required=True, help='复制困难样本的输出目录')
    ap.add_argument('--with_classes', action='store_true', help='按类名/索引分子目录')
    ap.add_argument('--path_col', default='image_path', help='CSV 中路径列名（若缺失将回退到 image_id/filename）')
    ap.add_argument('--is_difficult_col', default='', help='优先使用的困难样本标识列名；为空时自动检测')
    ap.add_argument('--class_col', default='', help='优先使用的类别列名；为空时自动检测')
    ap.add_argument('--manifest_csv', default='', help='原始完整 manifest（含 image_id 与 image_path）用于补全路径')
    ap.add_argument('--manifest_id_col', default='image_id', help='manifest 中 ID 列名或 sample_id')
    ap.add_argument('--manifest_path_col', default='image_path', help='manifest 中路径列名')
    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_col(header, candidates):
    for c in candidates:
        if c in header:
            return c
    return None


def _find_source(source_path: Path, img_root: Path) -> Optional[Path]:
    """
    Tries to find the source image file using multiple strategies.
    The order of strategies is:
    1. Check if the path from the CSV exists as is (relative to current dir).
    2. Check if the path is absolute.
    3. Try to combine img_root and the path from the CSV.
    4. Fallback to searching for the filename recursively within img_root.
    """
    # Strategy 1: Path from CSV is valid as-is (e.g., relative from project root)
    if source_path.exists():
        return source_path

    # Strategy 2: Path from CSV is absolute
    if source_path.is_absolute() and source_path.exists():
        return source_path

    # Strategy 3: Path from CSV is relative to img_root
    direct_path = img_root / source_path
    if direct_path.exists():
        return direct_path

    # --- Fallback Strategies ---
    # Strategy 4: Search for the filename recursively within img_root
    # This is useful if the CSV only contains the filename, not the full path.
    filename = source_path.name
    found_files = list(img_root.rglob(filename))
    if found_files:
        if len(found_files) > 1:
            print(f"警告: 找到多个同名文件 '{filename}'，将使用第一个: {found_files[0]}")
        return found_files[0]

    # Strategy 5: If filename has no extension, search with a wildcard.
    # This helps with cases where the CSV has 'image_123' and file is 'image_123.jpg'
    if not source_path.suffix:
        base_name = source_path.stem
        found_files = list(img_root.rglob(f"{base_name}.*"))
        if found_files:
            if len(found_files) > 1:
                print(f"警告: 找到多个同名文件 (不同后缀) '{base_name}.*'，将使用第一个: {found_files[0]}")
            return found_files[0]

    return None


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    img_root = Path(args.img_root) if args.img_root else None
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    manifest_df = None
    manifest_map = {}
    if args.manifest_csv:
        try:
            manifest_df = __import__('pandas').read_csv(args.manifest_csv)
            idc = args.manifest_id_col if args.manifest_id_col in manifest_df.columns else (
                'sample_id' if 'sample_id' in manifest_df.columns else 'image_id'
            )
            pc = args.manifest_path_col if args.manifest_path_col in manifest_df.columns else (
                'image_path' if 'image_path' in manifest_df.columns else None
            )
            if pc is None:
                raise ValueError('manifest 缺失路径列，无法建立映射')
            manifest_df[idc] = manifest_df[idc].astype(str)
            manifest_df[pc] = manifest_df[pc].astype(str)
            manifest_map = dict(zip(manifest_df[idc], manifest_df[pc]))
            print(f"Loaded manifest mapping: {len(manifest_map)} ids")
        except Exception as e:
            print(f"警告: 读取 manifest 失败: {e}")

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

        difficult_col = args.is_difficult_col if args.is_difficult_col and args.is_difficult_col in header else find_col(
            header, ['is_difficult', 'difficult', 'hard']
        )
        if difficult_col is None:
            raise ValueError('difficult.csv 必须包含 is_difficult/difficult/hard 列, 或通过 --is_difficult_col 指定')

        path_col = args.path_col if args.path_col in header else find_col(
            header, ['image_path', 'filepath', 'path', 'file', 'image', 'img_path']
        )
        if path_col is None:
            path_col = find_col(header, ['image_id', 'id', 'filename', 'Image Index'])
            if path_col is None:
                raise ValueError('无法找到路径或文件名列')

        class_col = args.class_col if (args.class_col and args.class_col in header) else find_col(
            header, ['class_name', 'class', 'label', 'class_index']
        )

        n_total = 0
        n_selected = 0
        copied_count = 0

        for row in reader:
            n_total += 1
            flag = row.get(difficult_col, '0')
            is_diff = str(flag).strip() in ['1', 'true', 'True', 'YES', 'yes']
            if not is_diff:
                continue
            n_selected += 1

            rel = str(row.get(path_col, '')).strip()
            # 清洗类似 tensor(123) 的伪字符串
            if rel.startswith('tensor(') and rel.endswith(')'):
                rel_clean = rel[len('tensor('):-1]
                rel = rel_clean.strip()
            # 如果没有路径列或路径看起来像纯数字/索引，则尝试用 manifest 映射补全
            if manifest_map:
                # 条件：当前 rel 不包含路径分隔符且没有扩展名
                if ('/' not in rel and '\\' not in rel and '.' not in Path(rel).name) or rel in manifest_map:
                    mapped = manifest_map.get(rel)
                    if mapped:
                        rel = mapped
            if not rel:
                continue

            src = _find_source(Path(rel), img_root)
            if src is None:
                print(f"警告: 未找到文件 (多级回退均失败): {rel}")
                continue

            if args.with_classes and class_col is not None:
                sub = str(row.get(class_col, 'unknown')).strip() or 'unknown'
                dest_dir = out_dir / sub
            else:
                dest_dir = out_dir
            ensure_dir(dest_dir)
            dest = dest_dir / src.name
            try:
                shutil.copy2(src, dest)
                copied_count += 1
            except Exception as e:
                print(f"警告: 复制失败 {src} -> {dest}: {e}")

    print(f"Done. selected={n_selected} / total={n_total}, copied={copied_count}, out={args.out_dir}")


if __name__ == '__main__':
    main()
