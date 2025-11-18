"""List and clean training output runs under a root directory.

Usage:
  # List runs (default root=outputs)
  python -m src.scripts.clean_outputs --list --root outputs

  # Delete a specific run directory under root (e.g., outputs/ord5k_cls)
  python -m src.scripts.clean_outputs --delete ord5k_cls --root outputs --yes

  # Delete multiple runs by name
  python -m src.scripts.clean_outputs --delete ord5k_cls --delete ord5k_fast --root outputs --yes
"""
import os
import argparse
import shutil
import time
from typing import List, Tuple


def human_size(nbytes: int) -> str:
    units = ['B','KB','MB','GB','TB']
    s = float(nbytes)
    i = 0
    while s >= 1024 and i < len(units)-1:
        s /= 1024.0
        i += 1
    return f"{s:.1f} {units[i]}"


def dir_size(path: str) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def list_runs(root: str) -> List[Tuple[str, int, float, int]]:
    if not os.path.exists(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            try:
                size = dir_size(p)
                mtime = os.path.getmtime(p)
                file_count = sum(len(files) for _, _, files in os.walk(p))
                out.append((name, size, mtime, file_count))
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs', help='Root directory containing runs')
    ap.add_argument('--list', action='store_true', help='List all runs')
    ap.add_argument('--delete', action='append', default=[], help='Run name(s) to delete under root')
    ap.add_argument('--yes', action='store_true', help='Do not prompt for confirmation')
    args = ap.parse_args()

    if args.list or not args.delete:
        runs = list_runs(args.root)
        if not runs:
            print(f'No runs under {args.root}')
        else:
            print(f'Runs under {args.root}:')
            for name, size, mtime, file_count in runs:
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                print(f'  - {name:20s}  size={human_size(size):>8s}  files={file_count:5d}  mtime={t}')

    if args.delete:
        for name in args.delete:
            path = os.path.join(args.root, name)
            if not os.path.isdir(path):
                print(f'[Skip] Not a directory: {path}')
                continue
            if not args.yes:
                ans = input(f'Confirm delete {path}? [y/N] ').strip().lower()
                if ans != 'y':
                    print('[Skip] User canceled')
                    continue
            try:
                shutil.rmtree(path)
                print(f'Deleted {path}')
            except Exception as e:
                print(f'Failed to delete {path}: {e}')


if __name__ == '__main__':
    main()
