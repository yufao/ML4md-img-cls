import os
import argparse
import pandas as pd
from glob import glob


def parse_args():
    ap = argparse.ArgumentParser(description="Build manifest CSV from images/ and optional masks/ directory")
    ap.add_argument("--images", type=str, required=True, help="images directory")
    ap.add_argument("--masks", type=str, default=None, help="masks directory (optional)")
    ap.add_argument("--exts", type=str, nargs="+", default=[".png", ".jpg", ".jpeg", ".tif"])
    ap.add_argument("--out", type=str, required=True, help="output CSV path")
    ap.add_argument("--id-from", type=str, choices=["stem","name"], default="stem", help="id generation")
    ap.add_argument("--mask-suffix", type=str, default=None, help="suffix for mask naming (e.g. _mask)")
    return ap.parse_args()


def main():
    args = parse_args()
    img_files = []
    for ext in args.exts:
        img_files.extend(glob(os.path.join(args.images, f"**/*{ext}"), recursive=True))
    img_files = sorted(list(set(img_files)))
    rows = []
    for ip in img_files:
        stem = os.path.splitext(os.path.basename(ip))[0]
        sid = stem if args.id_from == "stem" else os.path.basename(ip)
        mp = None
        if args.masks:
            if args.mask_suffix:
                mname = stem + args.mask_suffix
                candidates = []
                for ext in args.exts:
                    candidates.append(os.path.join(args.masks, f"{mname}{ext}"))
                    candidates.append(os.path.join(args.masks, f"{stem}{ext}"))
                mp = next((p for p in candidates if os.path.exists(p)), None)
            else:
                for ext in args.exts:
                    p = os.path.join(args.masks, f"{stem}{ext}")
                    if os.path.exists(p):
                        mp = p
                        break
        rows.append({"id": sid, "image_path": os.path.abspath(ip), "mask_path": os.path.abspath(mp) if mp else None})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote manifest: {args.out}, rows={len(df)}")


if __name__ == "__main__":
    main()
