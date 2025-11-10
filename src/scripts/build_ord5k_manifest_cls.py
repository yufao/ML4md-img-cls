import argparse
import os
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Build ORD5K classification manifest')
    parser.add_argument('--csv', required=True, help='Path to full_df.csv')
    parser.add_argument('--images_dir', required=True, help='Path to preprocessed_images directory')
    parser.add_argument('--out', default='manifests/ord5k_cls.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out_rows = []
    label_cols = ['N','D','G','C','A','H','M','O']
    img_dir = args.images_dir
    for _, r in df.iterrows():
        fname = r.get('filename')
        if not isinstance(fname, str):
            continue
        img_path = os.path.join(img_dir, fname)
        if not os.path.isfile(img_path):
            continue
        # prefer explicit 0/1 columns; fallback to parsing target
        try:
            vec = [int(r.get(c, 0)) for c in label_cols]
            # sanity: all 0? try target
            if sum(vec) == 0 and isinstance(r.get('target'), str):
                alt = json.loads(r.get('target'))
                if len(alt) == 8:
                    vec = alt
        except Exception:
            try:
                alt = json.loads(r.get('target', '[]'))
                if len(alt) == 8:
                    vec = alt
                else:
                    continue
            except Exception:
                continue
        out_rows.append({
            'sample_id': r.get('ID', fname),
            'filename': fname,
            'image_path': img_path,
            **{c: vec[i] for i, c in enumerate(label_cols)},
            'target_vec': json.dumps(vec)
        })

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Manifest saved to {args.out}, rows={len(out_df)}")


if __name__ == '__main__':
    main()
