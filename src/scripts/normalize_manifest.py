import argparse
import os
import pandas as pd
import yaml

DEFAULT_MAP = {
    'image_id': ['image_id','id','uid','img_id','filename','name'],
    'image_path': ['image_path','path','file','filepath','image','img_path','img'],
    'label': ['label','class','target','y','category'],
    'patient_id': ['patient_id','patient','case_id','pid','subject','group'],
}


def find_col(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None


def normalize(df: pd.DataFrame, alias: dict):
    out = {}
    for std_col, candidates in alias.items():
        src = find_col(df, candidates)
        if src is not None:
            out[std_col] = df[src]
        else:
            if std_col in ('patient_id',):
                out[std_col] = None
            else:
                raise ValueError(f"Required column '{std_col}' not found from candidates {candidates}")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--images_root', default='')
    ap.add_argument('--yaml_map', default='')
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    alias = DEFAULT_MAP.copy()
    if args.yaml_map and os.path.exists(args.yaml_map):
        with open(args.yaml_map, 'r') as f:
            custom = yaml.safe_load(f) or {}
        for k, v in custom.items():
            alias[k] = v if isinstance(v, list) else [v]
    out = normalize(df, alias)
    if args.images_root:
        # keep relative paths if provided; validation happens in dataset
        pass
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote normalized manifest to {args.out_csv} with columns {list(out.columns)}")

if __name__ == '__main__':
    main()
