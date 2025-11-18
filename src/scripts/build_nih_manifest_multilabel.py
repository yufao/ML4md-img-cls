"""构建 NIH14 多标签任务的清单 CSV（manifest）。

输入原始 `Data_Entry_2017.csv` 与图像根目录（包含 images_001..images_012），
自动为每张图解析“Finding Labels”为 14 个一热列，并解析绝对 `image_path` 与 `patient_id`。
支持可选 `labels_json` 列（便于简洁传输标签向量）。
"""
import os
import argparse
import json
import pandas as pd


NIH_CLASSES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
]


def find_image_path(images_root: str, filename: str) -> str:
    """在 NIH 的 12 个子目录中查找给定文件名的完整路径。"""
    for i in range(1, 13):
        d = os.path.join(images_root, f"images_{i:03d}", "images", filename)
        if os.path.exists(d):
            return d
    return ''


def main():
    ap = argparse.ArgumentParser(description='构建 NIH14 多标签清单 CSV')
    ap.add_argument('--nih_csv', required=True, help='Path to Data_Entry_2017.csv')
    ap.add_argument('--images_root', required=True, help='Path containing images_001..images_012')
    ap.add_argument('--out', default='manifests/nih14_multilabel.csv')
    ap.add_argument('--include_json_vec', action='store_true', help='Include labels_json column')
    args = ap.parse_args()

    df = pd.read_csv(args.nih_csv)
    rows = []
    for _, r in df.iterrows():
        fname = r['Image Index']
        labels_str = str(r['Finding Labels'])
        # Parse labels
        labels = [0]*len(NIH_CLASSES)
        if labels_str and labels_str != 'No Finding':
            for token in labels_str.split('|'):
                if token in NIH_CLASSES:
                    labels[NIH_CLASSES.index(token)] = 1
        p = find_image_path(args.images_root, fname)
        if not p:
            continue
        row = {
            'image_id': fname,
            'image_path': p,
            'patient_id': r.get('Patient ID', ''),
        }
        for i, c in enumerate(NIH_CLASSES):
            row[c] = labels[i]
        if args.include_json_vec:
            row['labels_json'] = json.dumps(labels)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Manifest saved to {args.out}, rows={len(out_df)}")


if __name__ == '__main__':
    main()
