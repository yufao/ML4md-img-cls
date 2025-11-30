"""多标签分类数据集（稳健的 CSV 映射与路径解析）。

支持：
- 显式标签列（每类一列）或单列 JSON 向量（labels_json）
- 基于别名自动匹配 image_id / image_path / patient_id 列
- 灰度/彩色混合图像，统一为指定模式（'rgb3' 或 'gray1'）
"""
import os
from typing import Optional, Any, Dict, List, Sequence
import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

DEFAULT_ALIASES = {
    'image_id': ['image_id', 'id', 'uid', 'img_id', 'filename', 'name', 'Image Index'],
    'image_path': ['image_path', 'path', 'file', 'filepath', 'image', 'img_path', 'img'],
    'patient_id': ['patient_id', 'patient', 'case_id', 'pid', 'subject', 'group', 'Patient ID'],
}

def _find_first_col(df: pd.DataFrame, candidates: Sequence[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_transforms(image_size: int = 224, train: bool = True, mode: str = 'rgb3', aug_strategy: str = 'default') -> T.Compose:
    if mode == 'rgb3':
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    else:
        norm_mean = [0.5]
        norm_std = [0.25]
    t_list: List[Any] = [T.Resize((image_size, image_size))]
    
    if train:
        if aug_strategy == 'none':
            # 保守策略：无几何变换，仅色彩抖动
            t_list += [T.ColorJitter(brightness=0.1, contrast=0.1)]
        elif aug_strategy == 'fundus':
            # 眼底图：垂直翻转安全，水平翻转破坏左右眼特征
            t_list += [T.RandomVerticalFlip(), T.RandomRotation(10)]
        elif aug_strategy == 'cxr':
            # 胸片：不翻转（重力依赖），仅微小旋转
            t_list += [T.RandomRotation(5)]
        elif aug_strategy == 'mri':
            # 脑部MRI：水平翻转安全且推荐（增加左右脑病灶多样性）
            t_list += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
        else:
            # 默认通用策略 (default)
            t_list += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
            
    t_list += [T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)]
    return T.Compose(t_list)


class MultiLabelImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        images_root: str = '',
        transform: Optional[Any] = None,
        mode: str = 'rgb3',
        id_col: Optional[str] = None,
        path_col: Optional[str] = None,
        patient_col: Optional[str] = None,
        label_cols: Optional[Sequence[str]] = None,
        labels_json_col: Optional[str] = None,
        aug_strategy: str = 'default',
    ):
        if dataframe is not None:
            self.df = dataframe.copy().reset_index(drop=True)
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Provide either csv_path or dataframe")
        self.images_root = images_root
        self.transform = transform
        self.mode = mode
        self.aug_strategy = aug_strategy
        self.id_col = id_col or _find_first_col(self.df, DEFAULT_ALIASES['image_id'])
        self.path_col = path_col or _find_first_col(self.df, DEFAULT_ALIASES['image_path'])
        self.patient_col = patient_col or _find_first_col(self.df, DEFAULT_ALIASES['patient_id'])
        if self.id_col is None or self.path_col is None:
            raise ValueError(f"Missing required id/path columns. Have: {list(self.df.columns)}")

        # 确定标签来源（显式列 or JSON 向量列）
        self.label_cols: Optional[List[str]] = None
        self.labels_json_col = None
        if label_cols:
            cols_ok = [c for c in label_cols if c in self.df.columns]
            if not cols_ok:
                raise ValueError(f"None of label_cols exist in CSV: {label_cols}")
            self.label_cols = list(cols_ok)
        elif labels_json_col and labels_json_col in self.df.columns:
            self.labels_json_col = labels_json_col
        else:
            # 自动识别 NIH14 常见列
            nih_classes = [
                'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax',
                'Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
            ]
            cols_ok = [c for c in nih_classes if c in self.df.columns]
            if cols_ok:
                self.label_cols = cols_ok
            else:
                # 兜底：尝试 'labels_json' 列
                if 'labels_json' in self.df.columns:
                    self.labels_json_col = 'labels_json'
                else:
                    raise ValueError('Cannot find label columns or labels_json column')

        # 推断类别数
        if self.label_cols is not None:
            self.num_classes = len(self.label_cols)
        else:
            # assume at least one row has a JSON list
            for i in range(min(32, len(self.df))):
                v = self.df.iloc[i][self.labels_json_col]
                try:
                    arr = json.loads(v)
                    if isinstance(arr, (list, tuple)):
                        self.num_classes = len(arr)
                        break
                except Exception:
                    pass
            if not hasattr(self, 'num_classes'):
                raise ValueError('Unable to infer num_classes from labels_json column')

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        c1 = os.path.normpath(os.path.join(self.images_root, rel_path))
        c2 = os.path.normpath(os.path.join(self.images_root, os.path.basename(rel_path)))
        if os.path.exists(c1):
            return c1
        if os.path.exists(c2):
            return c2
        if os.path.exists(rel_path):
            return rel_path
        return c1

    def _open_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode not in ['L', 'RGB']:
            img = img.convert('L') if img.mode != 'RGB' else img
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        img_path = self._resolve_path(rel_path)
        img = self._open_image(img_path)
        if self.mode == 'rgb3':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        elif self.mode == 'gray1':
            if img.mode != 'L':
                img = img.convert('L')
        else:
            raise ValueError("mode must be 'rgb3' or 'gray1'")

        if self.transform:
            img = self.transform(img)

        if self.label_cols is not None:
            labels = row[self.label_cols].astype(float).values
        else:
            try:
                labels = np.array(json.loads(row[self.labels_json_col]), dtype=np.float32)
            except Exception:
                labels = np.zeros(self.num_classes, dtype=np.float32)

        # Handle patient id optional
        patient_id = row[self.patient_col] if (self.patient_col and self.patient_col in self.df.columns) else ''

        return {
            'image': img,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'image_id': row[self.id_col],
            'patient_id': patient_id,
        }
