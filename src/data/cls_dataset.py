"""Classification dataset with robust CSV column mapping and mixed-channel support.

Features:
- Auto column alias matching for diverse CSV schemas
- Optional explicit column mapping via `column_alias`
- Mixed grayscale/RGB support; converts to requested mode ('rgb3' or 'gray1')
- Optional CT windowing using window_center/window_width
- Lightweight channel stats sampling (first N rows) to inform conversion behavior

This dataset targets single-label classification for binary/multiclass tasks.
"""

import os
from typing import Optional, Any, Dict, List
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

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
            # Conservative strategy: No geometric transforms, only color jitter
            t_list += [T.ColorJitter(brightness=0.1, contrast=0.1)]
        elif aug_strategy == 'fundus':
            t_list += [T.RandomVerticalFlip(), T.RandomRotation(10)]
        elif aug_strategy == 'cxr':
            t_list += [T.RandomRotation(5)]
        elif aug_strategy == 'mri':
            t_list += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
        else:
            t_list += [T.RandomHorizontalFlip(), T.RandomRotation(10)]
            
    t_list += [T.ToTensor(), T.Normalize(mean=norm_mean, std=norm_std)]
    return T.Compose(t_list)

"""
Classification dataset supporting:
- CSV path or pre-loaded DataFrame
- Patient-level metadata (patient_id optional)
- Mixed grayscale (1ch) / RGB (3ch) images unified to 3-channel input
- Optional CT windowing (window_center, window_width columns)

Required columns (minimum):
    image_id, image_path, label
Optional:
    patient_id, window_center, window_width, modality, channels

Modes:
    mode='rgb3'  -> always return 3-channel tensor (grayscale duplicated)
    mode='gray1' -> return single channel tensor (if upstream model first conv changed)

Transforms: pass externally via get_transforms
"""

def ct_window(img: Image.Image, center: float, width: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    low = center - width / 2.0
    high = center + width / 2.0
    arr = (arr - low) / (high - low)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode='L')

DEFAULT_ALIASES = {
    'image_id': ['image_id', 'id', 'uid', 'img_id', 'filename', 'name'],
    'image_path': ['image_path', 'path', 'file', 'filepath', 'image', 'img_path', 'img'],
    'label': ['label', 'class', 'target', 'y', 'category'],
    'patient_id': ['patient_id', 'patient', 'case_id', 'pid', 'subject', 'group'],
    'window_center': ['window_center', 'win_center', 'wc'],
    'window_width': ['window_width', 'win_width', 'ww'],
}

def _find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

class MedicalImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        images_root: str = '',
        transform: Optional[Any] = None,
        mode: str = 'rgb3',
        id_col: Optional[str] = None,
        path_col: Optional[str] = None,
        label_col: Optional[str] = None,
        patient_col: Optional[str] = None,
        apply_ct_window: bool = False,
        ct_center_col: Optional[str] = None,
        ct_width_col: Optional[str] = None,
        column_alias: Optional[Dict[str, Any]] = None,
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
        # resolve columns via explicit args > alias dict > defaults
        alias = DEFAULT_ALIASES.copy()
        if column_alias:
            for k, v in column_alias.items():
                if isinstance(v, str):
                    alias[k] = [v]
                elif isinstance(v, (list, tuple)):
                    alias[k] = list(v)
        self.id_col = id_col or _find_first_col(self.df, alias['image_id'])
        self.path_col = path_col or _find_first_col(self.df, alias['image_path'])
        self.label_col = label_col or _find_first_col(self.df, alias['label'])
        self.patient_col = patient_col or _find_first_col(self.df, alias['patient_id'])

        self.apply_ct_window = apply_ct_window
        self.ct_center_col = ct_center_col or _find_first_col(self.df, alias['window_center'])
        self.ct_width_col = ct_width_col or _find_first_col(self.df, alias['window_width'])

        for col in [self.id_col, self.path_col, self.label_col]:
            if col is None or col not in self.df.columns:
                raise ValueError("Missing required column(s). Mapped columns: "
                                 f"id={self.id_col}, path={self.path_col}, label={self.label_col}")

        # Sample channel stats for early feedback
        self._channel_stats = self._sample_channel_stats(sample_n=min(32, len(self.df)))

    def __len__(self) -> int:
        return len(self.df)

    def _open_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode not in ['L', 'RGB']:
            img = img.convert('L') if img.mode != 'RGB' else img
        return img

    def _sample_channel_stats(self, sample_n: int = 16) -> Dict[str, Any]:
        gray, rgb = 0, 0
        for i in range(min(sample_n, len(self.df))):
            p = self.df.iloc[i][self.path_col]
            # resolve like __getitem__ for accurate sampling
            if os.path.isabs(p):
                rp = p
            else:
                c1 = os.path.normpath(os.path.join(self.images_root, p))
                c2 = os.path.normpath(os.path.join(self.images_root, os.path.basename(p)))
                if os.path.exists(c1):
                    rp = c1
                elif os.path.exists(c2):
                    rp = c2
                elif os.path.exists(p):
                    rp = p
                else:
                    rp = c1
            try:
                with Image.open(rp) as im:
                    m = im.mode
                if m == 'L':
                    gray += 1
                elif m == 'RGB':
                    rgb += 1
            except Exception:
                continue
        return {'gray': gray, 'rgb': rgb, 'sampled': gray + rgb}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        # Robust path resolve: absolute > join(root, rel) > join(root, basename) > rel as-is
        if os.path.isabs(rel_path):
            img_path = rel_path
        else:
            c1 = os.path.normpath(os.path.join(self.images_root, rel_path))
            c2 = os.path.normpath(os.path.join(self.images_root, os.path.basename(rel_path)))
            if os.path.exists(c1):
                img_path = c1
            elif os.path.exists(c2):
                img_path = c2
            elif os.path.exists(rel_path):
                img_path = rel_path
            else:
                # still return c1; open will raise FileNotFoundError which surfaces clearly
                img_path = c1
        img = self._open_image(img_path)

        if self.apply_ct_window and self.ct_center_col in self.df.columns and self.ct_width_col in self.df.columns:
            try:
                c_val = row[self.ct_center_col]
                w_val = row[self.ct_width_col]
                if pd.notna(c_val) and pd.notna(w_val):
                    c = float(c_val); w = float(w_val)
                    if img.mode != 'L':
                        img = img.convert('L')
                    img = ct_window(img, c, w)
            except Exception:
                pass

        if self.mode == 'rgb3':
            if img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
        elif self.mode == 'gray1':
            if img.mode != 'L':
                img = img.convert('L')
        else:
            raise ValueError("mode must be 'rgb3' or 'gray1'")

        label = int(row[self.label_col])
        image_id = row[self.id_col]
        # Avoid None in batches: if no patient column, use empty string
        if self.patient_col and (self.patient_col in self.df.columns):
            patient_id = row[self.patient_col]
        else:
            patient_id = ''

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': label,
            'image_id': image_id,
            'patient_id': patient_id
        }


