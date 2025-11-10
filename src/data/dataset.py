import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class SegDataset(Dataset):
    def __init__(self, manifest_csv: str, indices=None, img_size: Tuple[int, int]=(256,256),
                 normalize: bool=True, num_classes: int=1, ignore_index: int=-1):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df.dropna(subset=["image_path"]).reset_index(drop=True)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.img_size = img_size
        self.normalize = normalize
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.has_mask = "mask_path" in self.df.columns and self.df["mask_path"].notna().all()

    def __len__(self):
        return len(self.df)

    def _open_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode not in ("RGB", "L", "I;16", "P"):
            img = img.convert("RGB")
        return img

    def _img_to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, self.img_size, InterpolationMode.BILINEAR)
        if img.mode == "L" or img.mode == "I;16":
            if img.mode == "I;16":
                arr = torch.from_numpy(np.array(img, dtype="float32"))
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                img = Image.fromarray((arr.numpy()*255).astype("uint8"), mode="L")
            x = TF.to_tensor(img)
            x = x.repeat(3, 1, 1)
        else:
            x = TF.to_tensor(img)
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            x = (x - mean) / std
        return x

    def _mask_to_tensor_binary(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        img = TF.resize(img, self.img_size, InterpolationMode.NEAREST)
        x = TF.to_tensor(img)
        x = (x > 0.5).float()
        return x  # 1xHxW

    def _mask_to_tensor_multiclass(self, img: Image.Image) -> torch.Tensor:
        if img.mode not in ("L", "P"):
            img = img.convert("L")
        img = TF.resize(img, self.img_size, InterpolationMode.NEAREST)
        t = torch.from_numpy(np.array(img, dtype="int64"))
        if self.num_classes > 1:
            mask = (t < 0) | (t >= self.num_classes)
            if self.ignore_index >= 0:
                t[mask] = self.ignore_index
            else:
                t[mask] = 0
        return t

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sid = row["id"]
        img = self._open_image(row["image_path"])
        x = self._img_to_tensor(img)
        if self.has_mask and isinstance(row.get("mask_path", None), str) and os.path.exists(row["mask_path"]):
            mimg = self._open_image(row["mask_path"])
            if self.num_classes <= 1:
                y = self._mask_to_tensor_binary(mimg)
            else:
                y = self._mask_to_tensor_multiclass(mimg)
        else:
            y = None
        return {"id": sid, "image": x, "mask": y}
