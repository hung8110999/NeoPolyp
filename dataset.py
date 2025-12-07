# dataset.py
import os
from typing import List, Union
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from sklearn.model_selection import KFold

# -------------------------
# PolypDataset - supports:
# 1) img_dir (str) + mask_dir (str) -> list files inside folder
# 2) img_paths (list) + mask_paths (list) -> use lists directly (K-Fold)
# -------------------------
class PolypDataset(Dataset):
    def __init__(
        self,
        img_source: Union[str, List[str]],
        mask_source: Union[str, List[str]],
        transform=None,
        img_size: tuple = None  # optional resize (H, W) applied in fallback path
    ):
        """
        img_source: either folder path (str) OR list of image file paths
        mask_source: either folder path (str) OR list of mask file paths
        transform: albumentations transform (expects keys image, mask) or None
        img_size: optional tuple (H, W) used only when transform is None (fallback resize)
        """
        # detect types
        self.transform = transform
        self.img_size = img_size

        # If folder path provided, list files
        if isinstance(img_source, str):
            if not os.path.isdir(img_source):
                raise ValueError(f"img_source is a string but not a directory: {img_source}")
            self.img_dir = img_source
            self.mask_dir = mask_source if isinstance(mask_source, str) else None
            self.img_paths = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)])
            # if mask_source is a directory, build mask paths by matching filenames
            if isinstance(mask_source, str):
                self.mask_paths = []
                for p in self.img_paths:
                    fname = os.path.basename(p)
                    cand = os.path.join(mask_source, fname)
                    if os.path.exists(cand):
                        self.mask_paths.append(cand)
                    else:
                        # try common extensions / basename matching
                        base, _ = os.path.splitext(fname)
                        found = False
                        for ext in ("png", "jpg", "jpeg"):
                            cand2 = os.path.join(mask_source, base + "." + ext)
                            if os.path.exists(cand2):
                                self.mask_paths.append(cand2)
                                found = True
                                break
                        if not found:
                            raise FileNotFoundError(f"Mask for {fname} not found in {mask_source}")
            else:
                raise ValueError("When img_source is a directory, mask_source must also be a directory path (str).")
        elif isinstance(img_source, list):
            # if list provided, require mask_source to be list of same length
            if not isinstance(mask_source, list):
                raise ValueError("When img_source is a list, mask_source must also be a list of same length.")
            if len(img_source) != len(mask_source):
                raise ValueError("img list and mask list must have same length.")
            self.img_paths = img_source
            self.mask_paths = mask_source
            self.img_dir = None
            self.mask_dir = None
        else:
            raise ValueError("img_source must be either a directory path (str) or a list of file paths.")

        if len(self.img_paths) == 0:
            raise ValueError("No images found/listed.")

    def __len__(self):
        return len(self.img_paths)

    def _read_image_mask(self, img_path, mask_path):
        # read image as RGB uint8
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # binarize mask (0 or 1)
        mask = (mask > 127).astype("uint8")
        return img, mask

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img, mask = self._read_image_mask(img_path, mask_path)

        if self.transform is not None:
            # albumentations style expected
            augmented = self.transform(image=img, mask=mask)
            img_t = augmented["image"]    # should be a tensor if ToTensorV2 used
            mask_t = augmented["mask"]
            # ensure mask tensor shape is (1,H,W)
            if isinstance(mask_t, np.ndarray):
                mask_t = torch.from_numpy(mask_t).unsqueeze(0).float()
            else:
                # if albumentations returned torch tensor, ensure dims
                if mask_t.ndim == 2:
                    mask_t = mask_t.unsqueeze(0).float()
            # if image returned as numpy, convert
            if isinstance(img_t, np.ndarray):
                img_t = torch.from_numpy(img_t).permute(2,0,1).float()
            return img_t.float(), mask_t.float()

        # fallback: use torchvision functional transforms
        # optionally resize
        if self.img_size is not None:
            H, W = self.img_size
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # to tensor and normalize
        img_t = TF.to_tensor(ImageFromArray(img=False, arr=img)) if False else torch.from_numpy(img.astype("float32") / 255.0).permute(2,0,1)
        # normalize using ImageNet stats
        img_t = TF.normalize(img_t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        mask_t = torch.from_numpy(mask.astype("float32")).unsqueeze(0)

        # random simple horizontal flip augmentation as example
        # (only when transform is not provided and you want augment; keep deterministic otherwise)
        # if self.transform is None and self.img_dir is not None and random.random() > 0.5:
        #     img_t = TF.hflip(img_t)
        #     mask_t = TF.hflip(mask_t)

        return img_t.float(), mask_t.float()

# -------------------------
# Utility: create k-fold splits from directories
# Returns list of tuples: (train_idx, val_idx, img_paths, mask_paths)
# -------------------------
def create_kfold_dataset(
    image_dir: str,
    mask_dir: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
):
    """
    Build k-fold splits from image & mask directories.
    Each returned fold tuple can be used as:
        train_dataset = PolypDataset([img_paths[i] for i in train_idx],
                                     [mask_paths[i] for i in train_idx],
                                     transform=...)
    """
    if not os.path.isdir(image_dir):
        raise ValueError(f"image_dir must be a directory path: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise ValueError(f"mask_dir must be a directory path: {mask_dir}")

    img_files = sorted(os.listdir(image_dir))
    img_paths = [os.path.join(image_dir, f) for f in img_files]

    # Build mask paths matched by filename (try same name, else try common extensions)
    mask_paths = []
    for p in img_paths:
        fname = os.path.basename(p)
        cand = os.path.join(mask_dir, fname)
        if os.path.exists(cand):
            mask_paths.append(cand)
        else:
            base, _ = os.path.splitext(fname)
            found = False
            for ext in ("png","jpg","jpeg"):
                cand2 = os.path.join(mask_dir, base + "." + ext)
                if os.path.exists(cand2):
                    mask_paths.append(cand2)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"No mask found for image {fname} in {mask_dir}")

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    idxs = list(range(len(img_paths)))
    for train_idx, val_idx in kf.split(idxs):
        folds.append((train_idx, val_idx, img_paths, mask_paths))
    return folds
