import cv2, numpy as np, torch
from torch.utils.data import Dataset
from albumentations import (Compose, GaussNoise, MotionBlur, RandomBrightnessContrast,
                            HorizontalFlip, ShiftScaleRotate, Normalize)
from albumentations.pytorch import ToTensorV2

class FLSList(Dataset):
    def __init__(self, list_file, img_size=224, augment=False):
        self.items = []
        with open(list_file) as f:
            for line in f:
                p, y = line.strip().rsplit(" ", 1)
                self.items.append((p, int(y)))
        self.img_size = img_size
        self.augment = augment
        self.tf = self._build_tf()

    def _log_comp(self, img):
        img = img.astype(np.float32) + 1e-6
        img = np.log1p(img)
        m = img.max() + 1e-6
        img = img / m
        return (img * 255.0).astype(np.uint8)

    def _pre(self, img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._log_comp(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = np.stack([img, img, img], axis=-1)  # 1ch -> 3ch
        return img

    def _build_tf(self):
        aug = []
        if self.augment:
            aug += [
                GaussNoise(var_limit=(5., 20.), p=0.4),
                MotionBlur(blur_limit=3, p=0.3),
                RandomBrightnessContrast(p=0.3),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.4),
            ]
        return Compose(aug + [Normalize(), ToTensorV2()])

    def __getitem__(self, i):
        path, y = self.items[i]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = self._pre(img)
        img = self.tf(image=img)["image"]  # Tensor [3,H,W]
        return img, y

    def __len__(self):
        return len(self.items)