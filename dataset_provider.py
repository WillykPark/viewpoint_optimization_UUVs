from __future__ import annotations
from typing import Optional, List
from pathlib import Path
import random
import cv2
import numpy as np

from obs_provider import ObsProvider

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetProvider(ObsProvider):
    def __init__(self, root: str | Path, img_size: int = 224):
        self.root = Path(root)
        self.img_size = int(img_size)

        if self.root.is_file():
            # Case B: split file mode
            split_path = self.root
            if split_path.suffix.lower() != ".txt":
                raise RuntimeError(f"Unsupported split file: {split_path}")

            # Read "<path> <label>" lines
            per_class: dict[int, list[Path]] = {}
            with open(split_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        p_str, lbl_str = line.rsplit(" ", 1)
                    except ValueError:
                        raise RuntimeError(f"Bad line in split file: {line}")
                    p = Path(p_str)
                    if not p.is_absolute():
                        # If path is relative, resolve against split file's parent
                        p = (split_path.parent / p).resolve()
                    if not p.exists():
                        raise RuntimeError(f"Image path not found from split: {p}")
                    try:
                        lbl = int(lbl_str)
                    except ValueError:
                        raise RuntimeError(f"Label must be int, got '{lbl_str}' in line: {line}")
                    per_class.setdefault(lbl, []).append(p)

            if not per_class:
                raise RuntimeError(f"No entries parsed from split: {split_path}")

            # Normalize into dense list-of-lists indexed by class id
            max_lbl = max(per_class.keys())
            self.files = []
            for cid in range(max_lbl + 1):
                lst = sorted(per_class.get(cid, []))
                if not lst:
                    raise RuntimeError(f"No images for class id {cid} in split {split_path}")
                self.files.append(lst)

            # Optional: synthetic class_dirs for compatibility
            self.class_dirs = [Path(f"class_{i}") for i in range(len(self.files))]

        else:
            # Case A: directory mode (original behavior)
            dir_root = self.root
            img_root = dir_root / "images"
            if not img_root.is_dir():
                if dir_root.is_dir():
                    img_root = dir_root
                else:
                    raise RuntimeError(f"Not found: {img_root}")

            self.class_dirs: List[Path] = sorted([d for d in img_root.iterdir() if d.is_dir()])
            if not self.class_dirs:
                raise RuntimeError(f"No class folders under: {img_root}")

            self.files: List[List[Path]] = []
            for d in self.class_dirs:
                lst = sorted([p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS])
                if not lst:
                    raise RuntimeError(f"No images found for class: {d.name}")
                self.files.append(lst)

    # ---------- Preprocessing ----------
    @staticmethod
    #log processing for sonar images
    def _log_comp(gray: np.ndarray) -> np.ndarray:
        gray = gray.astype(np.float32, copy=False)
        m = max(1.0, float(gray.max()))
        out = np.log1p(gray / m) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    
    #preprocessing function
    def _pre(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._log_comp(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        return img

    def get_frame(self, view_idx: int, class_idx: Optional[int] = None) -> np.ndarray:
        if class_idx is None:
            class_idx = 0  # safety fallback

        p = random.choice(self.files[class_idx])
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        return self._pre(img)