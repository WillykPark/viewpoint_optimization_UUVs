from __future__ import annotations
from typing import Optional, List, Tuple
from pathlib import Path
import random
import cv2
import numpy as np
import os, sys

# PPO 루트 경로 추가
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from Radon.radon import estimate_orientation, quantize_orientation
from obs_provider import ObsProvider

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetProviderRadon(ObsProvider):
    """
    - split.txt(train/val/test)을 받아서, 각 이미지에 대해
      Radon orientation을 계산하고, n_views 개의 각도 bin 으로 나눔.
    - self.items[class_idx][view_bin] = [(path, theta_deg), ...] 구조.
    - get_frame(view_idx, class_idx)가 들어오면
      해당 class의 view_bin = view_idx 에서 이미지를 뽑아줌
      (비어 있으면 근처 bin에서 fallback).
    """

    def __init__(self, root: str | Path, img_size: int = 224, n_views: int = 8):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.n_views = int(n_views)

        if not self.root.is_file():
            raise RuntimeError(f"DatasetProviderRadon expects split txt file, got: {self.root}")

        # 1) split 파일 읽기: "<path> <label>"
        raw_per_class: dict[int, List[Path]] = {}
        with open(self.root, "r") as f:
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
                    p = (self.root.parent / p).resolve()
                if not p.exists():
                    raise RuntimeError(f"Image not found from split: {p}")
                try:
                    lbl = int(lbl_str)
                except ValueError:
                    raise RuntimeError(f"Label must be int, got {lbl_str} in line: {line}")
                raw_per_class.setdefault(lbl, []).append(p)

        if not raw_per_class:
            raise RuntimeError(f"No entries in split: {self.root}")

        # 2) 각 이미지에 대해 orientation 추정 후, 각도 bin 으로 나누기
        #    items[class_idx][view_bin] = [(path, theta_deg), ...]
        max_lbl = max(raw_per_class.keys())
        self.items: List[List[List[Tuple[Path, float]]]] = []

        for cid in range(max_lbl + 1):
            paths = raw_per_class.get(cid, [])
            if not paths:
                raise RuntimeError(f"No images for class id {cid} in split {self.root}")

            bins: List[List[Tuple[Path, float]]] = [[] for _ in range(self.n_views)]

            for p in paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read image for radon: {p}")
                theta_est = estimate_orientation(img)          # 보통 degree 로 나온다고 가정
                theta_deg = float(theta_est)
                bin_id = quantize_orientation(theta_deg, n_views=self.n_views)
                if not (0 <= bin_id < self.n_views):
                    # 방어적 코드
                    bin_id = int(bin_id) % self.n_views
                bins[bin_id].append((p, theta_deg))

            # 혹시 특정 bin 이 완전히 비어 있으면, 나중에 fallback 로직이 처리
            self.items.append(bins)

    # ===== 전처리 공통 부분 =====
    @staticmethod
    def _log_comp(gray: np.ndarray) -> np.ndarray:
        gray = gray.astype(np.float32, copy=False)
        m = max(1.0, float(gray.max()))
        out = np.log1p(gray / m) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _pre(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._log_comp(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        return img

    # ===== 내부: 주어진 (class, view)에서 이미지 하나 고르는 함수 =====
    def _choose_from_bin(self, class_idx: int, view_idx: int) -> tuple[Path, float]:
        """
        class_idx, view_idx(0~n_views-1)에 해당하는 bin 에서 (path, theta_deg) 하나 선택.
        그 bin 이 비어 있으면 가까운 bin 쪽으로 fallback.
        """
        bins = self.items[class_idx]
        k = int(view_idx) % self.n_views

        # 1) 해당 bin에 이미지가 있으면 그 안에서 랜덤
        if bins[k]:
            return random.choice(bins[k])

        # 2) 없으면 양 옆 bin 쪽으로 점점 퍼지면서 탐색
        for offset in range(1, self.n_views):
            for sign in (-1, 1):
                b = (k + sign * offset) % self.n_views
                if bins[b]:
                    return random.choice(bins[b])

        # 여기까지 왔다는 건 해당 class 자체에 이미지가 아예 없다는 뜻
        raise RuntimeError(f"No images available for class {class_idx} in any bin")

    # ===== ObsProvider 인터페이스 구현 =====
    def get_frame(self, view_idx: int, class_idx: Optional[int] = None) -> np.ndarray:
        """
        PPO 환경에서 쓰는 함수.
        - view_idx : 에이전트가 선택한 viewpoint index (0~NV-1)
        - class_idx: true 클래스 id (env가 넘겨줌)
        """
        if class_idx is None:
            class_idx = 0

        p, theta_deg = self._choose_from_bin(class_idx, view_idx)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        return self._pre(img)

    def get_frame_with_angle(
        self,
        view_idx: int,
        class_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, float]:
        """
        (이미지, orientation_deg)를 함께 돌려주는 버전.
        - env.reset() 에서 theta_obj 를 잡거나
        - _obs_likelihood 안에서 orientation 정보가 필요할 때 사용.
        """
        if class_idx is None:
            class_idx = 0

        p, theta_deg = self._choose_from_bin(class_idx, view_idx)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img_pre = self._pre(img)
        return img_pre, theta_deg