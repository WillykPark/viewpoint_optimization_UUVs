# envs/obs_provider.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

class ObsProvider(ABC):
    @abstractmethod
    def get_frame(self, view_idx: int, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Args:
            view_idx: viewpint index
            class_idx: Grount Truth or None in the real env.

        Returns:
            img: np.ndarray, shape (H, W, 3), dtype float32, value in [0,1]
        """
        ...

    def get_frame_with_angle(
        self,
        view_idx: int,
        class_idx: Optional[int] = None,
        ) -> Tuple[np.ndarray, Optional[float]]:
        """
        기본 구현: 각도 정보는 없음 → (img, None) 반환.
        Radon Provider는 이 메서드를 override 해서 (img, theta_deg)를 반환.
        """
        img = self.get_frame(view_idx, class_idx)
        return img, None