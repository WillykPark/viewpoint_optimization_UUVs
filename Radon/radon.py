# radon_utils.py
import numpy as np
from skimage.transform import radon

def estimate_orientation(img_gray: np.ndarray, theta_res: int = 180) -> float:
    """
    단일 채널(grayscale) 이미지에서 dominant orientation(0~180deg) 추정.
    sonar 이미지는 구조가 강해서 이걸 그냥 viewpoint proxy로 쓴다.
    """
    # img_gray : (H,W), float 또는 uint8
    if img_gray.ndim != 2:
        raise ValueError("estimate_orientation expects grayscale image")

    # Radon transform
    theta = np.linspace(0., 180., theta_res, endpoint=False)
    R = radon(img_gray, theta=theta, circle=False)   # shape: (len_s, theta_res)

    # 각 θ별 에너지를 보고 가장 강한 방향 선택
    energy = R.var(axis=0)    # (theta_res,)
    k = int(np.argmax(energy))
    dom_theta = theta[k]      # 0~180 사이
    return float(dom_theta)


def quantize_orientation(theta_deg: float, n_views: int = 8) -> int:
    """
    0~180도 → n_views (보통 8) 개로 양자화.
    0, 180이 사실상 같은 방향이라 180을 0으로 말아준다.
    """
    theta_wrapped = theta_deg % 180.0
    bin_size = 180.0 / n_views
    vid = int(theta_wrapped // bin_size)
    if vid >= n_views:
        vid = n_views - 1
    return vid