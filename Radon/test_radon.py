# test_radon_orientation.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from radon import estimate_orientation, quantize_orientation

# 🔹 이미지 하나 로드 (임의의 소나 이미지)
IMG_PATH = "/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped/glass-bottle/object-sideways-frame-089.png"

# 1️⃣ Load grayscale
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

# 2️⃣ Radon-based orientation estimation
theta = estimate_orientation(img)
view_idx = quantize_orientation(theta, n_views=8)

print(f"Estimated orientation: {theta:.2f}°   →  quantized view: {view_idx}")

# 3️⃣ (Optional) Radon visualization
from skimage.transform import radon
radon_theta = np.linspace(0., 180., 180, endpoint=False)
R = radon(img, theta=radon_theta, circle=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"Original (View ≈ {view_idx}, θ={theta:.1f}°)")
axes[1].imshow(R, cmap="gray", aspect="auto")
axes[1].set_title("Radon Transform (variance across θ)")
axes[1].set_xlabel("θ (deg)")
plt.tight_layout()
plt.show()