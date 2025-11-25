import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.models import resnet18
from dataset_fls import FLSList
import json
import os
import torch, torch.nn as nn

class WithT(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.m(x) / torch.exp(self.logT)

# 0. 경로 설정
ROOT = "/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped"
SPLIT = Path(ROOT) / "splits" / "test.txt"
CLASSES = json.load(open(Path(ROOT)/"splits"/"classes.json"))
id2name = {v:k for k,v in CLASSES.items()}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. 모델 로드 (calibration 된 거)
base = resnet18(weights=None)
base.fc = torch.nn.Linear(base.fc.in_features, len(CLASSES))
model = WithT(base)
model.load_state_dict(torch.load("fls_resnet18_T.pth", map_location=device))
model.to(device).eval()

# 2. test 이미지 리스트 읽기
items = []
with open(SPLIT, "r") as f:
    for line in f:
        line = line.strip()
        if not line: 
            continue
        p_str, lbl_str = line.rsplit(" ", 1)
        items.append((p_str, int(lbl_str)))

os.makedirs("viz_out", exist_ok=True)

def preprocess(img, size=224):
    # dataset_fls랑 최대한 맞춰주자
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # log comp
    img_f = img.astype(np.float32)
    m = max(1.0, float(img_f.max()))
    img_f = np.log1p(img_f / m) * 255.0
    img_f = np.clip(img_f, 0, 255).astype(np.uint8)

    img_f = cv2.resize(img_f, (size, size), interpolation=cv2.INTER_AREA)
    img_f = np.stack([img_f, img_f, img_f], axis=-1).astype(np.float32) / 255.0

    # To tensor + ImageNet norm
    x = torch.from_numpy(img_f).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    x = (x - mean) / std
    return x

# 3. 몇 장만 시각화
for i, (img_path, gt) in enumerate(items[:20]):  # 처음 20장만
    raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    x = preprocess(raw).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_id = int(probs.argmax())
    pred_name = id2name[pred_id]
    conf = float(probs[pred_id])

    # 4. 원래 이미지에 박스/텍스트 그리기
    show = raw
    if show.ndim == 2:
        show = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)

    H, W = show.shape[:2]
    pad_top = 40
    pad_side = 80   # ← 양옆으로 여유 줌

    canvas = np.zeros((H + pad_top, W + pad_side, 3), dtype=np.uint8)
    # 가운데 배치
    canvas[pad_top:pad_top+H, pad_side//2:pad_side//2+W] = show

    text = f"{pred_name}"
    cv2.putText(
        canvas,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


    out_path = f"viz_out/pred_{i:03d}.png"
    cv2.imwrite(out_path, canvas)

print("saved to viz_out/")