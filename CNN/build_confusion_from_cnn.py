# CNN/build_confusion_from_cnn.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from dataset_fls import FLSList   # 이미 쓰고 있던 거
import json

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ROOT = "/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped"
with open(f"{ROOT}/splits/classes.json") as f:
    label_map = json.load(f)
C = len(label_map)   # = 17

val_txt = f"{ROOT}/splits/val.txt"
val_ds = FLSList(val_txt)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

# --- Temperature-calibrated ResNet18 (calibrate_temperature.py와 동일 구조) ---
base = resnet18(weights=None)
base.fc = nn.Linear(base.fc.in_features, C)
base.load_state_dict(torch.load("fls_resnet18_best.pth", map_location=device))

class WithT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.logT = nn.Parameter(torch.zeros(1))  # T는 state_dict 안에 이미 들어있다고 가정
    def forward(self, x):
        return self.m(x) / torch.exp(self.logT)

mT = WithT(base)
mT.load_state_dict(torch.load("fls_resnet18_T.pth", map_location=device))
mT.to(device).eval()

# --- confusion matrix count ---
M_count = np.zeros((C, C), dtype=np.int64)

with torch.no_grad():
    for x, y in val_dl:
        x = x.to(device)
        logits = mT(x)                 # [B, C]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        for t, p in zip(y_true, preds):
            M_count[t, p] += 1

# --- 행 기준 정규화해서 확률화 ---
M_hat = M_count.astype(np.float64)
row_sum = M_hat.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
M_hat /= row_sum

np.save("M_hat_from_cnn.npy", M_hat)
print("saved CNN-based confusion matrix to M_hat_from_cnn.npy")
print("shape:", M_hat.shape)