import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from dataset_fls import FLSList
import json
import typing_extensions as te
if not hasattr(te, "Sentinel") and hasattr(te, "_Sentinel"):
    te.Sentinel = te._Sentinel
import os, csv
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = "/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped"
C = len(json.load(open(f"{ROOT}/splits/classes.json")))
BATCH = 64
EPOCHS = 30
LR = 3e-4

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)
HIST_PATH = os.path.join(SAVE_DIR, "train_history.csv")
ACC_PNG  = os.path.join(SAVE_DIR, "val_acc_curve.png")
NLL_PNG  = os.path.join(SAVE_DIR, "val_nll_curve.png")
BEST_PATH = os.path.join(SAVE_DIR, "fls_resnet18_best.pth")

print("cwd =", os.getcwd())
print("history will be saved to:", HIST_PATH)

train_ds = FLSList(f"{ROOT}/splits/train.txt", augment=True)
val_ds   = FLSList(f"{ROOT}/splits/val.txt")
test_ds  = FLSList(f"{ROOT}/splits/test.txt")

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, C)
model.to(device)

opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

def eval_dl(m, dl):
    m.eval(); correct=0; total=0; nll=0.0
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            logits = m(x)
            loss = crit(logits, y)
            nll += loss.item()*y.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total, nll/total

with open(HIST_PATH, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch","val_acc","val_nll"])

hist = {"epoch":[], "val_acc":[], "val_nll":[]}
best_val_nll = float("inf")

for ep in range(1, EPOCHS+1):
    model.train()
    for x,y in train_dl:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

    val_acc, val_nll = eval_dl(model, val_dl)
    print(f"[{ep}] val_acc={val_acc:.3f}  val_nll={val_nll:.3f}")

    hist["epoch"].append(ep)
    hist["val_acc"].append(float(val_acc))
    hist["val_nll"].append(float(val_nll))

    with open(HIST_PATH, "a", newline="") as f:
        w = csv.writer(f); w.writerow([ep, float(val_acc), float(val_nll)])

    if val_nll < best_val_nll:
        best_val_nll = val_nll
        torch.save(model.state_dict(), BEST_PATH)
        print("  -> saved:", BEST_PATH)

print("csv exists?", os.path.exists(HIST_PATH), "size:", os.path.getsize(HIST_PATH))

assert len(hist["epoch"]) > 0, "hist가 비어 있습니다. 학습 루프에서 append가 호출되었는지 확인하세요."

plt.figure()
plt.plot(hist["epoch"], hist["val_acc"], marker="o", label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(ACC_PNG, dpi=200)

plt.figure()
plt.plot(hist["epoch"], hist["val_nll"], marker="o", label="Val NLL")
plt.xlabel("Epoch"); plt.ylabel("NLL (Cross-Entropy)"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(NLL_PNG, dpi=200)

print(f"done. best: {BEST_PATH} | saved {HIST_PATH}, {ACC_PNG}, {NLL_PNG}")