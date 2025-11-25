import torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from dataset_fls import FLSList
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import json

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def ece_score(y_true, probs, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    preds = probs.argmax(1)
    confs = probs.max(1)
    accs = (preds==y_true).astype(np.float32)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (confs>=bins[i]) & (confs<bins[i+1])
        else:
            m = (confs>=bins[i]) & (confs<=bins[i+1])
        if m.sum()==0: continue
        ece += (m.mean()) * abs(accs[m].mean() - confs[m].mean())
    return float(ece)

ROOT = "/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped"
C = len(json.load(open(f"{ROOT}/splits/classes.json")))
BATCH=64
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_ds = FLSList(f"{ROOT}/splits/test.txt")
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

class WithT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.m(x) / torch.exp(self.logT)

@torch.no_grad()
def eval_model(model, dl, name="model"):
    logits_all, y_all = [], []
    model.eval()
    for x,y in dl:
        x = x.to(device)
        logits = model(x)
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all)
    probs = softmax_np(logits)
    acc = accuracy_score(y_true, probs.argmax(1))
    nll = log_loss(y_true, probs, labels=list(range(C)))

    y_onehot = np.eye(C, dtype=np.float32)[y_true]   # [N, C]
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
    
    ece = ece_score(y_true, probs)
    print(f"[{name}]  Acc={acc:.3f}  NLL={nll:.3f}  Brier={brier:.3f}  ECE={ece:.3f}")
    return {"acc":acc,"nll":nll,"brier":brier,"ece":ece}

#before calibration
base_before = resnet18(weights=None)
base_before.fc = nn.Linear(base_before.fc.in_features, C)
base_before.load_state_dict(torch.load("fls_resnet18_best.pth", map_location=device))
base_before.to(device)

#after calibration
base_after = resnet18(weights=None)
base_after.fc = nn.Linear(base_after.fc.in_features, C)
mT = WithT(base_after)
mT.load_state_dict(torch.load("fls_resnet18_T.pth", map_location=device))
mT.to(device)

print("Models loaded.")

res_before = eval_model(base_before, test_dl, name="Before Calibration")
res_after  = eval_model(mT, test_dl, name="After  Calibration")

print("\n Summary Comparison")
for k in ["acc","nll","brier","ece"]:
    diff = res_before[k] - res_after[k]
    sign = "↓" if diff > 0 else "↑"
    print(f"{k.upper():>5}: {res_before[k]:.4f} → {res_after[k]:.4f}   ({sign} {abs(diff):.4f})")