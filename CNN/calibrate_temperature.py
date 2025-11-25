import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from dataset_fls import FLSList
import numpy as np
import json

ROOT = "/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped"
C = len(json.load(open(f"{ROOT}/splits/classes.json")))
BATCH = 64
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

val_ds = FLSList(f"{ROOT}/splits/val.txt")
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

base = resnet18(weights=None)
base.fc = nn.Linear(base.fc.in_features, C)
base.load_state_dict(torch.load("fls_resnet18_best.pth", map_location=device))
base.to(device).eval()

class WithT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.logT = nn.Parameter(torch.zeros(1) + np.log(1.0))
    def forward(self, x):
        return self.m(x) / torch.exp(self.logT)

mT = WithT(base).to(device)
opt = optim.LBFGS([mT.logT], lr=0.05, max_iter=50)
nll = nn.CrossEntropyLoss()

X, Y = [], []
with torch.no_grad():
    for x,y in val_dl:
        X.append(x.to(device)); Y.append(y.to(device))
X = torch.cat(X); Y = torch.cat(Y)

def closure():
    opt.zero_grad()
    loss = nll(mT(X), Y)
    loss.backward()
    return loss

opt.step(closure)
print("Optimized T:", float(torch.exp(mT.logT).detach().cpu()))
torch.save(mT.state_dict(), "fls_resnet18_T.pth")