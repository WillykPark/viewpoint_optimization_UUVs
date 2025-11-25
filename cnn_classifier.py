import torch, torch.nn as nn
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import Compose, ToTensor, Normalize

# Temperature
class WithT(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.m(x) / torch.exp(self.logT)

# ---- 로더 + 예측기 ----
class CalibratedResNet:
    def __init__(self, num_classes: int, weight_path: str, device: torch.device):
        self.device = device
        base = resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = WithT(base).to(device)
        # calibrate_temperature.py에서 저장한 fls_resnet18_T.pth 로드
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()

        # ImageNet based Generalization Parameters
        self.tf = Compose([
            ToTensor(),  # (H,W,3)[0,1] -> FloatTensor[3,H,W]
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        self.softmax = nn.Softmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, img_np_float01: np.ndarray) -> np.ndarray:
        x = self.tf(img_np_float01).unsqueeze(0).to(self.device)  # [1,3,H,W]
        logits = self.model(x)                                    # [1,C] (T가 내장됨)
        probs = self.softmax(logits)                              # [1,C]
        return probs.squeeze(0).cpu().numpy()