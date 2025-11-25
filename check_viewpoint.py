from sb3_contrib import MaskablePPO
from uuv_env_eig_cnn import AUVViewEnv
from Radon.dataset_provider_radon import DatasetProviderRadon
from cnn_classifier import CalibratedResNet
import numpy as np, torch

# env 만들 때 training 때랑 똑같이 설정
def make_env():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    obs_provider = DatasetProviderRadon(
        root="/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped/splits/val.txt",
        img_size=224
    )
    classifier = CalibratedResNet(
        num_classes=17,
        weight_path="/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/CNN/fls_resnet18_T.pth",
        device=device
    )
    return AUVViewEnv(
        C=17,
        tau=0.90,
        Tmax=12,
        mask_revisit=True,
        classifier=classifier,
        obs_provider=obs_provider,
        use_eig=True
    )

env = make_env()
model = MaskablePPO.load("auv_ppo_model_eig_cnn_radon.zip")

visit_counts = np.zeros(env.NV, dtype=int)
final_view_counts = np.zeros(env.NV, dtype=int)

N = 200  # 에피소드 수
for _ in range(N):
    obs, info = env.reset()
    visited = set([env.view])

    while True:
        mask = env._build_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        prev_view = env.view
        obs, r, done, trunc, info = env.step(action)

        if action != env.NV:
            visited.add(env.view)

        if done or trunc:
            break

    for v in visited:
        visit_counts[v] += 1
    final_view_counts[prev_view] += 1

print("visit_counts:", visit_counts)
print("final_view_counts:", final_view_counts)