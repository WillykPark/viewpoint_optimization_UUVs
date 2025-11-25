import numpy as np
import torch
from sb3_contrib import MaskablePPO

from uuv_env_eig_cnn import AUVViewEnv
from cnn_classifier import CalibratedResNet
from Radon.dataset_provider_radon import DatasetProviderRadon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Evaluation using:", device)

model = MaskablePPO.load("auv_ppo_model_eig_cnn_radon.zip", device=device)

def make_eval_env():

    # 1) Radon 기반 이미지 공급기 (val 또는 test split 사용)
    obs_provider = DatasetProviderRadon(
        root="/blue/eel6825/yo.park/APRIL/PPO/dataset/"
             "Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped/splits/test.txt",
        img_size=224,
    )

    # 2) Temperature-calibrated CNN (17 클래스)
    classifier = CalibratedResNet(
        num_classes=17,
        weight_path="/blue/eel6825/yo.park/APRIL/PPO/CNN/fls_resnet18_T.pth",
        device=device,
    )

    # 3) Env 구성 (C=17, EIG 사용)
    env = AUVViewEnv(
        C=17,
        tau=0.90,
        Tmax=12,
        mask_revisit=True,
        classifier=classifier,
        obs_provider=obs_provider,
        use_eig=True,
    )

    return env


def evaluate(env, model, n_episodes=50):
    acc, steps, dist = [], [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_steps = 0
        ep_dist = 0.0
        while True:
            mask = env._build_action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)

            prev_view = env.view
            obs, r, done, trunc, info = env.step(action)

            if action != env.NV:  # moved
                ep_dist += env._move_dist(prev_view, env.view)

            ep_steps += 1

            if done or trunc:
                acc.append(1.0 if info["pred"] == info["true"] else 0.0)
                steps.append(ep_steps)
                dist.append(ep_dist)
                break

    return {
        "acc": float(np.mean(acc)),
        "steps": float(np.mean(steps)),
        "distance": float(np.mean(dist)),
    }


# 단일 env로 평가
eval_env = make_eval_env()
metrics = evaluate(eval_env, model, n_episodes=100)
print(metrics)