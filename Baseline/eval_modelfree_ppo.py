# eval_ppofree.py (env가 준비됐다는 가정 하에 예시)
import torch
from sb3_contrib import MaskablePPO

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuv_env_model_free import AUVViewEnvModelFree
from cnn_classifier import CalibratedResNet
from Radon.dataset_provider_radon import DatasetProviderRadon

from eval_common import evaluate


def make_free_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_provider = DatasetProviderRadon(
        root="/blue/eel6825/yo.park/APRIL/PPO/dataset/"
             "Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/"
             "turntable-cropped/splits/test.txt",
        img_size=224,
    )

    classifier = CalibratedResNet(
        num_classes=17,
        weight_path="/blue/eel6825/yo.park/APRIL/PPO/CNN/fls_resnet18_T.pth",
        device=device,
    )

    env = AUVViewEnvModelFree(
        C=17,
        Tmax=12,
        mask_revisit=True,
        classifier=classifier,
        obs_provider=obs_provider,
    )
    return env


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskablePPO.load("auv_ppo_model_free.zip", device=device)

    env = make_free_env()

    def select_action_free(env, obs):
        mask = env._build_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    metrics = evaluate(env, select_action_free, n_episodes=100)
    print("[PURE MODEL-FREE PPO]")
    print(metrics)