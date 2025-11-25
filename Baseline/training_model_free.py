import torch
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuv_env_model_free import AUVViewEnvModelFree
from cnn_classifier import CalibratedResNet
from Radon.dataset_provider_radon import DatasetProviderRadon


# -----------------------------
# Action masking wrapper 함수
# -----------------------------
def mask_fn(env):
    return env._build_action_mask()


# -----------------------------
# Env 생성 함수
# -----------------------------
def make_free_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_provider = DatasetProviderRadon(
        root="/blue/eel6825/yo.park/APRIL/PPO/dataset/"
             "Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/"
             "turntable-cropped/splits/train.txt",
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

    # 🔴 반드시 마스킹 적용해야 MaskablePPO가 동작함
    env = ActionMasker(env, mask_fn)
    return env


# -----------------------------
# Main: PPO Training
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 벡터라이즈 + Monitor
    vec_env = DummyVecEnv([make_free_env])
    vec_env = VecMonitor(vec_env)

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        gamma=0.99,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        gae_lambda=0.95,
        n_epochs=10,
        device=device,
        tensorboard_log="./tb_logs_free/",
    )

    model.learn(total_timesteps=400_000, tb_log_name="free")
    model.save("auv_ppo_model_free.zip")