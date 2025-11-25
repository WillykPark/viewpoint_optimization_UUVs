from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from uuv_env_eig_cnn import AUVViewEnv
from cnn_classifier import CalibratedResNet
from Radon.dataset_provider_radon import DatasetProviderRadon
import torch, numpy as np


def mask_fn(env: AUVViewEnv):
    # sb3-contrib가 info가 아닌 env.method로 마스크를 읽을 수 있게 함수 제공
    return env._build_action_mask()

# CNN 기반 confusion matrix 로드
M_HAT_PATH = "/blue/eel6825/yo.park/APRIL/PPO/CNN/M_hat_from_cnn.npy"
M_hat = np.load(M_HAT_PATH)
print("Loaded M_hat:", M_hat.shape)

# 벡터라이즈 + 마스킹 래퍼
def make_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (1) Radon 기반 Provider
    obs_provider = DatasetProviderRadon(
        root="/blue/eel6825/yo.park/APRIL/PPO/dataset/"
             "Sonar Image/marine-debris-fls-datasets/md_fls_dataset/data/turntable-cropped/splits/val.txt",
        img_size=224
    )

    # (2) Temperature-Calibrated CNN Classifier
    classifier = CalibratedResNet(
        num_classes=17,
        weight_path="/blue/eel6825/yo.park/APRIL/PPO/CNN/fls_resnet18_T.pth",
        device=device
    )

    # (3) PPO 환경 생성
    env = AUVViewEnv(
        C=17,
        tau=0.90,
        Tmax=12,
        mask_revisit=True,
        classifier=classifier,
        obs_provider=obs_provider,
        use_eig=True,
        M_hat=M_hat,
    )

    env = ActionMasker(env, mask_fn)
    return env

vec_env = DummyVecEnv([make_env])

model = MaskablePPO(
    "MlpPolicy",
    vec_env,
    device="cuda",
    verbose=1,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    learning_rate=3e-4,
    clip_range=0.2,
    n_steps=2048,    
    batch_size=256,
    n_epochs=10,
    target_kl=0.01,
    tensorboard_log="./tb_logs"
)

model.learn(total_timesteps=400_000, tb_log_name="eig_cnn_radon")
model.save("auv_ppo_model_eig_cnn_radon.zip")
