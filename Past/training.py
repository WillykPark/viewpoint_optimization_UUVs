from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from uuv_env import AUVViewEnv

def mask_fn(env: AUVViewEnv):
    # sb3-contrib가 info가 아닌 env.method로 마스크를 읽을 수 있게 함수 제공
    return env._build_action_mask()

# 벡터라이즈 + 마스킹 래퍼
def make_env():
    env = AUVViewEnv(C=3, tau=0.90, Tmax=12, mask_revisit=True)
    env = ActionMasker(env, mask_fn)  # MaskablePPO 전용
    return env

vec_env = DummyVecEnv([make_env])

model = MaskablePPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    learning_rate=3e-4,
    clip_range=0.2,
    n_steps=2048,        # 데이터가 적으면 512~1024로 줄여도 됨
    batch_size=256,
    n_epochs=10,
    target_kl=0.01,
)

model.learn(total_timesteps=200_000)
model.save("auv_ppo_model.zip")
