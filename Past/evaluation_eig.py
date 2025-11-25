import numpy as np
from sb3_contrib import MaskablePPO
from uuv_env_eig import AUVViewEnv

model = MaskablePPO.load("auv_ppo_model.zip")

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
        "acc": np.mean(acc),
        "steps": np.mean(steps),
        "distance": np.mean(dist)
    }

# 단일 env로 평가
eval_env = AUVViewEnv(C=3, tau=0.90, Tmax=12, mask_revisit=True)
metrics = evaluate(eval_env, model, n_episodes=100)
print(metrics)