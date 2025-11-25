# eval_common.py
import numpy as np

def evaluate(env, select_action_fn, n_episodes=100):
    """
    env          : Gym-style env (AUVViewEnv or variant)
    select_action_fn(env, obs) -> action (int)
    n_episodes   : number of episodes for evaluation
    """
    acc_list = []
    steps_list = []
    dist_list = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_steps = 0
        ep_dist = 0.0

        while True:
            # 행동 선택
            action = select_action_fn(env, obs)

            # 이동 전 뷰 저장 (거리 계산용)
            prev_view = env.view

            obs, r, terminated, truncated, info = env.step(action)

            # stop(action == env.NV)가 아니면 거리 누적
            if action != env.NV:
                ep_dist += env._move_dist(prev_view, env.view)

            ep_steps += 1

            if terminated or truncated:
                # 정답 여부
                ep_acc = 1.0 if info["pred"] == info["true"] else 0.0
                acc_list.append(ep_acc)
                steps_list.append(ep_steps)
                dist_list.append(ep_dist)
                break

    metrics = {
        "acc": float(np.mean(acc_list)),
        "steps": float(np.mean(steps_list)),
        "distance": float(np.mean(dist_list)),
    }
    return metrics