# viz_trajectory.py
import numpy as np
import matplotlib.pyplot as plt
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from uuv_env_eig_cnn import AUVViewEnv
from cnn_classifier import CalibratedResNet
from Radon.dataset_provider_radon import DatasetProviderRadon


def mask_fn(env: AUVViewEnv):
    return env._build_action_mask()


def make_env_for_viz():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # (1) Radon 기반 provider
    obs_provider = DatasetProviderRadon(
        root="/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/"
             "APRI Lab/PPO/dataset/Sonar Image/marine-debris-fls-datasets/"
             "md_fls_dataset/data/turntable-cropped/splits/val.txt",
        img_size=224,
    )

    # (2) CNN + Temperature
    classifier = CalibratedResNet(
        num_classes=17,                 # 지금 학습한 클래스 개수
        weight_path="/Users/park-yong-kyoon/Documents/07.UF MS ECE/01.Study/Research/APRI Lab/PPO/CNN/fls_resnet18_T.pth",
        device=device,
    )

    # (3) Env 생성
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


def run_episode(env, model):
    """한 에피소드 돌리고 방문한 viewpoint 기록"""
    obs, info = env.reset()
    traj_views = [env.view]
    done = False
    trunc = False

    while not (done or trunc):
        mask = env._build_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        if action != env.NV:   # stop이 아니면 위치 갱신
            traj_views.append(env.view)

    return traj_views, info


def view_to_xy(env, vid):
    theta = env._angle(vid)
    r = env.radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def plot_trajectory(env, traj_views, info, save_path="traj_example.png"):
    xs, ys = [], []
    for v in range(env.NV):
        x, y = view_to_xy(env, v)
        xs.append(x); ys.append(y)

    path_x = [view_to_xy(env, v)[0] for v in traj_views]
    path_y = [view_to_xy(env, v)[1] for v in traj_views]

    plt.figure(figsize=(5, 5))

    plt.scatter(0, 0, c="red", s=80,
                label=f"Target (true={info['true']})")

    plt.scatter(xs, ys, c="gray", s=40, label="Viewpoints")
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x, y, str(i), fontsize=8, ha="center", va="center")

    plt.plot(path_x, path_y, "-o", c="blue", label="Agent path")
    plt.scatter(path_x[0], path_y[0], c="green", s=70, label="Start")
    plt.scatter(path_x[-1], path_y[-1], c="purple", s=70, label="Last")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Trajectory | pred={info['pred']} / true={info['true']}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("saved:", save_path)


if __name__ == "__main__":
    env = make_env_for_viz()
    model = MaskablePPO.load("auv_ppo_model_eig_cnn_radon.zip")

    traj_views, info = run_episode(env, model)
    print("visited views:", traj_views)
    print("pred:", info["pred"], " true:", info["true"])

    plot_trajectory(env, traj_views, info, save_path="traj_example.png")