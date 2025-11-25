# eval_greedy_eig.py
import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuv_env_eig_cnn import AUVViewEnv
from cnn_classifier import CalibratedResNet
from Radon.dataset_provider_radon import DatasetProviderRadon

from eval_common import evaluate


def make_env_for_greedy():
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

    env = AUVViewEnv(
        C=17,
        tau=0.90,
        Tmax=12,
        mask_revisit=True,
        classifier=classifier,
        obs_provider=obs_provider,
        use_eig=True,   # EIG 계산 가능하게
    )
    return env


def greedy_eig_policy(env, obs):
    """
    현재 belief와 view에서,
    EIG - 이동비용이 최대가 되는 액션 선택.
    """
    # 1) 우선 stop 가능한지부터 체크 (confidence threshold)
    if env.belief is not None and env.belief.max() >= env.tau:
        return env.NV  # stop action

    mask = env._build_action_mask()
    best_a = None
    best_score = -1e9

    for a in range(env.NV):  # viewpoint만 순회
        if not mask[a]:
            continue

        # (1) 정보이득
        ig = env._compute_eig(env.belief, a)

        # (2) 이동 비용
        move_cost = env.lambda1 * env._move_dist(env.view, a) + env.lambda2

        # (3) 최종 score
        score = ig - move_cost

        if score > best_score:
            best_score = score
            best_a = a

    if best_a is None:
        # 예외 처리: 다 막혀있으면 그냥 stop
        return env.NV
    return best_a


if __name__ == "__main__":
    env = make_env_for_greedy()

    def select_action_greedy(env, obs):
        return greedy_eig_policy(env, obs)

    metrics = evaluate(env, select_action_greedy, n_episodes=100)
    print("[MODEL-BASED GREEDY EIG]")
    print(metrics)