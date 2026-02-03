# train_baselines.py
import os
import argparse
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from cnn_classifier import CalibratedResNet
from uuv_env import AUVViewEnv
from Radon.dataset_provider_radon import DatasetProviderRadon


def make_env(args, baseline: str, classifier, obs_provider):
    """
    baseline: model_free_ppo | hybrid_ppo_no_ig | hybrid_ppo_with_ig
    """
    common = dict(
        C=args.C,
        tau=args.tau,
        Tmax=args.Tmax,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        R_ok=args.R_ok,
        R_err=args.R_err,
        mask_revisit=bool(args.mask_revisit),
        init_observe=bool(args.init_observe),
        seed=args.seed,
        classifier=classifier,
        obs_provider=obs_provider,
        M_hat=None,
    )

    if baseline == "model_free_ppo":
        env = AUVViewEnv(
            **common,
            policy_input="cnn",
            track_belief_for_eval=False,
            stop_only=True,
            use_eig=False,
        )
        return env

    if baseline == "hybrid_ppo_no_ig":
        env = AUVViewEnv(
            **common,
            policy_input="belief",
            track_belief_for_eval=True,
            stop_only=bool(args.stop_only),   # 학습에서도 통일 원하면 True 권장
            use_eig=False,
        )
        return env

    if baseline == "hybrid_ppo_with_ig":
        env = AUVViewEnv(
            **common,
            policy_input="belief",
            track_belief_for_eval=True,
            stop_only=bool(args.stop_only),
            use_eig=True,
            eig_scale=args.eig_scale,
        )
        return env

    raise ValueError(f"Unknown baseline: {baseline}")


def main():
    p = argparse.ArgumentParser()

    # dataset / cnn / device
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--cnn_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")   # "cuda:0"

    # which baseline to train
    p.add_argument("--baseline", type=str, required=True,
                   choices=["model_free_ppo", "hybrid_ppo_no_ig", "hybrid_ppo_with_ig"])

    # training
    p.add_argument("--total_steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", type=str, default="tb_logs")
    p.add_argument("--save_dir", type=str, default="checkpoints")

    # env params
    p.add_argument("--C", type=int, default=17)
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--Tmax", type=int, default=12)
    p.add_argument("--lambda1", type=float, default=0.3)
    p.add_argument("--lambda2", type=float, default=0.02)
    p.add_argument("--R_ok", type=float, default=10.0)
    p.add_argument("--R_err", type=float, default=12.0)
    p.add_argument("--mask_revisit", type=int, default=1)
    p.add_argument("--init_observe", type=int, default=1)

    # IG shaping
    p.add_argument("--eig_scale", type=float, default=2.0)

    # termination control (학습 시에도 통일하려면 1 추천)
    p.add_argument("--stop_only", type=int, default=1,
                   help="1이면 STOP+Tmax로 통일(추천), 0이면 tau 자동종료 허용(hybrid에서만 의미)")

    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)

    # classifier/provider: 한 번만 생성해서 env에 주입
    classifier = CalibratedResNet(
        num_classes=args.C,
        weight_path=args.cnn_ckpt,
        device=device
    )
    obs_provider = DatasetProviderRadon(root=args.data_root, split="train")

    # env 만들기 (VecEnv로 감싸기)
    def _thunk():
        env = make_env(args, args.baseline, classifier, obs_provider)
        return Monitor(env)

    venv = DummyVecEnv([_thunk])
    venv = VecMonitor(venv)

    # PPO 세팅 (너 환경에 맞춰 적당히 무난한 값)
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    run_name = f"{args.baseline}_seed{args.seed}"
    model.learn(total_timesteps=args.total_steps, tb_log_name=run_name)

    out_path = os.path.join(args.save_dir, f"{run_name}.zip")
    model.save(out_path)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()