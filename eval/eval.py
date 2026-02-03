import os
import csv
import json
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import torch

from cnn_classifier import CalibratedResNet
from uuv_env import AUVViewEnv
from Radon.dataset_provider_radon import DatasetProviderRadon
from Baseline.pomdp_lookahead import pomdp_lookahead_policy

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


# -----------------------
# Metrics
# -----------------------
@dataclass
class EpisodeResult:
    acc: float
    steps: int
    motion_cost: float
    step_cost: float
    forced_end: int
    entropy_reduction: Optional[float] = None


def stderr(arr: np.ndarray) -> float:
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def summarize_results(results: List[EpisodeResult]) -> Dict[str, Any]:
    acc = np.array([r.acc for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=float)
    motion = np.array([r.motion_cost for r in results], dtype=float)
    step_cost = np.array([r.step_cost for r in results], dtype=float)
    forced = np.array([r.forced_end for r in results], dtype=float)

    summary = {
        "acc_mean": float(acc.mean()), "acc_se": stderr(acc),
        "steps_mean": float(steps.mean()), "steps_se": stderr(steps),
        "motion_cost_mean": float(motion.mean()), "motion_cost_se": stderr(motion),
        "step_cost_mean": float(step_cost.mean()), "step_cost_se": stderr(step_cost),
        "forced_end_rate": float(forced.mean()),
        "n_episodes": len(results),
    }

    ent = [r.entropy_reduction for r in results if r.entropy_reduction is not None]
    if len(ent) > 0:
        ent = np.array(ent, dtype=float)
        summary["entropy_reduction_mean"] = float(ent.mean())
        summary["entropy_reduction_se"] = stderr(ent)
    else:
        summary["entropy_reduction_mean"] = None
        summary["entropy_reduction_se"] = None

    return summary


# -----------------------
# Policies
# -----------------------
def greedy_eig_policy(env: AUVViewEnv) -> int:
    """
    Greedy EIG planner baseline:
      a* = argmax_a [EIG(b,a) - lambda1 * move_dist - lambda2]
    """
    candidates = []
    for a in range(env.NV):
        if a == env.view:
            continue
        if env.mask_revisit and env.visited[a] == 1:
            continue
        candidates.append(a)

    if len(candidates) == 0:
        return env.stop_action

    best_a, best_score = None, -1e18
    for a in candidates:
        ig = env._compute_eig(env.belief, a)
        move_dist = env._move_dist(env.view, a)
        step_cost = env.lambda1 * move_dist + env.lambda2
        score = ig - step_cost
        if score > best_score:
            best_score, best_a = score, a

    return int(best_a)


def random_policy(env: AUVViewEnv) -> int:
    mask = env._build_action_mask()
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


def ppo_policy_factory(model) -> Callable[[np.ndarray, AUVViewEnv], int]:
    def _pi(obs: np.ndarray, env: AUVViewEnv) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return _pi


# -----------------------
# Rollout
# -----------------------
def rollout_episode(
    env: AUVViewEnv,
    policy: Callable[[np.ndarray, AUVViewEnv], int],
    max_steps_guard: int = 10_000,
    compute_entropy_reduction: bool = False,
) -> EpisodeResult:

    obs, info = env.reset()

    total_motion = 0.0
    total_step_cost = 0.0
    forced_end = 0

    # entropy metric: belief 쓸 때만 의미
    H0 = None
    if compute_entropy_reduction:
        b0 = env.belief.copy()
        H0 = float(-(np.clip(b0, 1e-12, 1.0) * np.log(np.clip(b0, 1e-12, 1.0))).sum())

    steps = 0
    done = False
    last_info = {}

    while not done and steps < max_steps_guard:
        a = policy(obs, env)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        last_info = info

        total_motion += float(info.get("move_cost", 0.0))
        total_step_cost += float(info.get("step_cost", 0.0))
        steps += 1

        # Tmax forced end 여부(환경 info에 t 넣어두면 정확)
        if done and info.get("t", None) is not None:
            if int(info["t"]) >= int(env.Tmax) and a != env.stop_action:
                forced_end = 1

    pred = int(last_info.get("pred", np.argmax(env.belief)))
    true = int(last_info.get("true", env.true_class))
    acc = 1.0 if pred == true else 0.0

    ent_red = None
    if compute_entropy_reduction and H0 is not None:
        bT = env.belief.copy()
        HT = float(-(np.clip(bT, 1e-12, 1.0) * np.log(np.clip(bT, 1e-12, 1.0))).sum())
        ent_red = float(H0 - HT)

    return EpisodeResult(
        acc=acc,
        steps=int(steps),
        motion_cost=float(total_motion),
        step_cost=float(total_step_cost),
        forced_end=int(forced_end),
        entropy_reduction=ent_red,
    )


def save_episode_csv(path: str, results: List[EpisodeResult]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acc", "steps", "motion_cost", "step_cost", "forced_end", "entropy_reduction"])
        for r in results:
            w.writerow([
                r.acc, r.steps, r.motion_cost, r.step_cost, r.forced_end,
                "" if r.entropy_reduction is None else r.entropy_reduction
            ])


# -----------------------
# Env factory
# -----------------------
def make_env_for_method(args, method: str, classifier, obs_provider) -> AUVViewEnv:
    """
    classifier / obs_provider는 main에서 한 번만 생성해서 넘겨준다 (재사용)
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
        M_hat=None,   # ✅ 지금은 None (fallback으로 generic confusion 사용)
    )

    if method == "model_free_ppo":
        return AUVViewEnv(
            **common,
            policy_input="cnn",
            track_belief_for_eval=False,
            stop_only=True,
            use_eig=False,
        )

    if method == "hybrid_ppo_no_ig":
        return AUVViewEnv(
            **common,
            policy_input="belief",
            track_belief_for_eval=True,
            stop_only=bool(args.hybrid_stop_only),
            use_eig=False,
        )

    if method == "hybrid_ppo_with_ig":
        return AUVViewEnv(
            **common,
            policy_input="belief",
            track_belief_for_eval=True,
            stop_only=bool(args.hybrid_stop_only),
            use_eig=True,
            eig_scale=args.eig_scale,
        )

    if method == "greedy_eig":
        return AUVViewEnv(
            **common,
            policy_input="belief",
            track_belief_for_eval=True,
            stop_only=bool(args.hybrid_stop_only),
            use_eig=False,   # planner는 decision에서 EIG 쓰니까 shaping reward는 꺼도 됨
        )

    if method == "random":
        return AUVViewEnv(
            **common,
            policy_input="cnn",
            track_belief_for_eval=False,
            stop_only=True,
            use_eig=False,
        )
    
    if method == "pomdp_lookahead":
        return AUVViewEnv(
            **common,
            policy_input="belief",
            track_belief_for_eval=True,
            stop_only=bool(args.hybrid_stop_only),
            use_eig=False,
        )

    raise ValueError(f"Unknown method: {method}")


# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="runs_eval")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)

    # ✅ 추가: 데이터/모델/디바이스
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--cnn_ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")  # "cuda:0" 가능

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
    p.add_argument("--eig_scale", type=float, default=2.0)

    # termination fairness
    p.add_argument("--hybrid_stop_only", type=int, default=0,
                   help="1이면 hybrid도 STOP+Tmax로 통일, 0이면 hybrid는 tau 자동종료 허용")

    # ppo model paths
    p.add_argument("--model_free_ckpt", type=str, default="")
    p.add_argument("--hybrid_no_ig_ckpt", type=str, default="")
    p.add_argument("--hybrid_with_ig_ckpt", type=str, default="")

    # methods
    p.add_argument("--methods", nargs="+",
                   default=["greedy_eig", "model_free_ppo", "hybrid_ppo_no_ig", "hybrid_ppo_with_ig"])

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ✅ device는 main()에서 args 기반으로 결정
    device = torch.device(args.device)

    # ✅ classifier / provider는 여기서 한 번만 생성 (중요)
    classifier = CalibratedResNet(
        num_classes=args.C,
        weight_path=args.cnn_ckpt,
        device=device
    )

    # DatasetProviderRadon 생성자 인자가 root/split인지 확인 필요
    obs_provider = DatasetProviderRadon(
        root=args.data_root,
        split="test"
    )

    summaries = []

    for method in args.methods:
        env = make_env_for_method(args, method, classifier, obs_provider)

        # policy 선택
        if method == "greedy_eig":
            policy = lambda obs, e: greedy_eig_policy(e)

        elif method == "pomdp_lookahead":
            policy = lambda obs, e: pomdp_lookahead_policy(e, depth=2, gamma=1.0, include_stop=True)

        elif method == "random":
            policy = lambda obs, e: random_policy(e)

        elif method in ["model_free_ppo", "hybrid_ppo_no_ig", "hybrid_ppo_with_ig"]:
            if PPO is None:
                raise RuntimeError("stable-baselines3 is not installed but PPO method requested.")

            ckpt = {
                "model_free_ppo": args.model_free_ckpt,
                "hybrid_ppo_no_ig": args.hybrid_no_ig_ckpt,
                "hybrid_ppo_with_ig": args.hybrid_with_ig_ckpt,
            }[method]

            if ckpt.strip() == "":
                raise ValueError(f"Checkpoint path missing for {method}.")

            model = PPO.load(ckpt)
            policy = ppo_policy_factory(model)

        else:
            raise ValueError(method)

        # entropy metric: model-free는 N/A
        compute_entropy = (method != "model_free_ppo") and getattr(env, "track_belief_for_eval", False)

        results = []
        for ep in range(args.episodes):
            res = rollout_episode(env, policy, compute_entropy_reduction=compute_entropy)
            results.append(res)

        # 저장
        out_csv = os.path.join(args.out_dir, f"results_{method}.csv")
        save_episode_csv(out_csv, results)

        # summary
        summ = summarize_results(results)
        summ["method"] = method
        summ["results_csv"] = out_csv
        summaries.append(summ)

        print(f"\n[{method}]")
        for k, v in summ.items():
            if k in ["method", "results_csv"]:
                continue
            print(f"  {k}: {v}")

    # summary 저장
    sum_csv = os.path.join(args.out_dir, "summary.csv")
    with open(sum_csv, "w", newline="") as f:
        keys = [
            "method",
            "acc_mean", "acc_se",
            "steps_mean", "steps_se",
            "motion_cost_mean", "motion_cost_se",
            "step_cost_mean", "step_cost_se",
            "forced_end_rate",
            "entropy_reduction_mean", "entropy_reduction_se",
            "n_episodes",
            "results_csv",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, None) for k in keys})

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved summary: {sum_csv}")


if __name__ == "__main__":
    main()