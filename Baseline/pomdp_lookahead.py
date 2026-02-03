# baselines_pomdp.py
import numpy as np

EPS = 1e-12

def entropy(p):
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

def normalize(p):
    s = float(np.sum(p))
    return p / (s + EPS)

def expected_stop_reward(belief, R_ok, R_err):
    """
    STOP했을 때 기대 보상:
      pred = argmax(b)
      P(correct)=max(b)
      E[R] = P(correct)*R_ok + (1-P(correct))*(-R_err)
    """
    p_correct = float(np.max(belief))
    return p_correct * float(R_ok) + (1.0 - p_correct) * (-float(R_err))

def observation_matrix_for_action(env, a):
    """
    env._compute_eig 내부에서 쓰는 것과 같은 방식으로
    view-angle에 따라 M을 섞어서 (C,C) observation model을 만든다.
    (없으면 그냥 base M 사용)
    """
    C = env.C
    M_base = env.M_hat if getattr(env, "M_hat", None) is not None else env.M  # (C,C)

    # angle-dependent degradation (env._compute_eig와 동일 컨셉)
    theta_view = env._angle(a)
    dtheta = env._angdist(theta_view, env.theta_obj)

    sigma = np.pi / 4.0
    weight = np.exp(- (dtheta ** 2) / (2.0 * sigma ** 2))

    M_uniform = np.ones_like(M_base) / C
    M_eff = weight * M_base + (1.0 - weight) * M_uniform
    return M_eff  # rows: true, cols: observed

def belief_update(b, M_eff, o):
    """
    b'(s) ∝ P(o | s, a) * b(s)
    여기서 M_eff[s, o] = P(o | s, a)
    """
    post = b * M_eff[:, o]
    return normalize(post)

def pomdp_lookahead_policy(env, depth=2, gamma=1.0, include_stop=True):
    """
    POMDP model-based planner baseline (belief tree search, exact over observations).
    - depth: lookahead depth (2 or 3 추천)
    - gamma: discount (보통 1.0)
    - include_stop: stop action도 후보로 포함할지
    """
    # 현재 상태 요약
    b0 = env.belief.copy()
    view0 = int(env.view)
    visited0 = env.visited.copy()
    t0 = int(env.t)

    # 가능한 action들
    def valid_actions(view, visited):
        acts = []
        for a in range(env.NV):
            if a == view:
                continue
            if env.mask_revisit and visited[a] == 1:
                continue
            acts.append(a)
        if include_stop:
            acts.append(env.stop_action)
        return acts

    # 재귀 value 계산
    def V(b, view, visited, t, d):
        # 강제 종료 시점이면 terminal reward
        if t >= env.Tmax:
            return expected_stop_reward(b, env.R_ok, env.R_err)

        # 깊이 0이면 “지금 stop했을 때 기대값”으로 종료(heuristic)
        if d == 0:
            return expected_stop_reward(b, env.R_ok, env.R_err)

        best = -1e18
        for a in valid_actions(view, visited):
            q = Q(b, view, visited, t, d, a)
            if q > best:
                best = q
        return best

    def Q(b, view, visited, t, d, a):
        # STOP
        if a == env.stop_action:
            return expected_stop_reward(b, env.R_ok, env.R_err)

        # MOVE cost
        move_dist = env._move_dist(view, a)
        step_cost = env.lambda1 * move_dist + env.lambda2
        r = -float(step_cost)

        # observation expectation
        M_eff = observation_matrix_for_action(env, a)  # (C,C)
        # P(o | a, b) = sum_s b(s) * P(o|s,a)
        Po = (b[:, None] * M_eff).sum(axis=0)  # (C,)

        # next "physical" state
        view2 = int(a)
        visited2 = visited.copy()
        visited2[view2] = 1.0
        t2 = t + 1

        # 기대 미래 가치
        exp_next = 0.0
        for o in range(env.C):
            p = float(Po[o])
            if p <= 0:
                continue
            b2 = belief_update(b, M_eff, o)
            exp_next += p * V(b2, view2, visited2, t2, d-1)

        return r + gamma * exp_next

    # 1-step에서 best action 선택
    best_a, best_q = None, -1e18
    for a in valid_actions(view0, visited0):
        q = Q(b0, view0, visited0, t0, depth, a)
        if q > best_q:
            best_q, best_a = q, a

    return int(best_a)