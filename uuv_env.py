import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPS = 1e-12

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + EPS)

class AUVViewEnv(gym.Env):
    """
    Single-target, 8 viewpoints, C-class recognition.

    Key modes:
      - Hybrid PPO: policy_input="belief"
      - Model-free PPO: policy_input="cnn"

    Termination modes:
      - stop_only=True  => episode ends ONLY when:
            (a) STOP action, or
            (b) Tmax reached (forced)
        (No tau-threshold auto-stop)
      - stop_only=False => original behavior: auto-stop by tau or Tmax.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 C=17,
                 radius=1.0,
                 tau=0.90,
                 Tmax=12,
                 lambda1=0.3,
                 lambda2=0.02,
                 R_ok=10.0,
                 R_err=12.0,
                 T_star=1.5,
                 mask_revisit=True,
                 seed=None,
                 classifier=None,
                 obs_provider=None,
                 M_hat=None,
                 use_eig=False,

                 # --- new/important switches ---
                 policy_input="belief",          # "belief" | "cnn"
                 track_belief_for_eval=True,     # if False: no belief update after observations
                 stop_only=False,                # if True: STOP + Tmax only (no tau auto-stop)
                 init_observe=True,              # reset()에서 초기 관측 1회 반영(선택)
                 eig_scale=2.0                   # EIG reward scaling
                 ):
        super().__init__()
        self.C = C
        self.NV = 8
        self.radius = radius
        self.tau = tau
        self.Tmax = Tmax
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.R_ok = R_ok
        self.R_err = R_err
        self.T_star = T_star
        self.mask_revisit = mask_revisit
        self.rng = np.random.default_rng(seed)

        self.classifier = classifier
        self.obs_provider = obs_provider
        self.use_eig = use_eig

        self.policy_input = policy_input
        self.track_belief_for_eval = track_belief_for_eval
        self.stop_only = stop_only
        self.init_observe = init_observe
        self.eig_scale = float(eig_scale)

        # STOP action index
        self.stop_action = self.NV

        # Confusion matrix fallback
        if M_hat is not None:
            self.M_hat = np.asarray(M_hat, dtype=np.float64)
            self.M = self.M_hat.copy()
        else:
            base = np.eye(self.C) * 0.85 + (np.ones((self.C, self.C)) - np.eye(self.C)) * (0.15 / (self.C - 1))
            self.M = base.astype(np.float64)
            self.M_hat = None

        # model-free state input용: 마지막 관측의 CNN 확률
        self.last_cnn_probs = np.ones(self.C, dtype=np.float32) / self.C

        # Observation:
        # [onehot(view) NV] + [mid C] + [visited NV] + [cos,sin 2]
        obs_dim = self.NV + self.C + self.NV + 2

        # cos/sin은 -1~1
        low = np.concatenate([
            np.zeros(self.NV, dtype=np.float32),
            np.zeros(self.C, dtype=np.float32),
            np.zeros(self.NV, dtype=np.float32),
            -np.ones(2, dtype=np.float32),
        ])
        high = np.concatenate([
            np.ones(self.NV, dtype=np.float32),
            np.ones(self.C, dtype=np.float32),
            np.ones(self.NV, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        ])

        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NV + 1)

        self._reset_episode_vars()

    # ---------- geometry helpers ----------
    def _angle(self, vid):
        return 2*np.pi * (vid % self.NV) / self.NV

    def _angdist(self, a, b):
        d = abs(a-b) % (2*np.pi)
        return min(d, 2*np.pi - d)

    def _angdiff_signed(self, a, b):
        d = (a - b) % (2*np.pi)
        if d > np.pi:
            d -= 2*np.pi
        return d

    def _rel_angle_feature(self):
        if not hasattr(self, "theta_obj"):
            return np.array([0.0, 0.0], dtype=np.float32)
        theta_view = self._angle(self.view)
        dtheta = self._angdiff_signed(theta_view, self.theta_obj)
        return np.array([np.cos(dtheta), np.sin(dtheta)], dtype=np.float32)

    def _move_dist(self, v_from, v_to):
        dtheta = self._angdist(self._angle(v_from), self._angle(v_to))
        return 2 * self.radius * np.sin(dtheta/2)

    # ---------- state/obs ----------
    def _state_vec(self):
        onehot = np.zeros(self.NV, dtype=np.float32)
        onehot[self.view] = 1.0

        ang_feat = self._rel_angle_feature()

        if self.policy_input == "belief":
            mid = self.belief.astype(np.float32)
        elif self.policy_input == "cnn":
            mid = self.last_cnn_probs.astype(np.float32)
        else:
            raise ValueError(f"Unknown policy_input: {self.policy_input}")

        return np.concatenate([onehot, mid, self.visited.astype(np.float32), ang_feat], axis=0)

    def _build_action_mask(self):
        mask = np.ones(self.NV + 1, dtype=bool)
        mask[self.view] = False
        if self.mask_revisit:
            mask[:self.NV] &= ~self.visited.astype(bool)
        mask[-1] = True
        return mask

    # ---------- observation model ----------
    def _obs_likelihood(self):
        # fallback
        if self.classifier is None or self.obs_provider is None:
            p = self.M[self.true_class].copy()
            logits = np.log(p + EPS)
            probs = softmax(logits / self.T_star).astype(np.float64)
            self.last_cnn_probs = probs.astype(np.float32)
            return probs

        # provider
        if hasattr(self.obs_provider, "get_frame_with_angle"):
            img, theta_deg = self.obs_provider.get_frame_with_angle(
                view_idx=self.view,
                class_idx=self.true_class,
            )
            self.theta_obj_deg = theta_deg
        else:
            img = self.obs_provider.get_frame(
                view_idx=self.view,
                class_idx=self.true_class,
            )
            self.theta_obj_deg = None

        probs = self.classifier.predict_proba(img)
        probs = np.clip(probs, EPS, 1.0)
        probs = probs / probs.sum()

        self.last_cnn_probs = probs.astype(np.float32)
        return probs.astype(np.float64)

    # ---------- EIG ----------
    def _compute_eig(self, belief, a):
        def entropy(p):
            p = np.clip(p, EPS, 1.0)
            p = p / np.sum(p)
            return -np.sum(p * np.log(p))

        M_base = self.M_hat if self.M_hat is not None else self.M

        theta_view = self._angle(a)
        dtheta = self._angdist(theta_view, self.theta_obj)

        sigma = np.pi / 4.0
        weight = np.exp(- (dtheta ** 2) / (2.0 * sigma ** 2))

        M_uniform = np.ones_like(M_base) / self.C
        M_eig = weight * M_base + (1.0 - weight) * M_uniform

        Po = (belief[None, :] * M_eig.T).sum(axis=1)

        E_H = 0.0
        for o in range(self.C):
            if Po[o] <= 0:
                continue
            post = M_eig[:, o] * belief
            post = post / (np.sum(post) + EPS)
            E_H += Po[o] * entropy(post)

        return float(entropy(belief) - E_H)

    # ---------- reset ----------
    def _reset_episode_vars(self):
        self.t = 0
        self.true_class = int(self.rng.integers(0, self.C))
        self.view = int(self.rng.integers(0, self.NV))

        self.visited = np.zeros(self.NV, dtype=np.float32)
        self.visited[self.view] = 1.0

        self.belief = np.ones(self.C, dtype=np.float64) / self.C
        self.last_cnn_probs = np.ones(self.C, dtype=np.float32) / self.C
        self.done = False

        # object orientation
        if self.obs_provider is not None and hasattr(self.obs_provider, "get_frame_with_angle"):
            _, theta_deg = self.obs_provider.get_frame_with_angle(
                view_idx=0,
                class_idx=self.true_class
            )
            self.theta_obj = np.deg2rad(theta_deg)
        else:
            self.theta_obj = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_episode_vars()

        # optional: take initial observation at starting view
        if self.init_observe:
            O0 = self._obs_likelihood()  # sets last_cnn_probs
            if self.track_belief_for_eval:
                self.belief = O0 * self.belief
                self.belief = self.belief / (np.sum(self.belief) + EPS)

        obs = self._state_vec()
        info = {"action_mask": self._build_action_mask()}
        return obs, info

    # ---------- step ----------
    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0.0
        terminated = False
        truncated = False

        # STOP ends the episode
        if action == self.stop_action:
            # final prediction: belief if available, else last cnn
            pred = int(np.argmax(self.belief)) if self.track_belief_for_eval else int(np.argmax(self.last_cnn_probs))
            reward += self.R_ok if pred == self.true_class else -self.R_err
            terminated = True

            obs = self._state_vec()
            info = {
                "action_mask": self._build_action_mask(),
                "pred": pred,
                "true": self.true_class,
                "move_cost": 0.0,
                "step_cost": 0.0,
                "info_gain": 0.0,
                "view": int(self.view),
            }
            return obs, float(reward), terminated, truncated, info

        # MOVE
        v_next = int(action)

        move_dist = self._move_dist(self.view, v_next)
        step_cost = (self.lambda1 * move_dist + self.lambda2)
        reward -= step_cost

        # optional EIG shaping
        info_gain = 0.0
        if self.use_eig and self.track_belief_for_eval:
            info_gain = self._compute_eig(self.belief, action)
            reward += self.eig_scale * info_gain

        # transition
        self.view = v_next
        self.visited[self.view] = 1.0

        # observe
        O = self._obs_likelihood()  # sets last_cnn_probs

        # belief update (only if enabled)
        if self.track_belief_for_eval:
            self.belief = O * self.belief
            self.belief = self.belief / (np.sum(self.belief) + EPS)

        # step count
        self.t += 1

        # --- Termination logic ---
        # (1번 방식) stop_only=True -> NO tau auto-stop
        if self.stop_only:
            if self.t >= self.Tmax:
                # forced end at Tmax
                pred = int(np.argmax(self.belief)) if self.track_belief_for_eval else int(np.argmax(self.last_cnn_probs))
                reward += self.R_ok if pred == self.true_class else -self.R_err
                terminated = True
        else:
            # original: tau or Tmax
            if self.track_belief_for_eval:
                stop_by_conf = (np.max(self.belief) >= self.tau)
                pred_for_stop = int(np.argmax(self.belief))
            else:
                stop_by_conf = (np.max(self.last_cnn_probs) >= self.tau)
                pred_for_stop = int(np.argmax(self.last_cnn_probs))

            if stop_by_conf or (self.t >= self.Tmax):
                reward += self.R_ok if pred_for_stop == self.true_class else -self.R_err
                terminated = True

        obs = self._state_vec()

        # report pred/true for evaluation
        pred_report = int(np.argmax(self.belief)) if self.track_belief_for_eval else int(np.argmax(self.last_cnn_probs))

        info = {
            "action_mask": self._build_action_mask(),
            "pred": pred_report,
            "true": self.true_class,
            "move_cost": float(move_dist),
            "step_cost": float(step_cost),
            "info_gain": float(info_gain),
            "view": int(self.view),
            "t": int(self.t),
        }
        return obs, float(reward), terminated, truncated, info