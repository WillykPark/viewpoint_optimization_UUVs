import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPS = 1e-12

class AUVViewEnvModelFree(gym.Env):
    """Model-Free PPO Environment (no belief, no EIG, no POMDP structure)."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        C=17,
        radius=1.0,
        Tmax=12,
        lambda1=0.3,
        lambda2=0.02,
        R_ok=10.0,
        R_err=12.0,
        mask_revisit=True,
        seed=None,
        classifier=None,
        obs_provider=None,
    ):
        super().__init__()
        self.C = C
        self.NV = 8
        self.radius = radius
        self.Tmax = Tmax
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.R_ok = R_ok
        self.R_err = R_err
        self.mask_revisit = mask_revisit
        self.classifier = classifier
        self.obs_provider = obs_provider
        self.rng = np.random.default_rng(seed)

        # state: one-hot(view) + cnn_probs + visited_mask
        obs_dim = self.NV + self.C + self.NV
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NV + 1)  # 8 moves + stop

        self._reset_episode_vars()

    # -----------------------------
    # Utility functions
    # -----------------------------
    def _angle(self, vid):
        return 2*np.pi * (vid % self.NV) / self.NV

    def _move_dist(self, v_from, v_to):
        dtheta = abs(self._angle(v_from) - self._angle(v_to)) % (2*np.pi)
        dtheta = min(dtheta, 2*np.pi - dtheta)
        return 2 * self.radius * np.sin(dtheta/2)

    def _state_vec(self):
        onehot = np.zeros(self.NV, dtype=np.float32)
        onehot[self.view] = 1.0

        return np.concatenate([
            onehot,
            self.last_probs.astype(np.float32),
            self.visited.astype(np.float32),
        ], axis=0)

    def _obs_cnn_probs(self):
        img = self.obs_provider.get_frame(
            view_idx=self.view,
            class_idx=self.true_class,
        )
        probs = self.classifier.predict_proba(img)
        probs = np.clip(probs, EPS, 1.0)
        probs = probs / probs.sum()
        return probs

    def _build_action_mask(self):
        mask = np.ones(self.NV + 1, dtype=bool)
        mask[self.view] = False  # can't stay in same place
        if self.mask_revisit:
            mask[:self.NV] &= ~self.visited.astype(bool)
        mask[-1] = True  # stop always allowed
        return mask

    # -----------------------------
    # Gym APIs
    # -----------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_episode_vars()
        obs = self._state_vec()
        info = {"action_mask": self._build_action_mask()}
        return obs, info

    def _reset_episode_vars(self):
        self.t = 0
        self.true_class = int(self.rng.integers(0, self.C))
        self.view = int(self.rng.integers(0, self.NV))
        self.visited = np.zeros(self.NV, dtype=np.float32)
        self.visited[self.view] = 1.0
        self.last_probs = np.ones(self.C) / self.C
        self.episode_reward = 0.0

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        terminated = False
        truncated = False

        if action == self.NV:  # STOP action
            pred = int(np.argmax(self.last_probs))
            reward += self.R_ok if pred == self.true_class else -self.R_err
            self.episode_reward += reward
            terminated = True

            obs = self._state_vec()
            info = {"action_mask": self._build_action_mask(),
                    "pred": pred, "true": self.true_class,
                    "episode": {"r": float(self.episode_reward),
                                "l": int(self.t)}}
            return obs, reward, terminated, truncated, info

        # MOVE ACTION
        v_next = int(action)

        # movement penalty
        reward -= (self.lambda1 * self._move_dist(self.view, v_next) + self.lambda2)
        self.episode_reward += reward

        self.view = v_next
        self.visited[self.view] = 1.0

        # CNN observation (NO belief update)
        self.last_probs = self._obs_cnn_probs()

        self.t += 1
        if self.t >= self.Tmax:
            pred = int(np.argmax(self.last_probs))
            reward += self.R_ok if pred == self.true_class else -self.R_err
            self.episode_reward += reward
            terminated = True

        obs = self._state_vec()
        info = {"action_mask": self._build_action_mask(),
                "pred": int(np.argmax(self.last_probs)),
                "true": self.true_class}

        if terminated:
            info["episode"] = {"r": float(self.episode_reward),
                               "l": int(self.t)}

        return obs, reward, terminated, truncated, info