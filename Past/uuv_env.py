import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPS = 1e-12

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + EPS)

class AUVViewEnv(gym.Env):
    """Single-target, 8 viewpoints, C-class recognition with belief-PPO."""
    metadata = {"render_modes": []}

    def __init__(self,
                 C=3,
                 radius=1.0,
                 tau=0.90,
                 Tmax=12,
                 lambda1=1.0,
                 lambda2=0.05,
                 R_ok=10.0,
                 R_err=12.0,
                 T_star=1.5,
                 mask_revisit=True,
                 seed=None):
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

        # ppo's input definition
        obs_dim = self.NV + self.C + self.NV   # onehot(view) + belief + visited_mask
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NV + 1)   # 8 move + stop

        # confusion matrix (arbitarily)
        base = np.eye(self.C) * 0.85 + (np.ones((self.C, self.C)) - np.eye(self.C)) * (0.15/(self.C-1))
        self.M = base.astype(np.float64)

        self._reset_episode_vars()

    # functions
    def _angle(self, vid):
        return 2*np.pi * (vid % self.NV) / self.NV

    def _angdist(self, a, b):
        d = abs(a-b) % (2*np.pi)
        return min(d, 2*np.pi - d)

    def _move_dist(self, v_from, v_to):
        dtheta = self._angdist(self._angle(v_from), self._angle(v_to))
        return 2 * self.radius * np.sin(dtheta/2)  # chord length

    def _state_vec(self):
        onehot = np.zeros(self.NV, dtype=np.float32)
        onehot[self.view] = 1.0
        return np.concatenate([onehot, self.belief.astype(np.float32), self.visited.astype(np.float32)], axis=0)

    def _obs_likelihood(self, true_class):
        """Return temperature-scaled class probability vector length C."""
        p = self.M[true_class].copy()                 # base probs per true class
        logits = np.log(p + EPS)                      # logits consistent with p
        pT = softmax(logits / self.T_star)            # temperature scaling
        return pT

    def _build_action_mask(self):
        mask = np.ones(self.NV + 1, dtype=bool)
        # forbid moving to current view
        mask[self.view] = False
        # optionally forbid revisits
        if self.mask_revisit:
            mask[:self.NV] &= ~self.visited.astype(bool)  # False where visited==1
        mask[-1] = True  # stop always allowed
        return mask

    # ---------- Gym APIs ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_episode_vars()
        obs = self._state_vec()
        info = {"action_mask": self._build_action_mask()}
        return obs, info

    def _reset_episode_vars(self):
        self.t = 0
        self.true_class = int(self.rng.integers(0, self.C))  # random target class
        self.view = int(self.rng.integers(0, self.NV))
        self.visited = np.zeros(self.NV, dtype=np.float32)
        self.visited[self.view] = 1.0
        self.belief = np.ones(self.C, dtype=np.float64) / self.C
        self.done = False

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        terminated = False
        truncated = False

        if action == self.NV:  # stop
            pred = int(np.argmax(self.belief))
            reward += self.R_ok if pred == self.true_class else -self.R_err
            terminated = True
            obs = self._state_vec()
            info = {"action_mask": self._build_action_mask(),
                    "pred": pred, "true": self.true_class}
            return obs, reward, terminated, truncated, info

        # move
        v_next = int(action)
        # movement penalty
        reward -= (self.lambda1 * self._move_dist(self.view, v_next) + self.lambda2)

        # transition
        self.view = v_next
        self.visited[self.view] = 1.0

        # observation likelihood (from simulator/classifier)
        O = self._obs_likelihood(self.true_class)  # len C
        # belief update
        self.belief = O * self.belief
        self.belief = self.belief / (np.sum(self.belief) + EPS)

        # stopping rule by MAP threshold
        self.t += 1
        if (np.max(self.belief) >= self.tau) or (self.t >= self.Tmax):
            pred = int(np.argmax(self.belief))
            reward += self.R_ok if pred == self.true_class else -self.R_err
            terminated = True

        obs = self._state_vec()
        info = {"action_mask": self._build_action_mask(),
                "pred": int(np.argmax(self.belief)), "true": self.true_class}
        return obs, reward, terminated, truncated, info