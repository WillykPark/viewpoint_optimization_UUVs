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
                 obs_provider=None,    # DatasetProvider / LiveProvider (get_frame(view_idx, class_idx) 구현)
                 M_hat=None,           # for EIG
                 use_eig=False):
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
        self.M_hat = M_hat                   
        self.use_eig = use_eig    

        # ppo's input definition
        obs_dim = self.NV + self.C + self.NV + 2  # onehot(view) + belief + visited_mask + angles(cos & sin)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NV + 1)   # 8 move + stop

        if M_hat is not None:
            # CNN에서 추정한 confusion matrix 사용
            self.M_hat = np.asarray(M_hat, dtype=np.float64)
            self.M = self.M_hat.copy()          # fallback 용
        else:
            # 옛날처럼 “장난감” 행렬 생성 (fallback)
            base = np.eye(self.C) * 0.85 + (np.ones((self.C, self.C)) - np.eye(self.C)) * (0.15 / (self.C - 1))
            self.M = base.astype(np.float64)
            self.M_hat = None

        self._reset_episode_vars()

    # functions
    def _angle(self, vid):
        return 2*np.pi * (vid % self.NV) / self.NV

    def _angdist(self, a, b):
        d = abs(a-b) % (2*np.pi)
        return min(d, 2*np.pi - d)
    
    # 부호 있는 각도 차이 (-pi ~ pi)
    def _angdiff_signed(self, a, b):
        d = (a - b) % (2*np.pi)
        if d > np.pi:
            d -= 2*np.pi
        return d

    # 현재 view vs 물체 방향 상대각도를 cos,sin으로 표현
    def _rel_angle_feature(self):
        if not hasattr(self, "theta_obj"):
            return np.array([0.0, 0.0], dtype=np.float32)
        theta_view = self._angle(self.view)
        dtheta = self._angdiff_signed(theta_view, self.theta_obj)
        return np.array([np.cos(dtheta), np.sin(dtheta)], dtype=np.float32)

    def _move_dist(self, v_from, v_to):
        dtheta = self._angdist(self._angle(v_from), self._angle(v_to))
        return 2 * self.radius * np.sin(dtheta/2)  # chord length

    def _state_vec(self):
        onehot = np.zeros(self.NV, dtype=np.float32)
        onehot[self.view] = 1.0

        ang_feat = self._rel_angle_feature()  # [cos(Δθ), sin(Δθ)]

        return np.concatenate(
            [
                onehot,
                self.belief.astype(np.float32),
                self.visited.astype(np.float32),
                ang_feat.astype(np.float32),
            ],
            axis=0,
        )


    def _obs_likelihood(self):
        """
        CNN + ObsProvider 기반 observation likelihood.
        Radon Provider가 있으면 (img, theta_deg)를 받고,
        없으면 예전 confusion-matrix 기반 시뮬레이터로 fallback.
        """
        # (1) CNN/Provider 없으면 기존 confusion-matrix 사용
        if self.classifier is None or self.obs_provider is None:
            p = self.M[self.true_class].copy()
            logits = np.log(p + EPS)
            return softmax(logits / self.T_star)

        # (2) Radon Provider 지원 시 → (img, theta_deg) 반환
        if hasattr(self.obs_provider, "get_frame_with_angle"):
            img, theta_deg = self.obs_provider.get_frame_with_angle(
                view_idx=self.view,
                class_idx=self.true_class,
            )
            # 물체의 절대 orientation 저장 (디버깅/분석용)
            #self.theta_obj_deg = theta_deg
        else:
            # 일반 Provider → 이미지만 받음
            img = self.obs_provider.get_frame(
                view_idx=self.view,
                class_idx=self.true_class,
            )
            self.theta_obj_deg = None

        # (3) CNN 예측 확률 계산 (Temperature calibration 적용된 모델)
        probs = self.classifier.predict_proba(img)
        probs = np.clip(probs, EPS, 1.0)
        return probs / probs.sum()

    def _build_action_mask(self):
        mask = np.ones(self.NV + 1, dtype=bool)
        # forbid moving to current view
        mask[self.view] = False
        # optionally forbid revisits
        if self.mask_revisit:
            mask[:self.NV] &= ~self.visited.astype(bool)  # False where visited==1
        mask[-1] = True  # stop always allowed
        return mask
    
    def _compute_eig(self, belief, a):
        """
        belief: 현 시점의 신념 분포 (길이 C)
        a     : 후보 액션(뷰포인트 index)

        -> 각도 차이 Δθ = angle(view a) - theta_obj 를 이용해서
           '각도에 따라 정보량이 달라지는' EIG를 계산.
        """
        def entropy(p):
            p = np.clip(p, EPS, 1.0)
            p = p / np.sum(p)
            return -np.sum(p * np.log(p))

        # 1) 기본 관측 모델 (confusion matrix)
        M_base = self.M_hat if self.M_hat is not None else self.M   # (C,C)

        # 2) 이 액션 a 가 바라보는 viewpoint 의 각도 (rad)
        theta_view = self._angle(a)              # 0 ~ 2π
        # 3) 물체 방향 theta_obj 와의 각도 차이 Δθ
        #    (self.theta_obj 는 reset()에서 설정된다고 가정)
        dtheta = self._angdist(theta_view, self.theta_obj)

        # 4) 각도 차이에 따른 "정보량 weight" (0~1)
        #    dtheta 작을수록 weight=1 근처 / 클수록 weight→0
        sigma = np.pi / 4.0   # 하이퍼파라미터 (≈45°)
        weight = np.exp(- (dtheta ** 2) / (2.0 * sigma ** 2))

        # 5) 각도에 따라 effective confusion matrix 구성
        #    - viewpoint가 정면이면(M_base에 가깝게)
        #    - 옆/뒤면이면(완전 random classifier 에 가깝게)
        M_uniform = np.ones_like(M_base) / self.C
        M_eig = weight * M_base + (1.0 - weight) * M_uniform   # (C,C)

        # 6) 표준 EIG 계산
        #    P(o | a, b) = sum_x M_eig[o,x] * b(x)
        Po = (belief[None, :] * M_eig.T).sum(axis=1)   # (C,)

        E_H = 0.0
        for o in range(self.C):
            if Po[o] <= 0:
                continue
            # posterior b'(x) ∝ M_eig[o,x] * b(x)
            post = M_eig[:, o] * belief
            post = post / (np.sum(post) + EPS)
            E_H += Po[o] * entropy(post)

        return entropy(belief) - E_H

    # Gym APIs
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
        self.episode_reward = 0.0

        # --- 여기 추가: 물체 방향 theta_obj 설정 ---
        if hasattr(self.obs_provider, "get_frame_with_angle"):
            # 이미지를 읽지 않고, 그 클래스에서 임의 하나 골라 각도만 가져옴
            _, theta_deg = self.obs_provider.get_frame_with_angle(
                view_idx=0,  # 임의 view
                class_idx=self.true_class
            )
            self.theta_obj = np.deg2rad(theta_deg)
        else:
            self.theta_obj = 0.0

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        terminated = False
        truncated = False

        if action == self.NV:  # stop
            pred = int(np.argmax(self.belief))
            reward += self.R_ok if pred == self.true_class else -self.R_err
            terminated = True
            self.episode_reward += reward
            obs = self._state_vec()
            info = {"action_mask": self._build_action_mask(),
                    "pred": pred, "true": self.true_class}

            info["episode"] = {"r": float(self.episode_reward),
                               "l": int(self.t),}
            return obs, reward, terminated, truncated, info

        # move
        v_next = int(action)
        # movement penalty
        reward -= (self.lambda1 * self._move_dist(self.view, v_next) + self.lambda2)

        # EIG-based reward shaping
        if self.use_eig:
            info_gain = self._compute_eig(self.belief, action)
            alpha = 2.0  # scaling hyperparameter, tuneable
            reward += alpha * info_gain

        self.episode_reward += reward

        # transition    
        self.view = v_next
        self.visited[self.view] = 1.0

        # observation likelihood (from simulator/classifier)
        O = self._obs_likelihood()  # len C
        # belief update
        self.belief = O * self.belief
        self.belief = self.belief / (np.sum(self.belief) + EPS)

        # stopping rule by MAP threshold
        self.t += 1
        if (np.max(self.belief) >= self.tau) or (self.t >= self.Tmax):
            pred = int(np.argmax(self.belief))
            bonus = self.R_ok if pred == self.true_class else -self.R_err
            reward += bonus
            self.episode_reward += bonus
            terminated = True
        else:
            pred = int(np.argmax(self.belief))

        obs = self._state_vec()
        info = {"action_mask": self._build_action_mask(),
                "pred": int(np.argmax(self.belief)), "true": self.true_class}

        
        if terminated or truncated:
            info["episode"] = {
                "r": float(self.episode_reward),
                "l": int(self.t),
            }

        return obs, reward, terminated, truncated, info