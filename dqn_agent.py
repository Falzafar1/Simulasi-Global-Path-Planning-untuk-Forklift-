"""
dqn_agent.py — DQN (Deep Q-Network) untuk robot navigation.

Aksi diskrit robot (4 aksi):
    0: maju
    1: mundur
    2: rotasi kiri
    3: rotasi kanan

State vektor:
    [lidar_0, ..., lidar_N-1,   # jarak ternormalisasi [0,1]
     dist_to_goal,               # jarak ke goal ternormalisasi
     angle_to_goal]              # sudut ke goal (sin, cos)
"""

import math
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ══════════════════════════════════════════════
#  HYPERPARAMETER DEFAULT
# ══════════════════════════════════════════════
DEFAULT_GAMMA       = 0.99
DEFAULT_LR          = 1e-3
DEFAULT_BATCH       = 64
DEFAULT_BUFFER_SIZE = 50_000
DEFAULT_EPS_START   = 1.0
DEFAULT_EPS_END     = 0.05
DEFAULT_EPS_DECAY   = 0.995     # per episode
DEFAULT_TARGET_UPDATE = 10      # episode

ACTIONS = ["forward", "backward", "rotate_left", "rotate_right"]
N_ACTIONS = len(ACTIONS)

# Reward
REWARD_GOAL      =  100.0
REWARD_COLLISION = -10.0
REWARD_STEP      =  -0.1
REWARD_CLOSER    =   2.0   # bonus mendekati goal
REWARD_PROXIMITY =   1.0   # bonus flat saat jarak < 1.0 m dari goal
REWARD_BACKWARD  =  -0.2   # penalti saat pakai aksi mundur (cegah robot mundur ke goal)
REWARD_ALIGNMENT =   0.3   # bonus proporsional cos(angle_to_goal): dorong robot menghadap goal

ACTION_BACKWARD  = 1       # indeks aksi "backward" di list ACTIONS


# ══════════════════════════════════════════════
#  JARINGAN DQN
# ══════════════════════════════════════════════
class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


# ══════════════════════════════════════════════
#  DQN AGENT
# ══════════════════════════════════════════════
class DQNAgent:
    def __init__(self, state_dim: int,
                 gamma:        float = DEFAULT_GAMMA,
                 lr:           float = DEFAULT_LR,
                 batch_size:   int   = DEFAULT_BATCH,
                 buffer_size:  int   = DEFAULT_BUFFER_SIZE,
                 eps_start:    float = DEFAULT_EPS_START,
                 eps_end:      float = DEFAULT_EPS_END,
                 eps_decay:    float = DEFAULT_EPS_DECAY,
                 target_update:int   = DEFAULT_TARGET_UPDATE,
                 device:       str   = None):

        self.state_dim     = state_dim
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.target_update = target_update
        self.episode_count = 0

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = DQNNetwork(state_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self.losses    : list[float] = []
        self.ep_rewards: list[float] = []

    # ── policy ────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            qs = self.policy_net(s)
            return int(qs.argmax(1).item())

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Greedy (inference, no exploration)."""
        with torch.no_grad():
            s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(s).argmax(1).item())

    # ── training step ─────────────────────────
    def push(self, state, action, reward, next_state, done):
        self.buffer.push(
            np.array(state, dtype=np.float32),
            action, reward,
            np.array(next_state, dtype=np.float32),
            float(done))

    def optimize(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        b     = Transition(*zip(*batch))

        states      = torch.FloatTensor(np.array(b.state)).to(self.device)
        actions     = torch.LongTensor(b.action).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(b.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(b.next_state)).to(self.device)
        dones       = torch.FloatTensor(b.done).to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target: r + γ·max Q'(s',a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        l = loss.item()
        self.losses.append(l)
        return l

    def end_episode(self, ep_reward: float):
        """Panggil di akhir setiap episode."""
        self.ep_rewards.append(ep_reward)
        self.episode_count += 1
        # Decay epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        # Sync target network
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── save / load ───────────────────────────
    def save(self, path: str):
        torch.save({
            "policy_state":  self.policy_net.state_dict(),
            "target_state":  self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "eps":           self.eps,
            "episode_count": self.episode_count,
            "ep_rewards":    self.ep_rewards,
            "losses":        self.losses[-1000:],
            "state_dim":     self.state_dim,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = None) -> "DQNAgent":
        data   = torch.load(path, map_location="cpu")
        agent  = cls(state_dim=data["state_dim"], device=device)
        agent.policy_net.load_state_dict(data["policy_state"])
        agent.target_net.load_state_dict(data["target_state"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.eps           = data.get("eps", DEFAULT_EPS_END)
        agent.episode_count = data.get("episode_count", 0)
        agent.ep_rewards    = data.get("ep_rewards", [])
        agent.losses        = data.get("losses", [])
        return agent

    # ── stats ─────────────────────────────────
    @property
    def avg_reward_100(self) -> float:
        if not self.ep_rewards: return 0.0
        return float(np.mean(self.ep_rewards[-100:]))

    @property
    def avg_loss_100(self) -> float:
        if not self.losses: return 0.0
        return float(np.mean(self.losses[-100:]))


# ══════════════════════════════════════════════
#  FUNGSI BANTU: BUAT STATE DARI ENV + LIDAR
# ══════════════════════════════════════════════
def make_state(lidar_data: list[dict], max_range: float,
               robot_x: float, robot_y: float, robot_theta: float,
               goal_x: float, goal_y: float,
               arena_w: float = 10.0, arena_h: float = 10.0) -> np.ndarray:
    """
    Susun vektor state:
      [d0/max,...,dN/max, dist_norm, sin(ang), cos(ang)]

    arena_w, arena_h digunakan untuk normalisasi diagonal yang akurat.
    Default 10x10 untuk kompatibilitas mundur.
    """
    lidar_norm = np.array([r["distance"] / max_range for r in lidar_data],
                          dtype=np.float32)
    dx   = goal_x - robot_x
    dy   = goal_y - robot_y
    dist = math.hypot(dx, dy)
    diag = math.hypot(arena_w, arena_h)    # diagonal sesuai ukuran arena aktual
    dist_norm = min(1.0, dist / diag)
    ang  = math.atan2(dy, dx) - robot_theta
    return np.concatenate([lidar_norm,
                           [dist_norm, math.sin(ang), math.cos(ang)]]).astype(np.float32)


def compute_reward(prev_dist: float, curr_dist: float,
                   hit: bool, reached_goal: bool,
                   action: int = -1,
                   angle_to_goal: float = 0.0,
                   goal_threshold: float = 0.5) -> float:
    """Hitung reward satu langkah.

    Reward shaping:
    - REWARD_CLOSER    : proporsional terhadap jarak yang ditempuh menuju goal
    - REWARD_PROXIMITY : bonus flat saat jarak < 1.0 m, cegah agent acuh di dekat goal
    - REWARD_BACKWARD  : penalti saat aksi mundur, cegah robot belajar mundur ke goal
    - REWARD_ALIGNMENT : bonus cos(angle_to_goal), dorong robot menghadap goal dulu

    Parameters
    ----------
    action        : indeks aksi yang diambil (0=forward,1=backward,2=rot_left,3=rot_right)
    angle_to_goal : sudut relatif robot ke goal (radian), sudah ternormalisasi [-pi, pi]
    """
    if reached_goal:
        return REWARD_GOAL
    if hit:
        return REWARD_COLLISION

    # Shaping: reward proporsional mendekati goal
    shaping = REWARD_CLOSER * (prev_dist - curr_dist)

    # Proximity bonus: dorong agent agar tetap bergerak saat sudah dekat
    proximity = REWARD_PROXIMITY if curr_dist < 1.0 else 0.0

    # Backward penalty: cegah robot belajar mundur menuju goal
    backward = REWARD_BACKWARD if action == ACTION_BACKWARD else 0.0

    # Alignment bonus: robot dapat reward lebih jika menghadap ke arah goal
    # cos(0) = 1.0 (tepat menghadap), cos(pi) = -1.0 (membelakangi)
    import math
    alignment = REWARD_ALIGNMENT * math.cos(angle_to_goal)

    return REWARD_STEP + shaping + proximity + backward + alignment