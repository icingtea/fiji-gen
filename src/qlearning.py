import torch
import random
from typing import Dict, Tuple, List


class QLearningAgent:
    def __init__(
        self,
        action_space: List[Tuple[str, float, float]],
        num_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.action_space = action_space
        self.n_actions = len(action_space)

        self.num_bins = num_bins

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table: Dict[Tuple[float, float], torch.Tensor] = {}

        self.fitness_min = float("inf")
        self.fitness_max = float("-inf")

        self.entropy_min = float("inf")
        self.entropy_max = float("-inf")

    def _update_ranges(self, state: torch.Tensor) -> None:
        """
        Update observed min/max for normalization.
        """
        f, e = float(state[0]), float(state[1])

        self.fitness_min = min(self.fitness_min, f)
        self.fitness_max = max(self.fitness_max, f)

        self.entropy_min = min(self.entropy_min, e)
        self.entropy_max = max(self.entropy_max, e)

    def _discretize(self, state: torch.Tensor) -> Tuple[int, int]:
        """
        Convert continuous state to discrete bins.
        """
        self._update_ranges(state)

        f, e = float(state[0]), float(state[1])

        def to_bin(x: float, x_min: float, x_max: float) -> int:
            if x_max - x_min < 1e-8:
                return 0
            val = int((x - x_min) / (x_max - x_min) * (self.num_bins - 1))
            return max(0, min(self.num_bins - 1, val))

        f_bin = to_bin(f, self.fitness_min, self.fitness_max)
        e_bin = to_bin(e, self.entropy_min, self.entropy_max)

        return (f_bin, e_bin)

    def _get_q(self, state: Tuple[float, float]) -> torch.Tensor:
        """
        Retrieve Q-values for a state.
        """
        if state not in self.q_table:
            self.q_table[state] = torch.zeros(self.n_actions)
        return self.q_table[state]

    def select_action(self, state: torch.Tensor) -> int:
        """
        ε-greedy action selection.
        """
        s = self._discretize(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        q_values = self._get_q(s)
        return int(torch.argmax(q_values).item())

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
    ) -> None:
        """
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        s = self._discretize(state)
        s_next = self._discretize(next_state)

        q = self._get_q(s)
        q_next = self._get_q(s_next)

        target = reward + self.gamma * torch.max(q_next)

        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
