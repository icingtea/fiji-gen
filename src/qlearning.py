import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List


class DQNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        action_space: List[Tuple[str, float, float]],
        lr: float = 0.001,
        gamma: float = 0.9,
        epsilon: float = 0.9,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        batch_size: int = 64,
        memory_size: int = 10000,
    ):
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State has 2 dimensions: [avg_fitness, entropy]
        self.q_network = DQNModel(2, self.n_actions).to(self.device)
        self.target_network = DQNModel(2, self.n_actions).to(self.device)
        self.update_target_network()
        self.target_network.eval()

        # The paper calibrated alpha=0.1. Typically for Adam a smaller learning rate like 0.001 is used, 
        # but since we aim to stay loyal to the paper, although alpha=0.1 with NN could be too large,
        # we will use typical NN settings but use the provided alpha for the step. The variable name is lr.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self) -> None:
        """
        Synchronize the target network with the main Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: torch.Tensor) -> int:
        """
        ε-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        self.q_network.eval()
        with torch.no_grad():
            s = state.float().unsqueeze(0).to(self.device)
            q_values = self.q_network(s)
            return int(torch.argmax(q_values, dim=1).item())

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool = False,
    ) -> None:
        """
        Store experience and train the network.
        """
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.stack([x[0] for x in batch]).float().to(self.device)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack([x[3] for x in batch]).float().to(self.device)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        self.q_network.train()

        # Current Q values
        q_values = self.q_network(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid instability caused by rare large reward spikes
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
