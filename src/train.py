import torch
from typing import List, Tuple

from env import RoyalRoadEnvironment
from qlearning import DQNAgent
from utils import plot_training


selection_methods: List[str] = ["elitism", "roulette", "rank"]

rates: List[float] = [0.3, 0.6, 0.9]

action_space: List[Tuple[str, float, float]] = [
    (s, ps, pm) for s in selection_methods for ps in rates for pm in rates
]


def train() -> None:
    env = RoyalRoadEnvironment(population_size=200)
    # epsilon_decay=0.97 reaches ~0.05 in ~100 episodes; target_update_freq syncs target
    # network every N steps within an episode for more stable bootstrapping.
    agent = DQNAgent(
        action_space,
        epsilon=0.5,
        epsilon_min=0.0,
        gamma=0.9,
    )

    rewards_log: List[float] = []
    best_log: List[float] = []

    episodes: int = 200
    generations: int = 300

    for ep in range(episodes):
        state: torch.Tensor = env.reset()
        total_reward: float = 0.0

        for step in range(generations):
            # Select discrete action index
            action_idx: int = agent.select_action(state)

            # Map to actual GA parameters
            action = action_space[action_idx]

            next_state, reward = env.step(action)

            # DQN update
            agent.update(state, action_idx, reward, next_state)

            # Sync target network periodically within the episode
            # if (step + 1) % target_update_freq == 0:

            state = next_state
            total_reward += reward

        # Decay exploration after each episode
        agent.decay_epsilon()

        # Final target sync at episode boundary

        best = float(env.fitness.max())
        agent.update_target_network()

        rewards_log.append(total_reward)
        best_log.append(best)

        print(
            f"Episode {ep:03d} | "
            f"Reward: {total_reward:.2f} | "
            f"Best: {best:.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    plot_training(rewards_log, [], best_log)


if __name__ == "__main__":
    train()
