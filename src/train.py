import torch
from typing import List, Tuple

from env import FSPEnvironment
from qlearning import QLearningAgent
from utils import plot_training, generate_instance


selection_methods: List[str] = ["elitism", "roulette", "rank"]

rates: List[float] = [0.3, 0.6, 0.9]

action_space: List[Tuple[str, float, float]] = [
    (s, ps, pm) for s in selection_methods for ps in rates for pm in rates
]


def train() -> None:
    processing_times: torch.Tensor = generate_instance(10, 5)

    env = FSPEnvironment(processing_times)
    agent = QLearningAgent(action_space)

    rewards_log: List[float] = []
    best_log: List[float] = []

    episodes: int = 100
    generations: int = 50

    for ep in range(episodes):
        state: torch.Tensor = env.reset()
        total_reward: float = 0.0

        for _ in range(generations):
            # Select discrete action index
            action_idx: int = agent.select_action(state)

            # Map to actual GA parameters
            action = action_space[action_idx]

            next_state, reward = env.step(action)

            # Q-learning update
            agent.update(state, action_idx, reward, next_state)

            state = next_state
            total_reward += reward

        # Decay exploration
        agent.decay_epsilon()

        best = float(env.fitness.min())

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
