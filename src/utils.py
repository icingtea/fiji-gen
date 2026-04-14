import torch
import matplotlib.pyplot as plt
from typing import List


def plot_training(rewards: List[float], losses: List[float], best: List[float]) -> None:
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title("Rewards")

    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title("Loss")

    plt.subplot(3, 1, 3)
    plt.plot(best)
    plt.title("Best Fitness")

    plt.tight_layout()
    plt.show()


def generate_instance(num_jobs: int, num_machines: int) -> torch.Tensor:
    return torch.randint(1, 100, (num_jobs, num_machines))
