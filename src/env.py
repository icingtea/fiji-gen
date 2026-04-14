import torch
from typing import Tuple, List
from ga import (
    initialize_population,
    evaluate_population,
    select_parents,
    crossover,
    mutate,
    create_next_population,
)


class FSPEnvironment:
    __slots__ = ("processing_times", "population_size", "population", "fitness")

    def __init__(self, processing_times: torch.Tensor, population_size: int = 50):
        self.processing_times = processing_times
        self.population_size = population_size

        self.population: List[torch.Tensor] = []
        self.fitness: torch.Tensor = torch.Tensor([])

    def reset(self) -> torch.Tensor:
        self.population = initialize_population(
            self.population_size, self.processing_times.size(0)
        )
        self.fitness = evaluate_population(self.population, self.processing_times)
        return self._get_state()

    def step(self, action: Tuple[str, float, float]) -> Tuple[torch.Tensor, float]:
        """
        Args:
            action: (selection_method, selection_rate, mutation_rate)

        Returns:
            next_state, rewards
        """
        method, selection_rate, mutation_rate = action

        parents = select_parents(self.population, self.fitness, method, selection_rate)

        offspring: List[torch.Tensor] = []
        children_reward = 0.0

        for parent_1, parent_2 in zip(parents[::2], parents[1::2]):
            child_1, child_2 = crossover(parent_1, parent_2)
            child_1, child_2 = crossover(parent_1, parent_2)

            child_1 = mutate(child_1, mutation_rate)
            child_2 = mutate(child_2, mutation_rate)

            offspring.extend([child_1, child_2])

            parent_fitness = evaluate_population(
                [parent_1, parent_2], self.processing_times
            )
            children_fitness = evaluate_population(
                [child_1, child_2], self.processing_times
            )

            children_reward += float(parent_fitness.sum() - children_fitness.sum())

        old_best = float(self.fitness.min())

        self.population = create_next_population(
            self.population, offspring, self.processing_times
        )
        self.fitness = evaluate_population(self.population, self.processing_times)

        new_best = float(self.fitness.min())

        reward = old_best - new_best  # + children_reward

        return self._get_state(), reward

    def _get_state(self) -> torch.Tensor:
        # State = [avg_fitness, entropy]
        avg = self.fitness.mean()

        probs = self.fitness / self.fitness.sum()
        entropy = -(probs * torch.log(probs + 1e-8)).sum()

        return torch.stack([avg, entropy])
