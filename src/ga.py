import torch
import random
from typing import cast, List, Tuple


GENOME_LENGTH = 32
BLOCK_SIZE = 8


def initialize_population(size: int) -> List[torch.Tensor]:
    population = []
    for _ in range(size):
        val = random.randint(0, (1 << 32) - 1)
        population.append(torch.tensor(val, dtype=torch.int64))
    return population


def compute_fitness(individual: torch.Tensor) -> float:
    fitness = 0.0
    val = individual.item()
    for i in range(0, GENOME_LENGTH, BLOCK_SIZE):
        if i + BLOCK_SIZE > GENOME_LENGTH:
            break
        all_one = True
        for j in range(BLOCK_SIZE):
            bit = (cast(int, val) >> (i + j)) & 1
            if not bit:
                all_one = False
                break
        if all_one:
            fitness += BLOCK_SIZE
    return fitness


def evaluate_population(
    population: List[torch.Tensor],
) -> torch.Tensor:
    return torch.tensor(
        [compute_fitness(ind) for ind in population], dtype=torch.float32
    )


def select_parents(
    population: List[torch.Tensor],
    fitness: torch.Tensor,
    method: str,
    selection_rate: float,
) -> List[torch.Tensor]:
    n = int(len(population) * selection_rate)

    if n == 0:
        return []

    if method == "elitism":
        idx = torch.argsort(fitness, descending=True)
        return [population[i] for i in idx[:n]]

    elif method == "roulette":
        total = fitness.sum()
        if total < 1e-8:
            # All individuals have zero fitness: uniform random selection
            probs = torch.ones(len(population)) / len(population)
        else:
            probs = fitness / total
        idx = torch.multinomial(probs, n, replacement=True)
        return [population[i] for i in idx]

    elif method == "rank":
        ranks = torch.argsort(torch.argsort(fitness))  # 0..n-1
        weights = ranks.float() + 1.0  # avoid zero weights
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, n, replacement=True)
        return [population[i] for i in idx]

    else:
        raise ValueError("Unknown selection method")


def crossover(
    parent_1: torch.Tensor,
    parent_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sample cut point in [1, GENOME_LENGTH-1] so at least 1 bit comes from each parent.
    # cut=0 would make left_mask=0 (child=pure copy of other parent, no recombination).
    cut = random.randint(1, GENOME_LENGTH - 1)

    left_mask = (1 << cut) - 1          # bits [0, cut)
    right_mask = ((1 << 32) - 1) ^ left_mask  # bits [cut, 31]

    p1_val = cast(int, parent_1.item())
    p2_val = cast(int, parent_2.item())

    child_1_val = (p1_val & left_mask) | (p2_val & right_mask)
    child_2_val = (p2_val & left_mask) | (p1_val & right_mask)

    return torch.tensor(child_1_val, dtype=torch.int64), torch.tensor(
        child_2_val, dtype=torch.int64
    )


def mutate(individual: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Apply mutation with probability `rate`.

    `rate` here is treated as the probability that a mutation *event* occurs
    (matching the paper's pm semantics, which for FSP is the probability of
    applying one shift-mutation). For the Royal Road bitstring, a mutation event
    flips exactly one randomly chosen bit.

    Using rate as a per-bit probability (the old behaviour) flips ~rate*32 bits
    per call, which at rate=0.3 destroys ~9.6 bits and makes convergence impossible.
    """
    if random.random() >= rate:
        return individual
    val = cast(int, individual.item())
    i = random.randint(0, GENOME_LENGTH - 1)
    val ^= 1 << i
    return torch.tensor(val, dtype=torch.int64)


def create_next_population(
    population: List[torch.Tensor],
    offspring: List[torch.Tensor],
) -> List[torch.Tensor]:
    combined = population + offspring
    fitness = evaluate_population(combined)

    idx = torch.argsort(fitness, descending=True)
    return [combined[i] for i in idx[: len(population)]]
