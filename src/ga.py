import torch
import random
from typing import List, Tuple


def initialize_population(size: int, num_jobs: int) -> List[torch.Tensor]:
    population = []
    for _ in range(size):
        permutation = torch.randperm(num_jobs)
        population.append(permutation)

    return population


def compute_makespan(
    sequence: torch.Tensor,  # (num_jobs, )
    processing_times: torch.Tensor,  # (num_jobs, num_machines)
) -> torch.Tensor:  # scalar
    num_jobs = sequence.size(0)
    num_machines = processing_times.size(1)

    completion = torch.zeros((num_jobs, num_machines))

    for i in range(num_jobs):
        job = sequence[i]

        for m in range(num_machines):
            # Case 1: First job on first machine
            # No waiting -> directly equal to processing time
            if i == 0 and m == 0:
                completion[i, m] = processing_times[job, m]

            # Case 2: First job but not first machine
            # Must wait for previous machine to finish
            elif i == 0:
                completion[i, m] = completion[i, m - 1] + processing_times[job, m]

            # Case 3: First machine but not first job
            # Must wait for previous job on same machine
            elif m == 0:
                completion[i, m] = completion[i - 1, m] + processing_times[job, m]

            # Case 4: General case
            # Must wait for BOTH:
            # 1. Previous job on same machine
            # 2. Same job on previous machine
            # Take the maximum of the two waiting times
            else:
                completion[i, m] = (
                    torch.max(
                        completion[i - 1, m],
                        completion[i, m - 1],
                    )
                    + processing_times[job, m]
                )

    # Makespan = completion time of last job on last machine
    return completion[-1, -1]


def evaluate_population(
    population: List[torch.Tensor],  # (num_jobs, )[]
    processing_times: torch.Tensor,  # (num_jobs, num_machines)
) -> torch.Tensor:  # (population_size, )
    return torch.stack(
        [compute_makespan(individual, processing_times) for individual in population]
    )


def select_parents(
    population: List[torch.Tensor],
    fitness: torch.Tensor,
    method: str,
    selection_rate: float,
) -> List[torch.Tensor]:
    n = int(len(population) * selection_rate)

    if method == "elitism":
        idx = torch.argsort(fitness)
        return [population[i] for i in idx[:n]]

    elif method == "roulette":
        probs = 1.0 / (fitness + 1e-8)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, n, replacement=True)
        return [population[i] for i in idx]

    elif method == "rank":
        ranks = torch.argsort(torch.argsort(fitness))
        probs = 1.0 / (ranks.float() + 1)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, n, replacement=True)
        return [population[i] for i in idx]

    else:
        raise ValueError("Unknown selection method")


def crossover(
    parent_1: torch.Tensor,  # (num_jobs, )
    parent_2: torch.Tensor,  # (num_jobs, )
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mechanism:
    - Select two cut points (i < j)
    - Copy segment p1[i:j] into child
    - Fill remaining positions using p2 order, starting from index j and wrapping around
    """

    size: int = parent_1.size(0)
    i, j = sorted(random.sample(range(size), 2))

    def build_child(parent_a: torch.Tensor, parent_b: torch.Tensor) -> torch.Tensor:
        child = torch.full((size,), -1, dtype=torch.long)

        # Step 1: copy segment
        child[i:j] = parent_a[i:j]

        used = set(child[i:j].tolist())

        # Step 2: iterate parent_b circularly starting from j
        idx_b = j
        idx_c = j

        while -1 in child:
            val = parent_b[idx_b % size].item()

            if val not in used:
                # Find next empty slot in child (circular)
                while child[idx_c % size] != -1:
                    idx_c += 1

                child[idx_c % size] = val
                used.add(val)

            idx_b += 1

        return child

    child_1 = build_child(parent_1, parent_2)
    child_2 = build_child(parent_2, parent_1)

    return child_1, child_2


def mutate(individual: torch.Tensor, rate: float) -> torch.Tensor:
    if random.random() > rate:
        return individual

    i, j = sorted(random.sample(range(len(individual)), 2))
    val = individual[j].clone()
    new = torch.cat((individual[:j], individual[j + 1 :]))
    new = torch.cat((new[:i], val.unsqueeze(0), new[i:]))
    return new


def create_next_population(
    population: List[torch.Tensor],
    offspring: List[torch.Tensor],
    processing_times: torch.Tensor,
) -> List[torch.Tensor]:
    combined = population + offspring
    fitness = evaluate_population(combined, processing_times)

    idx = torch.argsort(fitness)
    return [combined[i] for i in idx[: len(population)]]
