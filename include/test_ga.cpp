#include "ga.hpp"
#include <algorithm>
#include <iostream>
#include <random>

// Dummy GA matching first implementation
class DummyGA : public GA<double> {
  public:
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist{-100.0, 100.0};

    DummyGA(int pop_size, double mut_rate, unsigned seed)
        : GA(pop_size, mut_rate, seed), rng(seed) {}

    void init_population() override {
        population_.clear();

        for (unsigned i = 0; i < init_pop_size; ++i) {
            Individual<double> ind;

            ind.genome = dist(rng);

            // Match first version: fitness initialized to 0
            ind.fitness = 0.0;

            population_.push_back(ind);
        }
    }

    double compute_fitness(Individual<double>& individual) override {
        double x = individual.genome;

        // Same objective: maximize peak at 42
        return -((x - 42.0) * (x - 42.0));
    }

    std::vector<Pairing<double>> select_parents() override {
        std::vector<Pairing<double>> result;

        std::uniform_int_distribution<unsigned> idx(0, population_.size() - 1);

        for (unsigned i = 0; i < population_.size() / 2; ++i) {
            auto& p1 = population_[idx(rng)];
            auto& p2 = population_[idx(rng)];

            result.push_back(Pairing<double>{p1, p2});
        }

        return result;
    }

    Individual<double> crossover(Pairing<double>& pair) override {
        Individual<double> child;

        child.genome = (pair.parent_1.genome + pair.parent_2.genome) / 2.0;

        return child;
    }

    void mutate(Individual<double>& individual) override {
        // Match first version: always mutate using mut_rate as stddev
        std::normal_distribution<double> noise(0.0, mut_rate);

        individual.genome += noise(rng);
    }

    void drop_individuals(
        std::vector<Individual<double>>& population) override {

        // Match first version: sort by fitness descending
        std::sort(population.begin(), population.end(),
                  [](auto& a, auto& b) { return a.fitness > b.fitness; });

        // Keep top init_pop_size
        population.resize(init_pop_size);
    }

    bool check_halt(std::vector<Individual<double>>& population) override {

        // Find best individual
        auto best = std::max_element(
            population.begin(), population.end(),
            [](auto& a, auto& b) { return a.fitness < b.fitness; });

        // Stop when near optimum (same threshold)
        return best->fitness > -1e-3;
    }
};

int main() {
    DummyGA ga(50, 1.0, 42);

    ga.init_population();
    ga.run();

    auto& pop = ga.population();

    auto best = std::max_element(pop.begin(), pop.end(), [](auto& a, auto& b) {
        return a.fitness < b.fitness;
    });

    std::cout << "Best x: " << best->genome << "\n";
    std::cout << "Fitness: " << best->fitness << "\n";
}