#include <iostream>
#include <random>
#include <algorithm>
#include "par_ga.hpp"

// -------------------- Dummy GA --------------------

class DummyGA : public GA<double> {
public:
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist{-100.0, 100.0};

    DummyGA(unsigned pop_size, double mut_rate, unsigned seed)
        : GA(pop_size, mut_rate, seed), rng(seed) {}

    void init_population() override {
        population_.clear();
        for (unsigned i = 0; i < init_pop_size; i++) {
            Individual<double> ind;
            ind.genome = dist(rng);
            ind.fitness = 0.0;
            population_.push_back(ind);
        }
    }

    double compute_fitness(Individual<double>& ind) override {
        // maximize peak at 42
        double x = ind.genome;
        return -((x - 42.0) * (x - 42.0));
    }

    std::vector<Pairing<double>> select_parents() override {
        std::vector<Pairing<double>> pairs;

        std::uniform_int_distribution<unsigned> idx(0, population_.size() - 1);

        for (unsigned i = 0; i < population_.size() / 2; i++) {
            auto& p1 = population_[idx(rng)];
            auto& p2 = population_[idx(rng)];
            pairs.push_back({p1, p2});
        }

        return pairs;
    }

    Individual<double> crossover(Pairing<double>& pair) override {
        Individual<double> child;
        child.genome = (pair.parent_1.genome + pair.parent_2.genome) / 2.0;
        return child;
    }

    void mutate(Individual<double>& ind) override {
        std::normal_distribution<double> noise(0.0, mut_rate);
        ind.genome += noise(rng);
    }

    void drop_individuals(std::vector<Individual<double>>& population) override {
        // keep top half
        std::sort(population.begin(), population.end(),
            [](auto& a, auto& b) { return a.fitness > b.fitness; });

        population.resize(init_pop_size);
    }

    bool check_halt(std::vector<Individual<double>>& population) override {
        // stop if close to 42
        auto best = std::max_element(population.begin(), population.end(),
            [](auto& a, auto& b) { return a.fitness < b.fitness; });

        return best->fitness > -1e-3; // near optimal
    }
};

// -------------------- Main --------------------

int main() {
    unsigned n_threads = 4;
    unsigned pop_size = 50;
    double mut_rate = 1.0;
    unsigned n_migrants = 5;
    double migration_probability = 0.3;
    unsigned quorum = 2;

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; i++) {
        seeds.push_back(1234 + i * 1337);
    }

    IslandModel<DummyGA> model(
        n_threads,
        migration_probability,
        n_migrants,
        pop_size,
        mut_rate,
        quorum,
        seeds
    );

    std::cout << "Running island model...\n";

    model.island_model_run();

    std::cout << "Finished.\n";

    auto populations = model.populations();

    for (unsigned i = 0; i < populations.size(); i++) {
        auto& pop = populations[i];

        auto best = std::max_element(pop.begin(), pop.end(),
            [](auto& a, auto& b) { return a.fitness < b.fitness; });

        std::cout << "Island " << i << " best: "
                  << best->genome << " (fitness: "
                  << best->fitness << ")\n";
    }

    return 0;
}