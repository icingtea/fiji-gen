#include "ga.hpp"
#include "par_ga.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class RoyalRoadGA : public GA<unsigned> {
  public:
    std::mt19937 rng;

    const unsigned genome_length = 32;
    const unsigned block_size = 8;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
    std::uniform_int_distribution<unsigned> bit_dist{0u, UINT32_MAX};

    RoyalRoadGA(int pop_size, double mut_rate, unsigned seed)
        : GA(pop_size, mut_rate, seed), rng(seed) {}

    void init_population() override {
        population_.clear();

        for (unsigned i = 0; i < init_pop_size; ++i) {
            Individual<unsigned> ind;

            // Random genome
            ind.genome = bit_dist(rng);

            ind.fitness = compute_fitness(ind);
            population_.push_back(ind);
        }
    }

    double compute_fitness(Individual<unsigned>& individual) override {
        double fitness = 0.0;

        for (unsigned i = 0; i + block_size <= genome_length; i += block_size) {
            bool all_one = true;

            for (unsigned j = 0; j < block_size; ++j) {
                unsigned bit = (individual.genome >> (i + j)) & 1u;

                if (!bit) {
                    all_one = false;
                    break;
                }
            }

            if (all_one) {
                fitness += block_size;
            }
        }

        return fitness;
    }

    std::vector<Pairing<unsigned>> select_parents(
        std::vector<Individual<unsigned>>& population) override {

        std::vector<Pairing<unsigned>> result;

        if (population.empty())
            return result;

        std::uniform_int_distribution<size_t> idx(0, population.size() - 1);

        for (unsigned i = 0; i < population.size() / 2; ++i) {
            size_t i1 = idx(rng);
            size_t i2 = idx(rng);
            result.emplace_back(Pairing<unsigned>{i1, i2});
        }

        return result;
    }

    Individual<unsigned> crossover(Pairing<unsigned>& pair) override {
        Individual<unsigned> child;

        std::uniform_int_distribution<unsigned> cut_dist(1, genome_length - 1);
        unsigned cut = cut_dist(rng);

        // Create masks
        unsigned left_mask = (1u << cut) - 1;
        unsigned right_mask = ~left_mask;

        // Combine parents by index
        child.genome =
            (population_[pair.parent_1_index].genome & left_mask) |
            (population_[pair.parent_2_index].genome & right_mask);

        return child;
    }

    void mutate(Individual<unsigned>& individual) override {
        for (unsigned i = 0; i < genome_length; ++i) {
            if (prob_dist(rng) < mut_rate) {
                individual.genome ^= (1u << i);
            }
        }
    }

    void drop_individuals(
        std::vector<Individual<unsigned>>& population) override {

        std::sort(population.begin(), population.end(),
                  [](auto& a, auto& b) { return a.fitness > b.fitness; });

        if (population.size() > init_pop_size) {
            population.resize(init_pop_size);
        }
    }

    bool check_halt(std::vector<Individual<unsigned>>& population) override {
        if (population.empty())
            return true;

        auto best = std::max_element(
            population.begin(), population.end(),
            [](auto& a, auto& b) { return a.fitness < b.fitness; });

        double max_fitness = genome_length;

        return best->fitness >= max_fitness;
    }
};

int main_seq() {
    RoyalRoadGA ga(100, 0.01, 42);

    ga.init_population();
    ga.run();

    auto& pop = ga.population();

    auto best = std::max_element(pop.begin(), pop.end(), [](auto& a, auto& b) {
        return a.fitness < b.fitness;
    });

    std::cout << "Best fitness: " << best->fitness << "\n";

    return 0;
}

int main_par() {
    unsigned n_threads = 4;
    unsigned pop_size = 100;
    double mut_rate = 0.01;
    unsigned n_migrants = 5;
    double migration_probability = 0.3;
    unsigned quorum = 2;

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; i++) {
        seeds.push_back(1234 + i * 1337);
    }

    IslandModel<RoyalRoadGA> model(n_threads, migration_probability, n_migrants,
                                   pop_size, mut_rate, quorum, seeds);

    std::cout << "Running island model...\n";

    model.island_model_run();

    std::cout << "Finished.\n";

    auto populations = model.populations();

    for (unsigned i = 0; i < populations.size(); i++) {
        auto& pop = populations[i];

        auto best =
            std::max_element(pop.begin(), pop.end(), [](auto& a, auto& b) {
                return a.fitness < b.fitness;
            });

        std::cout << "Island " << i << " best fitness: " << best->fitness
                  << "\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {

    // Check if argument is provided
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [seq|par]\n";
        return 1;
    }

    // Convert argument to std::string
    std::string mode = argv[1];

    // Select execution mode
    if (mode == "seq") {
        main_seq();
    } else if (mode == "par") {
        main_par();
    } else {
        std::cout << "Invalid argument. Use 'seq' or 'par'.\n";
        return 1;
    }

    return 0;
}