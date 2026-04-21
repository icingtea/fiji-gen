// deceptive_royal_road_problem.cpp
// Sequential and parallel island-model GA for the Deceptive Royal Road Problem.
// Genome = 128-bit string via std::vector<uint8_t>.
// The 128-bit genome is split into 16 evaluation blocks of size 8.
// Fitness per block: If fully ones, +16. Else, +(8 - number_of_ones).
// A strong local minimum therefore forms around all-zeros.
// Run with: ./deceptive_royal_road_problem [seq|par]

#include "ga.hpp"
#include "par_ga.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <map>
#include <fstream>
#include <sstream>
#include <string>

// ============================================================
// Utility: Load Parameters from File
// ============================================================
std::map<std::string, std::string> load_params(const std::string& filename) {
    std::map<std::string, std::string> params;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open parameter file " << filename << ". Using defaults.\n";
        return params;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key;
        if (iss >> key) {
            std::string remainder;
            std::getline(iss, remainder);
            size_t start = remainder.find_first_not_of(" \t");
            if (start != std::string::npos) {
                params[key] = remainder.substr(start);
            }
        }
    }
    return params;
}

// ============================================================
// DeceptiveRoyalRoadGA — Non-RL GA Baseline
// ============================================================

class DeceptiveRoyalRoadGA : public GA<std::vector<uint8_t>> {
  public:
    std::mt19937 rng;
    unsigned GENOME_LENGTH;
    unsigned BLOCK_SIZE;
    unsigned NUM_BLOCKS;
    double MAX_FITNESS;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
    std::uniform_int_distribution<uint8_t> bit_dist{0, 1};

    DeceptiveRoyalRoadGA(int pop_size, double mut_rate, unsigned seed, unsigned genome_length, unsigned block_size)
        : GA(pop_size, mut_rate, seed), rng(seed), GENOME_LENGTH(genome_length), BLOCK_SIZE(block_size),
          NUM_BLOCKS(genome_length / block_size), MAX_FITNESS(static_cast<double>((genome_length / block_size) * (block_size * 2))) {}

    void init_population() override {
        population_.clear();
        generation = 0;

        for (int i = 0; i < static_cast<int>(init_pop_size); ++i) {
            Individual<std::vector<uint8_t>> ind;
            ind.genome.resize(GENOME_LENGTH);
            for (unsigned j = 0; j < GENOME_LENGTH; ++j) {
                ind.genome[j] = bit_dist(rng);
            }
            ind.fitness = compute_fitness(ind);
            population_.push_back(std::move(ind));
        }
    }

    double compute_fitness(Individual<std::vector<uint8_t>>& ind) override {
        double fitness = 0.0;

        for (unsigned i = 0; i < NUM_BLOCKS; ++i) {
            unsigned ones = 0;
            for (unsigned j = 0; j < BLOCK_SIZE; ++j) {
                if (ind.genome[i * BLOCK_SIZE + j] == 1) {
                    ones++;
                }
            }

            if (ones == BLOCK_SIZE) {
                fitness += static_cast<double>(BLOCK_SIZE * 2); // Global peak (R_opt)
            } else {
                fitness += static_cast<double>(BLOCK_SIZE - ones); // Local deceptive path
            }
        }
        return fitness;
    }

    std::vector<Pairing<std::vector<uint8_t>>> select_parents(
        std::vector<Individual<std::vector<uint8_t>>>& pop) override {

        std::vector<Pairing<std::vector<uint8_t>>> result;
        if (pop.empty())
            return result;

        std::uniform_int_distribution<size_t> idx(0, pop.size() - 1);

        // Simple random parent selection (from royal_road_problem.cpp)
        for (size_t i = 0; i < pop.size() / 2; ++i) {
            result.emplace_back(
                Pairing<std::vector<uint8_t>>{idx(rng), idx(rng)});
        }
        return result;
    }

    // Two-Point Crossover for bit vectors
    Individual<std::vector<uint8_t>> crossover(
        Pairing<std::vector<uint8_t>>& pair) override {

        Individual<std::vector<uint8_t>> child;
        child.genome.resize(GENOME_LENGTH);

        std::uniform_int_distribution<unsigned> cut_dist(1, GENOME_LENGTH - 1);
        unsigned cut1 = cut_dist(rng);
        unsigned cut2 = cut_dist(rng);

        if (cut1 > cut2) std::swap(cut1, cut2);

        const auto& p1 = population_[pair.parent_1_index].genome;
        const auto& p2 = population_[pair.parent_2_index].genome;

        // p1 -> p2 -> p1
        std::copy(p1.begin(), p1.begin() + cut1, child.genome.begin());
        std::copy(p2.begin() + cut1, p2.begin() + cut2, child.genome.begin() + cut1);
        std::copy(p1.begin() + cut2, p1.end(), child.genome.begin() + cut2);

        return child;
    }

    void mutate(Individual<std::vector<uint8_t>>& ind) override {
        for (unsigned i = 0; i < GENOME_LENGTH; ++i) {
            if (prob_dist(rng) < mut_rate) {
                ind.genome[i] ^= 1; // bit flip
            }
        }
    }

    void drop_individuals(
        std::vector<Individual<std::vector<uint8_t>>>& pop) override {
        std::sort(pop.begin(), pop.end(), [](const auto& a, const auto& b) {
            return a.fitness > b.fitness;
        });
        if (pop.size() > init_pop_size) {
            pop.resize(init_pop_size);
        }
    }

    bool check_halt(
        std::vector<Individual<std::vector<uint8_t>>>& pop) override {
        if (pop.empty())
            return true;

        auto best_it = std::max_element(
            pop.begin(), pop.end(),
            [](const auto& a, const auto& b) { return a.fitness < b.fitness; });

        // Maximum possible fitness is MAX_FITNESS
        return best_it->fitness >= MAX_FITNESS;
    }
};

// ============================================================
// Execution Modes
// ============================================================

double get_param(const std::map<std::string, std::string>& params, const std::string& key, double def) {
    auto it = params.find(key);
    if (it != params.end()) return std::stod(it->second);
    return def;
}

int get_param(const std::map<std::string, std::string>& params, const std::string& key, int def) {
    auto it = params.find(key);
    if (it != params.end()) return std::stoi(it->second);
    return def;
}

int main_seq(const std::map<std::string, std::string>& params) {
    int pop_size = get_param(params, "pop_size", 1000);
    double mut_rate = get_param(params, "mut_rate", 0.05);
    unsigned genome_length = static_cast<unsigned>(get_param(params, "GENOME_LENGTH", 128));
    unsigned block_size = static_cast<unsigned>(get_param(params, "BLOCK_SIZE", 8));

    std::cout << "=== Deceptive Royal Road Problem ===\n";
    std::cout << genome_length << " bits | " << (genome_length / block_size) << " Blocks | Max Fitness: " 
              << (genome_length / block_size) * (block_size * 2) << " \n\n";
    std::cout << "Running sequential GA...\n";

    DeceptiveRoyalRoadGA ga(pop_size, mut_rate, 42, genome_length, block_size);

    ga.init_population();
    ga.run();

    auto& pop = ga.population();
    auto best = std::max_element(
        pop.begin(), pop.end(),
        [](const auto& a, const auto& b) { return a.fitness < b.fitness; });

    std::cout << "Finished in " << ga.generation << " generations.\n";
    std::cout << "Best fitness: " << best->fitness << "\n";

    return 0;
}

int main_par(const std::map<std::string, std::string>& params) {
    unsigned n_threads = get_param(params, "n_threads", 4);
    unsigned pop_size = get_param(params, "pop_size", 10000);
    double mut_rate = get_param(params, "mut_rate", 0.05); // Higher mutation often better on deceptive traps
    unsigned n_migrants = get_param(params, "n_migrants", 500);
    double migration_probability = get_param(params, "migration_probability", 0.5);
    unsigned quorum = get_param(params, "quorum", 2);
    unsigned genome_length = static_cast<unsigned>(get_param(params, "GENOME_LENGTH", 128));
    unsigned block_size = static_cast<unsigned>(get_param(params, "BLOCK_SIZE", 8));

    std::cout << "=== Deceptive Royal Road Problem ===\n";
    std::cout << genome_length << " bits | " << (genome_length / block_size) << " Blocks | Max Fitness: " 
              << (genome_length / block_size) * (block_size * 2) << " \n\n";

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; i++) {
        seeds.push_back(1234 + i * 1337);
    }

    std::vector<DeceptiveRoyalRoadGA> agents;
    agents.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; i++) {
        agents.emplace_back(pop_size, mut_rate, seeds[i], genome_length, block_size);
    }

    IslandModel<DeceptiveRoyalRoadGA> model(n_threads, migration_probability,
                                            n_migrants, quorum, std::move(agents), seeds);

    std::cout << "Running island model (threads=" << n_threads
              << ", pop_per_island=" << pop_size << ")...\n";

    model.island_model_run();

    std::cout << "\n--- Results ---\n";

    auto populations = model.populations();

    for (unsigned i = 0; i < populations.size(); i++) {
        auto& pop = populations[i];
        if (pop.empty()) {
            std::cout << "Island " << i << ": empty\n";
            continue;
        }

        auto best = std::max_element(
            pop.begin(), pop.end(),
            [](const auto& a, const auto& b) { return a.fitness < b.fitness; });

        std::cout << "Island " << i << " best fitness: " << best->fitness
                  << " (generations: " << model.generations()[i] << ")\n";
    }

    return 0;
}

// Entry point
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [seq|par] [params.txt (optional)]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::map<std::string, std::string> params;

    if (argc >= 3) {
        params = load_params(argv[2]);
    }

    if (mode == "seq") {
        main_seq(params);
    } else if (mode == "par") {
        main_par(params);
    } else {
        std::cout << "Invalid argument. Use 'seq' or 'par'.\n";
        return 1;
    }

    return 0;
}
