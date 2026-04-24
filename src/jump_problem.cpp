// jump_problem.cpp
// Sequential and parallel island-model GA for the Jump Problem.
// Genome = n-bit string via std::vector<uint8_t>.
// Fitness function Jump_{n,k}:
//   u(x) = sum(x)
//   if u(x) == n: fitness = n
//   if u(x) <= n - k: fitness = u(x)
//   if n - k < u(x) < n: fitness = n - u(x)
// Run with: ./jump_problem [seq|par] [params.txt]

#include "ga.hpp"
#include "par_ga.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ============================================================
// Utility: Load Parameters from File
// ============================================================
json load_params(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open parameter file " << filename << ". Using defaults.\n";
        return json::object();
    }
    json j;
    file >> j;
    return j;
}

// ============================================================
// JumpGA — Non-RL GA Baseline
// ============================================================

class JumpGA : public GA<std::vector<uint8_t>> {
  public:
    std::mt19937 rng;
    unsigned GENOME_LENGTH;
    unsigned JUMP_SIZE;
    double MAX_FITNESS;
    double selection_rate;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
    std::uniform_int_distribution<uint8_t> bit_dist{0, 1};

    JumpGA(int pop_size, double mut_rate, unsigned seed,
           unsigned genome_length, unsigned jump_size,
           double sel_rate = 0.5)
        : GA(pop_size, mut_rate, seed), rng(seed), GENOME_LENGTH(genome_length),
          JUMP_SIZE(jump_size), MAX_FITNESS(static_cast<double>(genome_length)),
          selection_rate(sel_rate) {}

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
        unsigned u = 0;
        for (unsigned i = 0; i < GENOME_LENGTH; ++i) {
            if (ind.genome[i] == 1) {
                u++;
            }
        }

        if (u == GENOME_LENGTH) {
            return static_cast<double>(GENOME_LENGTH);
        } else if (u <= GENOME_LENGTH - JUMP_SIZE) {
            return static_cast<double>(u);
        } else {
            return static_cast<double>(GENOME_LENGTH - u);
        }
    }

    std::vector<Pairing<std::vector<uint8_t>>> select_parents(
        std::vector<Individual<std::vector<uint8_t>>>& pop) override {

        std::vector<Pairing<std::vector<uint8_t>>> result;
        if (pop.empty())
            return result;

        // Elitism: top selection_rate fraction forms the mating pool
        size_t n = static_cast<size_t>(pop.size() * selection_rate);
        if (n < 2)
            n = 2;
        n = std::min(n, pop.size());

        std::vector<size_t> order(pop.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + n, order.end(),
                          [&](size_t a, size_t b) {
                              return pop[a].fitness > pop[b].fitness;
                          });

        // Random pairings drawn from the elite pool
        std::uniform_int_distribution<size_t> pick(0, n - 1);
        for (size_t i = 0; i < pop.size() / 2; ++i)
            result.emplace_back(Pairing<std::vector<uint8_t>>{order[pick(rng)],
                                                              order[pick(rng)]});
        return result;
    }

    Individual<std::vector<uint8_t>> crossover(
        Pairing<std::vector<uint8_t>>& pair) override {

        Individual<std::vector<uint8_t>> child;
        child.genome.resize(GENOME_LENGTH);

        const auto& p1 = population_[pair.parent_1_index].genome;
        const auto& p2 = population_[pair.parent_2_index].genome;

        // Choose single point crossover
        std::uniform_int_distribution<unsigned> cut_dist(1, GENOME_LENGTH - 1);
        unsigned cut = cut_dist(rng);

        std::copy(p1.begin(), p1.begin() + cut, child.genome.begin());
        std::copy(p2.begin() + cut, p2.end(), child.genome.begin() + cut);

        return child;
    }

    void mutate(Individual<std::vector<uint8_t>>& ind) override {
        // Bit-flip mutation
        for (unsigned i = 0; i < GENOME_LENGTH; ++i) {
            if (prob_dist(rng) < mut_rate) {
                ind.genome[i] ^= 1;
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

        return best_it->fitness >= MAX_FITNESS;
    }
};

// ============================================================
// Execution Modes
// ============================================================

template <typename T>
T get_nested_param(const json& params, const std::string& section, const std::string& key, T def) {
    if (params.contains(section) && params[section].contains(key)) {
        return params[section][key].get<T>();
    }
    return def;
}

json main_seq(const json& params) {
    int pop_size = get_nested_param(params, "genetic_algorithm", "pop_size", 1000);
    double mut_rate = get_nested_param(params, "genetic_algorithm", "mut_rate", 0.01);
    unsigned genome_length = static_cast<unsigned>(get_nested_param(params, "problem", "GENOME_LENGTH", 100u));
    unsigned jump_size = static_cast<unsigned>(get_nested_param(params, "problem", "JUMP_SIZE", 4u));
    double selection_rate = get_nested_param(params, "genetic_algorithm", "selection_rate", 0.5);

    std::cout << "=== Jump Problem ===\n";
    std::cout << genome_length << " bits | Jump Size: " << jump_size << " | Max Fitness: " 
              << genome_length << " \n\n";
    std::cout << "Running sequential GA...\n";

    std::random_device rd;
    JumpGA ga(pop_size, mut_rate, rd(), genome_length, jump_size, selection_rate);

    auto start_time = std::chrono::high_resolution_clock::now();

    ga.init_population();
    ga.run();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    auto& pop = ga.population();
    auto best = std::max_element(
        pop.begin(), pop.end(),
        [](const auto& a, const auto& b) { return a.fitness < b.fitness; });

    std::cout << "Finished in " << ga.generation << " generations.\n";
    std::cout << "Best fitness: " << best->fitness << "\n";
    std::cout << "Time elapsed: " << duration << " ms\n";

    json result;
    result["mode"]          = "seq";
    result["time_ms"]       = duration;
    result["genome_length"] = genome_length;
    result["jump_size"]     = jump_size;
    result["pop_size"]      = pop_size;
    result["generations"]   = ga.generation;
    result["best_fitness"]  = best->fitness;

    return result;
}

json main_par(const json& params) {
    unsigned n_threads = get_nested_param(params, "parallel", "n_threads", 4u);
    unsigned pop_size = get_nested_param(params, "genetic_algorithm", "pop_size", 10000u);
    double mut_rate = get_nested_param(params, "genetic_algorithm", "mut_rate", 0.01);
    unsigned n_migrants = get_nested_param(params, "parallel", "n_migrants", 500u);
    double migration_probability = get_nested_param(params, "parallel", "migration_probability", 0.5);
    unsigned quorum = get_nested_param(params, "parallel", "quorum", 2u);
    unsigned genome_length = static_cast<unsigned>(get_nested_param(params, "problem", "GENOME_LENGTH", 100u));
    unsigned jump_size = static_cast<unsigned>(get_nested_param(params, "problem", "JUMP_SIZE", 4u));
    double selection_rate = get_nested_param(params, "genetic_algorithm", "selection_rate", 0.5);

    std::cout << "=== Jump Problem ===\n";
    std::cout << genome_length << " bits | Jump Size: " << jump_size << " | Max Fitness: " 
              << genome_length << " \n\n";

    std::vector<unsigned> seeds;
    std::random_device rd;
    for (unsigned i = 0; i < n_threads; i++) {
        seeds.push_back(rd());
    }

    std::vector<JumpGA> agents;
    agents.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; i++) {
        agents.emplace_back(pop_size, mut_rate, seeds[i], genome_length, jump_size, selection_rate);
    }

    IslandModel<JumpGA> model(n_threads, migration_probability,
                              n_migrants, quorum, std::move(agents), seeds);

    std::cout << "Running island model (threads=" << n_threads
              << ", pop_per_island=" << pop_size << ")...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    model.island_model_run();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\n--- Results ---\n";

    auto populations = model.populations();

    json islands_data = json::array();
    double overall_best = 0.0;
    unsigned min_gens = std::numeric_limits<unsigned>::max();
    unsigned max_gens = 0;

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

        if (best->fitness > overall_best) overall_best = best->fitness;
        min_gens = std::min(min_gens, model.generations()[i]);
        max_gens = std::max(max_gens, model.generations()[i]);

        json island_info;
        island_info["island_id"] = i;
        island_info["best_fitness"] = best->fitness;
        island_info["generations"] = model.generations()[i];
        islands_data.push_back(island_info);
    }

    std::cout << "Time elapsed: " << duration << " ms\n";

    json result;
    result["mode"]                 = "par";
    result["time_ms"]              = duration;
    result["genome_length"]        = genome_length;
    result["jump_size"]            = jump_size;
    result["island_pop"]           = pop_size;
    result["n_threads"]            = n_threads;
    result["n_migrants"]           = n_migrants;
    result["overall_best_fitness"] = overall_best;
    result["min_generations"]      = min_gens;
    result["max_generations"]      = max_gens;
    result["islands"]              = islands_data;

    return result;
}

// Entry point
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [seq|par] [params.json] [output.jsonl]\n";
        return 1;
    }

    std::string mode = argv[1];
    json params = load_params(argv[2]);
    std::string output_file = argv[3];

    json result;
    if (mode == "seq") {
        result = main_seq(params);
    } else if (mode == "par") {
        result = main_par(params);
    } else {
        std::cout << "Invalid argument. Use 'seq' or 'par'.\n";
        return 1;
    }

    std::ofstream out(output_file, std::ios::app);
    if (out.is_open()) {
        out << result.dump() << "\n";
    } else {
        std::cerr << "Failed to open output file: " << output_file << "\n";
        return 1;
    }

    return 0;
}
