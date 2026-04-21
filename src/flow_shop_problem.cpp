// flow_shop_problem.cpp
// Sequential and parallel island-model GA for the Permutation Flow Shop Scheduling Problem.
// Genome = permutation of jobs (std::vector<int>).
// Objective = minimise makespan (fitness stored as -makespan so higher = better).
// Run with: ./flow_shop_problem [seq|par]

#include "ga.hpp"
#include "par_ga.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

// ============================================================
// Problem instance — 50 jobs × 10 machines, processing times in [1, 99]
// Fixed seed so the instance is reproducible.
// ============================================================

static constexpr int N_JOBS     = 50;
static constexpr int N_MACHINES = 10;
static constexpr int MAX_PT     = 99;
static constexpr unsigned INSTANCE_SEED = 42;

static std::vector<std::vector<int>> make_instance() {
    std::mt19937 g(INSTANCE_SEED);
    std::uniform_int_distribution<int> d(1, MAX_PT);
    std::vector<std::vector<int>> pt(N_JOBS, std::vector<int>(N_MACHINES));
    for (auto& row : pt)
        for (auto& v : row)
            v = d(g);
    return pt;
}

static const std::vector<std::vector<int>> PROCESSING_TIMES = make_instance();

static double compute_lower_bound() {
    double lb = 0.0;
    for (int m = 0; m < N_MACHINES; ++m) {
        double s = 0.0;
        for (int j = 0; j < N_JOBS; ++j)
            s += PROCESSING_TIMES[j][m];
        lb = std::max(lb, s);
    }
    return lb;
}

static const double LOWER_BOUND = compute_lower_bound();

// ============================================================
// FlowShopGA — Standard GA
// ============================================================

class FlowShopGA : public GA<std::vector<int>> {
  public:
    std::mt19937 rng;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};

    static constexpr int MAX_STAGNATION = 300;
    int    stagnation_count_  = 0;
    double best_ms_seen_      = std::numeric_limits<double>::max();

    FlowShopGA(int pop_size, double mut_rate, unsigned seed)
        : GA(pop_size, mut_rate, seed), rng(seed) {}

    void init_population() override {
        population_.clear();
        stagnation_count_ = 0;
        best_ms_seen_     = std::numeric_limits<double>::max();
        generation        = 0;

        for (int i = 0; i < static_cast<int>(init_pop_size); ++i) {
            Individual<std::vector<int>> ind;
            ind.genome  = random_permutation();
            ind.fitness = compute_fitness(ind);
            population_.push_back(std::move(ind));
        }
    }

    double compute_fitness(Individual<std::vector<int>>& ind) override {
        return -compute_makespan(ind.genome);
    }

    // Select parents via Roulette Wheel equivalent logic
    std::vector<Pairing<std::vector<int>>> select_parents(
        std::vector<Individual<std::vector<int>>>& pop) override {

        std::vector<Pairing<std::vector<int>>> result;
        if (pop.empty()) return result;

        size_t n = pop.size();

        std::vector<double> weights(n);
        for (size_t k = 0; k < n; ++k)
            weights[k] = 1.0 / (-pop[k].fitness + 1e-8);
        double total = 0.0;
        for (double w : weights) total += w;

        std::uniform_real_distribution<double> spin(0.0, total);
        std::vector<size_t> pool;
        pool.reserve(n);

        for (size_t k = 0; k < n; ++k) {
            double pick = spin(rng), cum = 0.0;
            size_t chosen = n - 1;
            for (size_t j = 0; j < n; ++j) {
                cum += weights[j];
                if (cum >= pick) { chosen = j; break; }
            }
            pool.push_back(chosen);
        }

        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        for (size_t i = 0; i < n / 2; ++i)
            result.emplace_back(Pairing<std::vector<int>>{pool[pick(rng)],
                                                          pool[pick(rng)]});
        return result;
    }

    Individual<std::vector<int>> crossover(
        Pairing<std::vector<int>>& pair) override {

        const auto& p1 = population_[pair.parent_1_index].genome;
        const auto& p2 = population_[pair.parent_2_index].genome;
        return Individual<std::vector<int>>{build_child(p1, p2), 0.0};
    }

    void mutate(Individual<std::vector<int>>& ind) override {
        if (prob_dist(rng) >= mut_rate) return;

        auto& g = ind.genome;
        int sz = static_cast<int>(g.size());
        if (sz < 2) return;

        std::uniform_int_distribution<int> pos(0, sz - 1);
        int i = pos(rng), j = pos(rng);
        if (i == j) return;
        if (i > j) std::swap(i, j);

        int val = g[j];
        g.erase(g.begin() + j);
        g.insert(g.begin() + i, val);
    }

    void drop_individuals(
        std::vector<Individual<std::vector<int>>>& pop) override {
        std::sort(pop.begin(), pop.end(),
                  [](const auto& a, const auto& b) {
                      return a.fitness > b.fitness;
                  });
        if (pop.size() > init_pop_size)
            pop.resize(init_pop_size);
    }

    bool check_halt(
        std::vector<Individual<std::vector<int>>>& pop) override {
        if (pop.empty()) return true;

        auto best_it = std::max_element(
            pop.begin(), pop.end(),
            [](const auto& a, const auto& b) { return a.fitness < b.fitness; });
        double best_ms = -best_it->fitness;

        if (best_ms < best_ms_seen_ - 1e-9) {
            best_ms_seen_    = best_ms;
            stagnation_count_ = 0;
        } else {
            ++stagnation_count_;
        }

        return stagnation_count_ >= MAX_STAGNATION;
    }

  private:
    std::vector<int> random_permutation() {
        std::vector<int> perm(N_JOBS);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        return perm;
    }

    double compute_makespan(const std::vector<int>& seq) const {
        int n = static_cast<int>(seq.size());
        std::vector<std::vector<double>> C(n,
            std::vector<double>(N_MACHINES, 0.0));

        for (int i = 0; i < n; ++i) {
            int job = seq[i];
            for (int m = 0; m < N_MACHINES; ++m) {
                double pt = PROCESSING_TIMES[job][m];
                if      (i == 0 && m == 0) C[i][m] = pt;
                else if (i == 0)           C[i][m] = C[i][m-1] + pt;
                else if (m == 0)           C[i][m] = C[i-1][m] + pt;
                else    C[i][m] = std::max(C[i-1][m], C[i][m-1]) + pt;
            }
        }
        return C[n-1][N_MACHINES-1];
    }

    std::vector<int> build_child(const std::vector<int>& pa,
                                 const std::vector<int>& pb) {
        int sz = static_cast<int>(pa.size());
        std::uniform_int_distribution<int> pos(0, sz - 1);
        int ci = pos(rng), cj = pos(rng);
        if (ci > cj) std::swap(ci, cj);

        std::vector<int> child(sz, -1);
        std::set<int>    used;

        for (int k = ci; k < cj; ++k) {
            child[k] = pa[k];
            used.insert(pa[k]);
        }

        int idx_b = cj, idx_c = cj;
        while (true) {
            bool complete = true;
            for (int x : child) if (x == -1) { complete = false; break; }
            if (complete) break;

            int val = pb[idx_b % sz];
            if (used.find(val) == used.end()) {
                while (child[idx_c % sz] != -1) ++idx_c;
                child[idx_c % sz] = val;
                used.insert(val);
            }
            ++idx_b;
        }

        return child;
    }
};

// ============================================================
// Execution Modes
// ============================================================

int main_seq() {
    int pop_size = 200;
    double mut_rate = 0.05;

    std::cout << "Flow shop — " << N_JOBS << " jobs × " << N_MACHINES
              << " machines | Lower bound: " << LOWER_BOUND << "\n\n";
    std::cout << "Running sequential GA...\n";

    FlowShopGA ga(pop_size, mut_rate, 42);

    ga.init_population();
    ga.run();

    auto& pop = ga.population();
    auto best = std::max_element(pop.begin(), pop.end(), [](const auto& a, const auto& b) {
        return a.fitness < b.fitness;
    });

    std::cout << "Finished in " << ga.generation << " generations.\n";
    std::cout << "Best makespan: " << -best->fitness << "\n";

    return 0;
}

int main_par() {
    unsigned n_threads = 4;
    unsigned pop_size = 100;
    double mut_rate = 0.05;
    unsigned n_migrants = 5;
    double migration_probability = 0.2;
    unsigned quorum = 2;

    std::cout << "Flow shop — " << N_JOBS << " jobs × " << N_MACHINES
              << " machines | Lower bound: " << LOWER_BOUND << "\n\n";

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; i++) {
        seeds.push_back(1234 + i * 1337);
    }

    IslandModel<FlowShopGA> model(n_threads, migration_probability, n_migrants,
                                   pop_size, mut_rate, quorum, seeds);

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

        auto best =
            std::max_element(pop.begin(), pop.end(), [](const auto& a, const auto& b) {
                return a.fitness < b.fitness;
            });

        std::cout << "Island " << i << " best makespan: " << -best->fitness
                  << "\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [seq|par]\n";
        return 1;
    }

    std::string mode = argv[1];

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
