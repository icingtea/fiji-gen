// flow_shop_problem_rl.cpp
// DQN agent guiding a permutation GA for the Flow Shop Scheduling Problem.
// Genome = permutation of jobs (std::vector<int>).
// Objective = minimise makespan (fitness stored as -makespan so higher = better).
// Run with: ./flow_shop_problem_rl [rlga|par_rlga]

#include "par_rl_ga.hpp"
#include "rl_ga.hpp"
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

// ============================================================
// Problem instance — 20 jobs × 5 machines, processing times in [1, 20]
// Fixed seed so the instance is reproducible.
// ============================================================

static constexpr int N_JOBS     = 50;
static constexpr int N_MACHINES = 10;
static constexpr int MAX_PT     = 99;  // Taillard benchmark range [1,99]
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

// Simple lower bound: max over machines of sum of processing times on that machine.
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

static const double LOWER_BOUND        = compute_lower_bound();
static const double MAX_MAKESPAN_UB    = static_cast<double>(N_JOBS) *
                                         N_MACHINES * MAX_PT; // loose upper bound

// ============================================================
// Action space — identical to royal road: 27 (method × sel_rate × mut_rate)
// ============================================================

enum class SelectionMethod : int { ELITISM = 0, ROULETTE = 1, RANK = 2 };

struct Action {
    SelectionMethod method;
    double selection_rate;
    double mutation_rate;
};

static const std::vector<Action> ACTION_SPACE = [] {
    std::vector<Action> actions;
    const std::vector<SelectionMethod> methods = {SelectionMethod::ELITISM,
                                                  SelectionMethod::ROULETTE,
                                                  SelectionMethod::RANK};
    const std::vector<double> sel_rates = {0.3, 0.6, 0.9};
    const std::vector<double> mut_rates = {0.01, 0.05, 0.1};

    for (auto m : methods)
        for (auto sr : sel_rates)
            for (auto mr : mut_rates)
                actions.push_back({m, sr, mr});
    return actions;
}();

static const int N_ACTIONS = static_cast<int>(ACTION_SPACE.size()); // 27
static const int STATE_DIM = 2; // [normalised_avg_makespan, entropy]

// ============================================================
// DQNNet — identical architecture to royal road counterpart
// Linear(2→64)→ReLU ×5 → Linear(64→27)
// ============================================================

struct DQNNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr},
                      fc4{nullptr}, fc5{nullptr}, fc6{nullptr};

    DQNNetImpl(int in_dim, int out_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 64));
        fc4 = register_module("fc4", torch::nn::Linear(64, 64));
        fc5 = register_module("fc5", torch::nn::Linear(64, 64));
        fc6 = register_module("fc6", torch::nn::Linear(64, out_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        x = torch::relu(fc4->forward(x));
        x = torch::relu(fc5->forward(x));
        return fc6->forward(x);
    }
};
TORCH_MODULE(DQNNet);

// ============================================================
// FlowShopRLGA — RLGA<std::vector<int>, int>
// Genome: permutation of job indices [0, N_JOBS).
// Fitness stored as −makespan so higher = better throughout the framework.
// ============================================================

class FlowShopRLGA : public RLGA<std::vector<int>, int> {
  public:
    std::mt19937 rng;

    // Current GA parameters, set by apply_action()
    SelectionMethod current_method      = SelectionMethod::ELITISM;
    double          current_selection_rate = 0.5;
    double          current_mutation_rate  = 0.01;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};

    // Stagnation halt: stop if no improvement for MAX_STAGNATION generations
    static constexpr int MAX_STAGNATION = 300; // larger problem needs more patience
    int    stagnation_count_  = 0;
    double best_ms_seen_      = std::numeric_limits<double>::max();

    explicit FlowShopRLGA(int pop_size, double mut_rate, unsigned seed)
        : RLGA(pop_size, mut_rate, seed), rng(seed),
          current_mutation_rate(mut_rate) {}

    // --------------------------------------------------------
    // GA pure-virtual implementations
    // --------------------------------------------------------

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

    // fitness = −makespan  (higher = better)
    double compute_fitness(Individual<std::vector<int>>& ind) override {
        return -compute_makespan(ind.genome);
    }

    std::vector<Pairing<std::vector<int>>> select_parents(
        std::vector<Individual<std::vector<int>>>& pop) override {

        std::vector<Pairing<std::vector<int>>> result;
        if (pop.empty()) return result;

        size_t n = static_cast<size_t>(pop.size() * current_selection_rate);
        if (n < 2) n = 2;
        n = std::min(n, pop.size());

        std::vector<size_t> pool;
        pool.reserve(n);

        if (current_method == SelectionMethod::ELITISM) {
            // Top-n by fitness (= lowest makespan)
            std::vector<size_t> order(pop.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + n, order.end(),
                              [&](size_t a, size_t b) {
                                  return pop[a].fitness > pop[b].fitness;
                              });
            pool.assign(order.begin(), order.begin() + n);

        } else if (current_method == SelectionMethod::ROULETTE) {
            // prob ∝ 1/makespan  (mirrors Python: probs = 1/(fitness+eps))
            std::vector<double> weights(pop.size());
            for (size_t k = 0; k < pop.size(); ++k)
                weights[k] = 1.0 / (-pop[k].fitness + 1e-8);
            double total = 0.0;
            for (double w : weights) total += w;
            std::uniform_real_distribution<double> spin(0.0, total);
            for (size_t k = 0; k < n; ++k) {
                double pick = spin(rng), cum = 0.0;
                size_t chosen = pop.size() - 1;
                for (size_t j = 0; j < pop.size(); ++j) {
                    cum += weights[j];
                    if (cum >= pick) { chosen = j; break; }
                }
                pool.push_back(chosen);
            }

        } else { // RANK
            // rank 0 = best (highest fitness = lowest makespan)
            // prob ∝ 1/(rank+1)  (mirrors Python)
            std::vector<size_t> sorted(pop.size());
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
                return pop[a].fitness > pop[b].fitness; // descending fitness
            });
            // sorted[0] = best; assign weight 1/(0+1)=1, sorted[1] -> 1/2, ...
            std::vector<double> weights(pop.size());
            for (size_t r = 0; r < sorted.size(); ++r)
                weights[sorted[r]] = 1.0 / (double)(r + 1);
            double total = 0.0;
            for (double w : weights) total += w;
            std::uniform_real_distribution<double> spin(0.0, total);
            for (size_t k = 0; k < n; ++k) {
                double pick = spin(rng), cum = 0.0;
                size_t chosen = pop.size() - 1;
                for (size_t j = 0; j < pop.size(); ++j) {
                    cum += weights[j];
                    if (cum >= pick) { chosen = j; break; }
                }
                pool.push_back(chosen);
            }
        }

        // Pair up from pool
        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        for (size_t i = 0; i < pop.size() / 2; ++i)
            result.emplace_back(Pairing<std::vector<int>>{pool[pick(rng)],
                                                          pool[pick(rng)]});
        return result;
    }

    // Order Crossover (OX) — mirrors crossover() in Python ga.py
    Individual<std::vector<int>> crossover(
        Pairing<std::vector<int>>& pair) override {

        const auto& p1 = population_[pair.parent_1_index].genome;
        const auto& p2 = population_[pair.parent_2_index].genome;
        return Individual<std::vector<int>>{build_child(p1, p2), 0.0};
    }

    // Insertion mutation — mirrors mutate() in Python ga.py:
    // with probability rate, remove element at j and insert it at i (i < j).
    void mutate(Individual<std::vector<int>>& ind) override {
        if (prob_dist(rng) >= current_mutation_rate) return;

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

    // Keep top init_pop_size by fitness (lowest makespan)
    void drop_individuals(
        std::vector<Individual<std::vector<int>>>& pop) override {
        std::sort(pop.begin(), pop.end(),
                  [](const auto& a, const auto& b) {
                      return a.fitness > b.fitness;
                  });
        if (pop.size() > init_pop_size)
            pop.resize(init_pop_size);
    }

    // Stagnation-based halt: stop if no improvement for MAX_STAGNATION gens.
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

    // --------------------------------------------------------
    // RLGA pure-virtual implementations
    // --------------------------------------------------------

    void apply_action(int action_idx) override {
        assert(action_idx >= 0 && action_idx < N_ACTIONS);
        const Action& a = ACTION_SPACE[action_idx];
        current_method         = a.method;
        current_selection_rate = a.selection_rate;
        current_mutation_rate  = a.mutation_rate;
    }

    // Reward = improvement in best makespan (positive = improvement).
    // In −makespan space this is simply: new_best_fitness − old_best_fitness.
    double calculate_reward(
        std::span<const Individual<std::vector<int>>> old_pop,
        std::vector<Individual<std::vector<int>>>&    new_pop,
        std::vector<Individual<std::vector<int>>>&    /*children*/,
        std::vector<Pairing<std::vector<int>>>&       /*pairings*/) override {

        double old_best = std::numeric_limits<double>::lowest();
        for (const auto& ind : old_pop)
            old_best = std::max(old_best, ind.fitness);

        double new_best = std::numeric_limits<double>::lowest();
        for (const auto& ind : new_pop)
            new_best = std::max(new_best, ind.fitness);

        return new_best - old_best; // positive when makespan decreases
    }

    // DQN state: [normalised_avg_makespan, entropy_of_makespan_distribution]
    // Mirrors env.py:_get_state() adapted for minimisation.
    torch::Tensor get_dqn_state() const {
        if (population_.empty())
            return torch::zeros({STATE_DIM});

        double total_ms = 0.0;
        for (const auto& ind : population_)
            total_ms += -ind.fitness; // makespan = −fitness

        double avg_ms  = total_ms / static_cast<double>(population_.size());
        double norm_avg = std::clamp(avg_ms / MAX_MAKESPAN_UB, 0.0, 1.0);

        // Entropy using proportional makespan weights
        double sum_ms = total_ms + 1e-8;
        double entropy = 0.0;
        for (const auto& ind : population_) {
            double p = (-ind.fitness) / sum_ms;
            if (p > 0.0) entropy -= p * std::log(p + 1e-8);
        }

        return torch::tensor({(float)norm_avg, (float)entropy});
    }

  private:
    // --------------------------------------------------------
    // Helpers
    // --------------------------------------------------------

    std::vector<int> random_permutation() {
        std::vector<int> perm(N_JOBS);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        return perm;
    }

    // Compute flow shop makespan (mirrors compute_makespan() in Python ga.py)
    double compute_makespan(const std::vector<int>& seq) const {
        int n = static_cast<int>(seq.size());
        // completion[i][m]
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

    // Build one OX child from (parent_a, parent_b)
    std::vector<int> build_child(const std::vector<int>& pa,
                                 const std::vector<int>& pb) {
        int sz = static_cast<int>(pa.size());
        std::uniform_int_distribution<int> pos(0, sz - 1);
        int ci = pos(rng), cj = pos(rng);
        if (ci > cj) std::swap(ci, cj);

        std::vector<int> child(sz, -1);
        std::set<int>    used;

        // Step 1: copy pa[ci:cj] into child
        for (int k = ci; k < cj; ++k) {
            child[k] = pa[k];
            used.insert(pa[k]);
        }

        // Step 2: fill remaining slots in circular order from pb starting at cj
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
// Experience replay buffer (identical to royal road counterpart)
// ============================================================

struct Transition {
    torch::Tensor state;
    int   action;
    float reward;
    torch::Tensor next_state;
    float done;
};

class ReplayBuffer {
  public:
    explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {}

    void push(Transition t) {
        if (buffer_.size() >= capacity_) buffer_.pop_front();
        buffer_.push_back(std::move(t));
    }

    bool ready(size_t batch_size) const {
        return buffer_.size() >= batch_size;
    }

    std::vector<Transition> sample(size_t n, std::mt19937& rng) const {
        std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);
        std::vector<Transition> batch;
        batch.reserve(n);
        for (size_t i = 0; i < n; ++i)
            batch.push_back(buffer_[dist(rng)]);
        return batch;
    }

  private:
    size_t capacity_;
    std::deque<Transition> buffer_;
};

// ============================================================
// FlowShopDQNAgent — mirrors RoyalRoadDQNAgent exactly
// ============================================================

class FlowShopDQNAgent : public Agent<int, FlowShopRLGA> {
  public:
    torch::Device device;
    DQNNet q_network;
    DQNNet target_network;
    torch::optim::Adam optimizer;

    ReplayBuffer memory;
    size_t batch_size;

    std::mt19937 rng;
    std::uniform_real_distribution<double> eps_dist{0.0, 1.0};
    std::uniform_int_distribution<int>     random_action{0, N_ACTIONS - 1};

    torch::Tensor current_state;
    int last_action = 0;

    explicit FlowShopDQNAgent(FlowShopRLGA&& ga_in,
                              size_t gen_per_ep,
                              size_t n_ep,
                              double gamma,
                              double epsilon,
                              double epsilon_decay,
                              double epsilon_min,
                              double lr,
                              size_t batch_size,
                              size_t memory_size,
                              unsigned rng_seed)
        : Agent<int, FlowShopRLGA>(std::move(ga_in), gen_per_ep, n_ep,
                                   gamma, epsilon, epsilon_decay,
                                   epsilon_min, lr),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          q_network(STATE_DIM, N_ACTIONS),
          target_network(STATE_DIM, N_ACTIONS),
          optimizer(q_network->parameters(),
                    torch::optim::AdamOptions(lr)),
          memory(memory_size),
          batch_size(batch_size),
          rng(rng_seed)
    {
        q_network->to(device);
        target_network->to(device);
        sync_target();
        target_network->eval();
        current_state = ga.get_dqn_state();
    }

    void sync_target() {
        torch::NoGradGuard ng;
        auto src = q_network->parameters();
        auto dst = target_network->parameters();
        for (size_t i = 0; i < src.size(); ++i)
            dst[i].copy_(src[i]);
    }

    int select_action() override {
        current_state = ga.get_dqn_state();

        if (eps_dist(rng) < epsilon) {
            last_action = random_action(rng);
        } else {
            q_network->eval();
            torch::NoGradGuard ng;
            auto q_vals = q_network->forward(
                current_state.unsqueeze(0).to(device)); // [1, N_ACTIONS]
            last_action = static_cast<int>(
                q_vals.argmax(1).item<int64_t>());
        }
        return last_action;
    }

    void update() override {
        torch::Tensor next_state = ga.get_dqn_state();
        bool done = ga.check_halt(ga.population());

        memory.push({current_state, last_action,
                     static_cast<float>(latest_reward),
                     next_state, done ? 1.0f : 0.0f});

        if (!memory.ready(batch_size)) return;

        auto batch = memory.sample(batch_size, rng);

        std::vector<torch::Tensor> sv, nsv;
        std::vector<int64_t> av;
        std::vector<float>   rv, dv;

        for (auto& tr : batch) {
            sv.push_back(tr.state);
            nsv.push_back(tr.next_state);
            av.push_back(tr.action);
            rv.push_back(tr.reward);
            dv.push_back(tr.done);
        }

        auto states      = torch::stack(sv).to(device);
        auto next_states = torch::stack(nsv).to(device);
        auto actions     = torch::tensor(av, torch::kInt64)
                               .unsqueeze(1).to(device);
        auto rewards     = torch::tensor(rv).unsqueeze(1).to(device);
        auto dones       = torch::tensor(dv).unsqueeze(1).to(device);

        q_network->train();
        auto q_vals = q_network->forward(states).gather(1, actions);

        torch::Tensor targets;
        {
            torch::NoGradGuard ng;
            auto next_q = std::get<0>(
                target_network->forward(next_states).max(1, /*keepdim=*/true));
            targets = rewards + gamma * next_q * (1.0f - dones);
        }

        auto loss = torch::mse_loss(q_vals, targets);
        optimizer.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(q_network->parameters(), 1.0);
        optimizer.step();
    }

    void post_step() override {
        update(); // decay is per-episode, applied in the main loop after sync_target()
    }
};

// ============================================================
// Utility: current best makespan in agent population
// ============================================================

static double best_makespan(FlowShopDQNAgent& agent) {
    const auto& pop = agent.ga.population();
    if (pop.empty()) return std::numeric_limits<double>::max();
    auto it = std::max_element(pop.begin(), pop.end(),
                               [](const auto& a, const auto& b) {
                                   return a.fitness < b.fitness;
                               });
    return -it->fitness; // convert back from −makespan to makespan
}

// ============================================================
// main_rlga — single-threaded DQN over episodes
// ============================================================

int main_rlga() {
    constexpr int    pop_size       = 200;
    constexpr double mut_rate       = 0.05;
    constexpr unsigned seed         = 42;
    constexpr size_t gen_per_ep     = 600;
    constexpr size_t n_ep           = 100;
    constexpr double gamma          = 0.9;
    constexpr double epsilon        = 0.5;
    constexpr double epsilon_decay  = 0.97;
    constexpr double epsilon_min    = 0.0;
    constexpr double lr             = 0.001;
    constexpr size_t batch_size     = 128;
    constexpr size_t memory_size    = 20000;

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";
    std::cout << "Flow shop — " << N_JOBS << " jobs × " << N_MACHINES
              << " machines | Lower bound: " << LOWER_BOUND << "\n\n";

    FlowShopRLGA ga(pop_size, mut_rate, seed);
    FlowShopDQNAgent agent(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                           epsilon_decay, epsilon_min, lr,
                           batch_size, memory_size, seed);

    std::cout << "=== Flow Shop RLGA (DQN) ===\n"
              << "Action space: " << N_ACTIONS
              << " | State dim: " << STATE_DIM << "\n\n";

    double global_best_ms = std::numeric_limits<double>::max();

    for (size_t ep = 0; ep < n_ep; ++ep) {
        agent.ga.init_population();
        agent.current_state = agent.ga.get_dqn_state();
        double total_reward = 0.0;
        size_t ep_gens      = 0;

        for (size_t gen = 0; gen < gen_per_ep; ++gen) {
            agent.step();
            ++ep_gens;
            total_reward += agent.latest_reward;

            double bms = best_makespan(agent);
            if (bms < global_best_ms) global_best_ms = bms;

            if (agent.ga.check_halt(agent.ga.population()))
                break;
        }

        agent.sync_target();
        // Decay once per episode — mirrors train.py:agent.decay_epsilon()
        agent.epsilon = std::max(agent.epsilon_min,
                                 agent.epsilon * agent.epsilon_decay);
        agent.curr_ep++;

        std::cout << "Ep " << std::setw(3) << ep
                  << " | gens=" << std::setw(4) << ep_gens
                  << " | reward=" << std::setw(8) << std::fixed
                  << std::setprecision(2) << total_reward
                  << " | best_ms=" << std::setw(7) << std::setprecision(1)
                  << best_makespan(agent)
                  << " | global_best=" << std::setw(7) << global_best_ms
                  << " | eps=" << std::setprecision(4) << agent.epsilon
                  << "\n";
    }

    std::cout << "\nFinal best makespan: " << global_best_ms
              << "  (LB=" << LOWER_BOUND << ")\n";
    return 0;
}

// ============================================================
// main_par_rlga — parallel DQN via RL_IslandModel
// ============================================================

int main_par_rlga() {
    constexpr unsigned n_threads    = 4;
    constexpr int    pop_size       = 100;
    constexpr double mut_rate       = 0.05;
    constexpr size_t gen_per_ep     = 800;
    constexpr size_t n_ep           = 100;
    constexpr double gamma          = 0.9;
    constexpr double epsilon        = 0.5;
    constexpr double epsilon_decay  = 0.995;
    constexpr double epsilon_min    = 0.0;
    constexpr double lr             = 0.001;
    constexpr size_t batch_size     = 128;
    constexpr size_t memory_size    = 20000;
    constexpr unsigned n_migrants         = 5;
    constexpr double migration_probability = 0.2;
    constexpr unsigned quorum       = 2;

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";
    std::cout << "Flow shop — " << N_JOBS << " jobs × " << N_MACHINES
              << " machines | Lower bound: " << LOWER_BOUND << "\n\n";

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; ++i)
        seeds.push_back(1234u + i * 1337u);

    std::vector<FlowShopDQNAgent> agents;
    agents.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; ++i) {
        FlowShopRLGA ga(pop_size, mut_rate, seeds[i]);
        ga.init_population();
        agents.emplace_back(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                            epsilon_decay, epsilon_min, lr,
                            batch_size, memory_size, seeds[i]);
        // Note: epsilon decay happens per-step inside post_step for par_rlga
        // (no per-episode boundary exists in the island model)
        agents.back().epsilon_decay = std::pow(epsilon_decay,
            1.0 / static_cast<double>(gen_per_ep)); // rescale to per-step
    }

    RL_IslandModel<FlowShopDQNAgent> model(n_threads, migration_probability,
                                           n_migrants, quorum,
                                           std::move(agents), seeds);

    std::cout << "=== Parallel Flow Shop RLGA island model (DQN) ===\n"
              << "Threads=" << n_threads
              << " | quorum=" << quorum
              << " | pop_per_island=" << pop_size << "\n\n";

    model.island_model_run();

    std::cout << "\n--- Results ---\n";
    auto populations = model.populations();
    for (unsigned i = 0; i < populations.size(); ++i) {
        auto& pop = populations[i];
        if (pop.empty()) {
            std::cout << "Island " << i << ": empty\n";
            continue;
        }
        auto best = std::max_element(
            pop.begin(), pop.end(),
            [](const auto& a, const auto& b) { return a.fitness < b.fitness; });
        std::cout << "Island " << i
                  << " best makespan: " << -best->fitness << "\n";
    }

    return 0;
}

// ============================================================
// Entry point
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [rlga|par_rlga]\n";
        return 1;
    }

    std::string mode = argv[1];
    if      (mode == "rlga")     return main_rlga();
    else if (mode == "par_rlga") return main_par_rlga();
    else {
        std::cerr << "Invalid argument. Use 'rlga' or 'par_rlga'.\n";
        return 1;
    }
}
