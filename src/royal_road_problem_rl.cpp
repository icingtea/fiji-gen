// royal_road_problem_rl.cpp
// DQN agent guiding the Royal Road GA (mirrors temp/src/{qlearning,env,train}.py).
// Run with: ./royal_road_problem_rl [rlga|par_rlga]

#include "par_rl_ga.hpp"
#include "rl_ga.hpp"
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

// ============================================================
// Action space: 3 methods × 3 selection rates × 3 mutation rates = 27 actions
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
static const int STATE_DIM = 2; // [avg_fitness, entropy]
static constexpr double MAX_FITNESS = 32.0;

// ============================================================
// DQNNet — mirrors DQNModel in qlearning.py
// Linear(2→64)→ReLU → (×4 64→64 ReLU) → Linear(64→27)
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
// RoyalRoadRLGA — RLGA<unsigned, int>
// Action type `int` = index into ACTION_SPACE.
// ============================================================

class RoyalRoadRLGA : public RLGA<unsigned, int> {
  public:
    std::mt19937 rng;

    static constexpr unsigned GENOME_LENGTH = 32;
    static constexpr unsigned BLOCK_SIZE = 8;

    // Current parameters, updated by apply_action()
    SelectionMethod current_method = SelectionMethod::ELITISM;
    double current_selection_rate = 0.5;
    double current_mutation_rate = 0.01;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
    std::uniform_int_distribution<unsigned> bit_dist{0u, UINT32_MAX};

    explicit RoyalRoadRLGA(int pop_size, double mut_rate, unsigned seed)
        : RLGA(pop_size, mut_rate, seed), rng(seed),
          current_mutation_rate(mut_rate) {}

    // --------------------------------------------------------
    // GA pure-virtual implementations
    // --------------------------------------------------------

    void init_population() override {
        population_.clear();
        for (unsigned i = 0; i < init_pop_size; ++i) {
            Individual<unsigned> ind;
            ind.genome = bit_dist(rng);
            ind.fitness = compute_fitness(ind);
            population_.push_back(ind);
        }
    }

    double compute_fitness(Individual<unsigned>& individual) override {
        double fitness = 0.0;
        for (unsigned i = 0; i + BLOCK_SIZE <= GENOME_LENGTH; i += BLOCK_SIZE) {
            bool all_one = true;
            for (unsigned j = 0; j < BLOCK_SIZE; ++j) {
                if (!((individual.genome >> (i + j)) & 1u)) {
                    all_one = false;
                    break;
                }
            }
            if (all_one)
                fitness += BLOCK_SIZE;
        }
        return fitness;
    }

    std::vector<Pairing<unsigned>> select_parents(
        std::vector<Individual<unsigned>>& population) override {

        std::vector<Pairing<unsigned>> result;
        if (population.empty())
            return result;

        size_t n = static_cast<size_t>(population.size() * current_selection_rate);
        if (n < 2)
            n = 2;
        n = std::min(n, population.size());

        std::vector<size_t> pool;
        pool.reserve(n);

        if (current_method == SelectionMethod::ELITISM) {
            std::vector<size_t> order(population.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + n, order.end(),
                              [&](size_t a, size_t b) {
                                  return population[a].fitness >
                                         population[b].fitness;
                              });
            pool.assign(order.begin(), order.begin() + n);

        } else if (current_method == SelectionMethod::ROULETTE) {
            double total = 0.0;
            for (auto& ind : population)
                total += ind.fitness;
            if (total < 1e-9)
                total = 1.0;

            std::uniform_real_distribution<double> spin(0.0, total);
            for (size_t k = 0; k < n; ++k) {
                double pick = spin(rng);
                double cum = 0.0;
                size_t chosen = population.size() - 1;
                for (size_t j = 0; j < population.size(); ++j) {
                    cum += population[j].fitness;
                    if (cum >= pick) {
                        chosen = j;
                        break;
                    }
                }
                pool.push_back(chosen);
            }

        } else { // RANK
            std::vector<size_t> sorted(population.size());
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
                return population[a].fitness < population[b].fitness;
            });
            double rank_total = (double)population.size() *
                                (population.size() + 1) / 2.0;
            std::uniform_real_distribution<double> spin(0.0, rank_total);
            for (size_t k = 0; k < n; ++k) {
                double pick = spin(rng);
                double cum = 0.0;
                size_t chosen = sorted.back();
                for (size_t j = 0; j < sorted.size(); ++j) {
                    cum += (double)(j + 1);
                    if (cum >= pick) {
                        chosen = sorted[j];
                        break;
                    }
                }
                pool.push_back(chosen);
            }
        }

        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        for (size_t i = 0; i < population.size() / 2; ++i) {
            size_t p1 = pool[pick(rng)];
            size_t p2 = pool[pick(rng)];
            result.emplace_back(Pairing<unsigned>{p1, p2});
        }
        return result;
    }

    Individual<unsigned> crossover(Pairing<unsigned>& pair) override {
        Individual<unsigned> child;
        std::uniform_int_distribution<unsigned> cut_dist(1, GENOME_LENGTH - 1);
        unsigned cut = cut_dist(rng);
        unsigned left_mask = (1u << cut) - 1;
        unsigned right_mask = ~left_mask;
        child.genome =
            (population_[pair.parent_1_index].genome & left_mask) |
            (population_[pair.parent_2_index].genome & right_mask);
        return child;
    }

    void mutate(Individual<unsigned>& individual) override {
        for (unsigned i = 0; i < GENOME_LENGTH; ++i) {
            if (prob_dist(rng) < current_mutation_rate)
                individual.genome ^= (1u << i);
        }
    }

    void drop_individuals(
        std::vector<Individual<unsigned>>& population) override {
        std::sort(population.begin(), population.end(),
                  [](auto& a, auto& b) { return a.fitness > b.fitness; });
        if (population.size() > init_pop_size)
            population.resize(init_pop_size);
    }

    bool check_halt(
        std::vector<Individual<unsigned>>& population) override {
        if (population.empty())
            return true;
        auto it = std::max_element(
            population.begin(), population.end(),
            [](auto& a, auto& b) { return a.fitness < b.fitness; });
        return it->fitness >= static_cast<double>(GENOME_LENGTH);
    }

    // --------------------------------------------------------
    // RLGA pure-virtual implementations
    // --------------------------------------------------------

    void apply_action(int action_idx) override {
        assert(action_idx >= 0 && action_idx < N_ACTIONS);
        const Action& a = ACTION_SPACE[action_idx];
        current_method = a.method;
        current_selection_rate = a.selection_rate;
        current_mutation_rate = a.mutation_rate;
    }

    double calculate_reward(
        std::span<const Individual<unsigned>> old_population,
        std::vector<Individual<unsigned>>& new_population,
        std::vector<Individual<unsigned>>& /*children*/,
        std::vector<Pairing<unsigned>>& /*pairings*/) override {

        double old_best = std::numeric_limits<double>::lowest();
        for (const auto& ind : old_population)
            old_best = std::max(old_best, ind.fitness);

        double new_best = std::numeric_limits<double>::lowest();
        for (const auto& ind : new_population)
            new_best = std::max(new_best, ind.fitness);

        return new_best - old_best;
    }

    // ----------------------------------------------------------
    // DQN state: [avg_fitness, entropy] — mirrors env.py:_get_state()
    // ----------------------------------------------------------
    torch::Tensor get_dqn_state() const {
        if (population_.empty())
            return torch::zeros({STATE_DIM});

        double total = 0.0;
        for (const auto& ind : population_)
            total += ind.fitness;
        double avg = total / static_cast<double>(population_.size());

        // Fitness-distribution entropy (same formula as env.py)
        double sum_sq = total + 1e-8;
        double entropy = 0.0;
        for (const auto& ind : population_) {
            double p = ind.fitness / sum_sq;
            if (p > 0.0)
                entropy -= p * std::log(p + 1e-8);
        }

        return torch::tensor({(float)avg, (float)entropy});
    }
};

// ============================================================
// Experience replay buffer
// ============================================================

struct Transition {
    torch::Tensor state;
    int action;
    float reward;
    torch::Tensor next_state;
    float done; // 1.0 if terminal, else 0.0
};

class ReplayBuffer {
  public:
    explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {}

    void push(Transition t) {
        if (buffer_.size() >= capacity_)
            buffer_.pop_front();
        buffer_.push_back(std::move(t));
    }

    bool ready(size_t batch_size) const {
        return buffer_.size() >= batch_size;
    }

    // Sample `n` random transitions (with replacement)
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
// RoyalRoadDQNAgent — DQN agent (mirrors DQNAgent in qlearning.py)
// ============================================================

class RoyalRoadDQNAgent : public Agent<int, RoyalRoadRLGA> {
  public:
    torch::Device device;
    DQNNet q_network;
    DQNNet target_network;
    torch::optim::Adam optimizer;

    ReplayBuffer memory;
    size_t batch_size;

    std::mt19937 rng;
    std::uniform_real_distribution<double> eps_dist{0.0, 1.0};
    std::uniform_int_distribution<int> random_action{0, N_ACTIONS - 1};

    // State snapshots for the current step
    torch::Tensor current_state;
    int last_action = 0;

    explicit RoyalRoadDQNAgent(RoyalRoadRLGA&& ga_in,
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
        : Agent<int, RoyalRoadRLGA>(std::move(ga_in), gen_per_ep, n_ep,
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

    // Copy the online network weights into the target network
    void sync_target() {
        torch::NoGradGuard ng;
        auto src_params = q_network->parameters();
        auto dst_params = target_network->parameters();
        for (size_t i = 0; i < src_params.size(); ++i)
            dst_params[i].copy_(src_params[i]);
    }

    // ε-greedy action selection using the online network
    int select_action() override {
        current_state = ga.get_dqn_state();

        if (eps_dist(rng) < epsilon) {
            last_action = random_action(rng);
        } else {
            q_network->eval();
            torch::NoGradGuard ng;
            auto s = current_state.unsqueeze(0).to(device);
            auto q_vals = q_network->forward(s); // [1, N_ACTIONS]
            last_action = static_cast<int>(q_vals.argmax(1).item<int64_t>());
        }
        return last_action;
    }

    // Store transition + minibatch SGD update
    void update() override {
        torch::Tensor next_state = ga.get_dqn_state();
        bool done = ga.check_halt(ga.population());

        memory.push({current_state,
                     last_action,
                     static_cast<float>(latest_reward),
                     next_state,
                     done ? 1.0f : 0.0f});

        if (!memory.ready(batch_size))
            return;

        auto batch = memory.sample(batch_size, rng);

        // Build tensors from batch
        std::vector<torch::Tensor> s_vec, ns_vec;
        std::vector<int64_t> a_vec;
        std::vector<float> r_vec, d_vec;

        for (auto& tr : batch) {
            s_vec.push_back(tr.state);
            ns_vec.push_back(tr.next_state);
            a_vec.push_back(tr.action);
            r_vec.push_back(tr.reward);
            d_vec.push_back(tr.done);
        }

        auto states      = torch::stack(s_vec).to(device);          // [B, 2]
        auto next_states = torch::stack(ns_vec).to(device);         // [B, 2]
        auto actions     = torch::tensor(a_vec, torch::kInt64)
                               .unsqueeze(1).to(device);            // [B, 1]
        auto rewards     = torch::tensor(r_vec).unsqueeze(1).to(device); // [B, 1]
        auto dones       = torch::tensor(d_vec).unsqueeze(1).to(device); // [B, 1]

        q_network->train();

        // Current Q-values for the taken actions
        auto q_values = q_network->forward(states).gather(1, actions); // [B,1]

        // Target Q-values via frozen target network
        torch::Tensor targets;
        {
            torch::NoGradGuard ng;
            auto next_q = std::get<0>(
                target_network->forward(next_states).max(1, /*keepdim=*/true)); // [B,1]
            targets = rewards + gamma * next_q * (1.0f - dones);
        }

        auto loss = torch::mse_loss(q_values, targets);

        optimizer.zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(q_network->parameters(), 1.0);
        optimizer.step();
    }

    // Called after each GA step
    void post_step() override {
        update(); // decay is per-episode, applied in the main loop after sync_target()
    }
};

// ============================================================
// main_rlga — single-threaded DQN over episodes
// ============================================================

int main_rlga() {
    constexpr int pop_size       = 100;
    constexpr double mut_rate    = 0.01;
    constexpr unsigned seed      = 42;
    constexpr size_t gen_per_ep  = 300;
    constexpr size_t n_ep        = 200;
    constexpr double gamma       = 0.9;
    constexpr double epsilon     = 0.5;
    constexpr double epsilon_decay = 0.97;
    constexpr double epsilon_min = 0.0;
    constexpr double lr          = 0.001;
    constexpr size_t batch_size  = 64;
    constexpr size_t memory_size = 10000;

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";

    RoyalRoadRLGA ga(pop_size, mut_rate, seed);
    RoyalRoadDQNAgent agent(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                            epsilon_decay, epsilon_min, lr,
                            batch_size, memory_size, seed);

    std::cout << "=== RLGA (DQN) ===\n"
              << "Action space: " << N_ACTIONS << " | "
              << "State dim: " << STATE_DIM << "\n\n";

    bool solved = false;
    size_t total_gens = 0;
    for (size_t ep = 0; ep < n_ep && !solved; ++ep) {
        agent.ga.init_population();
        agent.current_state = agent.ga.get_dqn_state();
        double total_reward = 0.0;
        double best_fitness = 0.0;
        size_t ep_gens = 0;

        for (size_t gen = 0; gen < gen_per_ep; ++gen) {
            agent.step();
            ++ep_gens;
            total_reward += agent.latest_reward;

            for (auto& ind : agent.ga.population())
                best_fitness = std::max(best_fitness, ind.fitness);

            if (agent.ga.check_halt(agent.ga.population())) {
                solved = true;
                break;
            }
        }
        total_gens += ep_gens;

        agent.sync_target();
        // Decay once per episode — mirrors train.py:agent.decay_epsilon()
        agent.epsilon = std::max(agent.epsilon_min,
                                 agent.epsilon * agent.epsilon_decay);
        agent.curr_ep++;

        std::cout << "Ep " << std::setw(3) << ep
                  << " | gens=" << std::setw(4) << ep_gens
                  << " | reward=" << std::setw(8) << std::fixed
                  << std::setprecision(2) << total_reward
                  << " | best=" << best_fitness
                  << " | eps=" << std::setprecision(4) << agent.epsilon
                  << "\n";

        if (solved)
            std::cout << "\nSolved at episode " << ep
                      << " after " << ep_gens << " gens this episode"
                      << " (" << total_gens << " total gens)\n";
    }

    return 0;
}

// ============================================================
// main_par_rlga — parallel DQN via RL_IslandModel
// ============================================================

int main_par_rlga() {
    constexpr unsigned n_threads = 4;
    constexpr int pop_size       = 50;
    constexpr double mut_rate    = 0.01;
    constexpr size_t gen_per_ep  = 500;
    constexpr size_t n_ep        = 200;
    constexpr double gamma       = 0.9;
    constexpr double epsilon     = 0.5;
    constexpr double epsilon_decay = 0.995;
    constexpr double epsilon_min = 0.0;
    constexpr double lr          = 0.001;
    constexpr size_t batch_size  = 64;
    constexpr size_t memory_size = 10000;
    constexpr unsigned n_migrants = 3;
    constexpr double migration_probability = 0.2;
    constexpr unsigned quorum    = 2;

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; ++i)
        seeds.push_back(1234u + i * 1337u);

    std::vector<RoyalRoadDQNAgent> agents;
    agents.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; ++i) {
        RoyalRoadRLGA ga(pop_size, mut_rate, seeds[i]);
        ga.init_population();
        agents.emplace_back(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                            epsilon_decay, epsilon_min, lr,
                            batch_size, memory_size, seeds[i]);
        // Note: epsilon decay happens per-step inside post_step for par_rlga
        // (no per-episode boundary exists in the island model)
        agents.back().epsilon_decay = std::pow(epsilon_decay,
            1.0 / static_cast<double>(gen_per_ep)); // rescale to per-step
    }

    RL_IslandModel<RoyalRoadDQNAgent> model(n_threads, migration_probability,
                                            n_migrants, quorum,
                                            std::move(agents), seeds);

    std::cout << "=== Parallel RLGA island model (DQN) ===\n"
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
            [](auto& a, auto& b) { return a.fitness < b.fitness; });
        std::cout << "Island " << i << " best fitness: " << best->fitness << "\n";
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
    if (mode == "rlga") {
        return main_rlga();
    } else if (mode == "par_rlga") {
        return main_par_rlga();
    } else {
        std::cerr << "Invalid argument. Use 'rlga' or 'par_rlga'.\n";
        return 1;
    }
}
