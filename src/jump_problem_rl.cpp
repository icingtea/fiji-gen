// jump_problem_rl.cpp
// DQN agent guiding a Sequential/Parallel GA for the Jump Problem.
// Genome = n-bit string via std::vector<uint8_t>.
// Run with: ./jump_problem_rl [rlga|par_rlga] [params.txt]

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
#include <span>
#include <string>
#include <vector>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ============================================================
// Utility: Load Parameters from File
// ============================================================
json load_params(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open parameter file " << filename
                  << ". Using defaults.\n";
        return json::object();
    }
    json j;
    file >> j;
    return j;
}

// ============================================================
// Action space
// ============================================================

enum class SelectionMethod : int { ELITISM = 0, ROULETTE = 1, RANK = 2 };

struct Action {
    SelectionMethod method;
    double selection_rate;
    double mutation_rate;
};

std::vector<Action> ACTION_SPACE;
int N_ACTIONS = 0;

void init_action_space(const std::vector<double>& custom_mut_rates,
                       const std::vector<double>& custom_sel_rates) {
    ACTION_SPACE.clear();
    const std::vector<SelectionMethod> methods = {SelectionMethod::ELITISM,
                                                  SelectionMethod::ROULETTE,
                                                  SelectionMethod::RANK};
    std::vector<double> sel_rates = custom_sel_rates.empty()
                                        ? std::vector<double>{0.3, 0.6, 0.9}
                                        : custom_sel_rates;
    std::vector<double> mut_rates =
        custom_mut_rates.empty() ? std::vector<double>{0.01, 0.05, 0.2, 0.4}
                                 : custom_mut_rates;

    for (auto m : methods)
        for (auto sr : sel_rates)
            for (auto mr : mut_rates)
                ACTION_SPACE.push_back({m, sr, mr});

    N_ACTIONS = static_cast<int>(ACTION_SPACE.size());
}

static const int STATE_DIM = 2; // [avg_fitness, entropy]

// ============================================================
// DQNNet
// ============================================================

struct DQNNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr},
        fc5{nullptr}, fc6{nullptr};

    DQNNetImpl(int in_dim, int out_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 128));
        fc3 = register_module("fc3", torch::nn::Linear(128, 128));
        fc4 = register_module("fc4", torch::nn::Linear(128, 128));
        fc5 = register_module("fc5", torch::nn::Linear(128, 64));
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
// JumpRLGA — RLGA<std::vector<uint8_t>, int>
// ============================================================

class JumpRLGA : public RLGA<std::vector<uint8_t>, int> {
  public:
    std::mt19937 rng;

    // Current GA parameters, set by apply_action()
    SelectionMethod current_method = SelectionMethod::ELITISM;
    double current_selection_rate = 0.5;
    double current_mutation_rate = 0.01;

    unsigned GENOME_LENGTH;
    unsigned JUMP_SIZE;
    double MAX_FITNESS;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
    std::uniform_int_distribution<uint8_t> bit_dist{0, 1};

    explicit JumpRLGA(int pop_size, double mut_rate, unsigned seed,
                      unsigned genome_length, unsigned jump_size)
        : RLGA(pop_size, mut_rate, seed), rng(seed),
          GENOME_LENGTH(genome_length), JUMP_SIZE(jump_size),
          MAX_FITNESS(static_cast<double>(genome_length)),
          current_mutation_rate(mut_rate) {}

    // --------------------------------------------------------
    // GA pure-virtual implementations
    // --------------------------------------------------------

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

        size_t n = static_cast<size_t>(pop.size() * current_selection_rate);
        if (n < 2)
            n = 2;
        n = std::min(n, pop.size());

        std::vector<size_t> pool;
        pool.reserve(n);

        if (current_method == SelectionMethod::ELITISM) {
            std::vector<size_t> order(pop.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + n, order.end(),
                              [&](size_t a, size_t b) {
                                  return pop[a].fitness > pop[b].fitness;
                              });
            pool.assign(order.begin(), order.begin() + n);

        } else if (current_method == SelectionMethod::ROULETTE) {
            std::vector<double> weights(pop.size());
            for (size_t k = 0; k < pop.size(); ++k)
                weights[k] = pop[k].fitness + 1e-8;

            double total = 0.0;
            for (double w : weights)
                total += w;

            std::uniform_real_distribution<double> spin(0.0, total);
            for (size_t k = 0; k < n; ++k) {
                double pick = spin(rng), cum = 0.0;
                size_t chosen = pop.size() - 1;
                for (size_t j = 0; j < pop.size(); ++j) {
                    cum += weights[j];
                    if (cum >= pick) {
                        chosen = j;
                        break;
                    }
                }
                pool.push_back(chosen);
            }

        } else { // RANK
            std::vector<size_t> sorted(pop.size());
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
                return pop[a].fitness > pop[b].fitness; // descending
            });

            std::vector<double> weights(pop.size());
            for (size_t r = 0; r < sorted.size(); ++r)
                weights[sorted[r]] = 1.0 / (double)(r + 1);

            double total = 0.0;
            for (double w : weights)
                total += w;

            std::uniform_real_distribution<double> spin(0.0, total);
            for (size_t k = 0; k < n; ++k) {
                double pick = spin(rng), cum = 0.0;
                size_t chosen = pop.size() - 1;
                for (size_t j = 0; j < pop.size(); ++j) {
                    cum += weights[j];
                    if (cum >= pick) {
                        chosen = j;
                        break;
                    }
                }
                pool.push_back(chosen);
            }
        }

        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        for (size_t i = 0; i < pop.size() / 2; ++i)
            result.emplace_back(Pairing<std::vector<uint8_t>>{pool[pick(rng)],
                                                              pool[pick(rng)]});
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
            if (prob_dist(rng) < current_mutation_rate) {
                ind.genome[i] ^= 1;
            }
        }
    }

    void drop_individuals(
        std::vector<Individual<std::vector<uint8_t>>>& pop) override {
        std::sort(pop.begin(), pop.end(), [](const auto& a, const auto& b) {
            return a.fitness > b.fitness;
        });
        if (pop.size() > init_pop_size)
            pop.resize(init_pop_size);
    }

    // Halt when optimum is reached.
    bool check_halt(
        std::vector<Individual<std::vector<uint8_t>>>& pop) override {
        if (pop.empty())
            return true;

        auto best_it = std::max_element(
            pop.begin(), pop.end(),
            [](const auto& a, const auto& b) { return a.fitness < b.fitness; });

        return best_it->fitness >= MAX_FITNESS;
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
        std::span<const Individual<std::vector<uint8_t>>> old_pop,
        std::vector<Individual<std::vector<uint8_t>>>& new_pop,
        std::vector<Individual<std::vector<uint8_t>>>& /*children*/,
        std::vector<Pairing<std::vector<uint8_t>>>& /*pairings*/) override {

        double old_best = std::numeric_limits<double>::lowest();
        for (const auto& ind : old_pop)
            old_best = std::max(old_best, ind.fitness);

        double new_best = std::numeric_limits<double>::lowest();
        for (const auto& ind : new_pop)
            new_best = std::max(new_best, ind.fitness);

        return new_best - old_best;
    }

    // DQN state: [normalised_avg_fitness, entropy]
    torch::Tensor get_dqn_state() const {
        if (population_.empty())
            return torch::zeros({STATE_DIM});

        double total_fit = 0.0;
        for (const auto& ind : population_)
            total_fit += ind.fitness;

        double avg_fit = total_fit / static_cast<double>(population_.size());
        double norm_avg = std::clamp(avg_fit / MAX_FITNESS, 0.0, 1.0);

        // Entropy
        double sum_sq = total_fit + 1e-8;
        double entropy = 0.0;
        for (const auto& ind : population_) {
            double p = ind.fitness / sum_sq;
            if (p > 0.0)
                entropy -= p * std::log(p + 1e-8);
        }

        return torch::tensor({(float)norm_avg, (float)entropy});
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
    float done;
};

class ReplayBuffer {
  public:
    explicit ReplayBuffer(size_t capacity) : capacity_(capacity) {}

    void push(Transition t) {
        if (buffer_.size() >= capacity_)
            buffer_.pop_front();
        buffer_.push_back(std::move(t));
    }

    bool ready(size_t batch_size) const { return buffer_.size() >= batch_size; }

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
// JumpDQNAgent
// ============================================================

class JumpDQNAgent : public Agent<int, JumpRLGA> {
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

    torch::Tensor current_state;
    int last_action = 0;

    explicit JumpDQNAgent(JumpRLGA&& ga_in, size_t gen_per_ep, size_t n_ep,
                          double gamma, double epsilon, double epsilon_decay,
                          double epsilon_min, double lr, size_t batch_size,
                          size_t memory_size, unsigned rng_seed)
        : Agent<int, JumpRLGA>(std::move(ga_in), gen_per_ep, n_ep, gamma,
                               epsilon, epsilon_decay, epsilon_min, lr),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          q_network(STATE_DIM, N_ACTIONS), target_network(STATE_DIM, N_ACTIONS),
          optimizer(q_network->parameters(), torch::optim::AdamOptions(lr)),
          memory(memory_size), batch_size(batch_size), rng(rng_seed) {
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
            last_action = static_cast<int>(q_vals.argmax(1).item<int64_t>());
        }
        return last_action;
    }

    void update() override {
        torch::Tensor next_state = ga.get_dqn_state();
        bool done = ga.check_halt(ga.population());

        memory.push({current_state, last_action,
                     static_cast<float>(latest_reward), next_state,
                     done ? 1.0f : 0.0f});

        if (!memory.ready(batch_size))
            return;

        auto batch = memory.sample(batch_size, rng);

        std::vector<torch::Tensor> sv, nsv;
        std::vector<int64_t> av;
        std::vector<float> rv, dv;

        for (auto& tr : batch) {
            sv.push_back(tr.state);
            nsv.push_back(tr.next_state);
            av.push_back(tr.action);
            rv.push_back(tr.reward);
            dv.push_back(tr.done);
        }

        auto states = torch::stack(sv).to(device);
        auto next_states = torch::stack(nsv).to(device);
        auto actions = torch::tensor(av, torch::kInt64).unsqueeze(1).to(device);
        auto rewards = torch::tensor(rv).unsqueeze(1).to(device);
        auto dones = torch::tensor(dv).unsqueeze(1).to(device);

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

    void post_step() override { update(); }
};

static double best_fitness_of(JumpDQNAgent& agent) {
    const auto& pop = agent.ga.population();
    if (pop.empty())
        return 0.0;
    auto it = std::max_element(
        pop.begin(), pop.end(),
        [](const auto& a, const auto& b) { return a.fitness < b.fitness; });
    return it->fitness;
}

// ============================================================
// main_rlga — single-threaded DQN over episodes
// ============================================================

template <typename T>
T get_nested_param(const json& params, const std::string& section,
                   const std::string& key, T def) {
    if (params.contains(section) && params[section].contains(key)) {
        return params[section][key].get<T>();
    }
    return def;
}

json main_rlga(const json& params) {
    int pop_size =
        get_nested_param(params, "genetic_algorithm", "pop_size", 10000);
    double mut_rate =
        get_nested_param(params, "genetic_algorithm", "mut_rate", 0.01);
    unsigned seed = std::random_device{}();
    size_t gen_per_ep =
        get_nested_param(params, "reinforcement_learning", "gen_per_ep", 100u);
    size_t n_ep =
        get_nested_param(params, "reinforcement_learning", "n_ep", 200u);
    double gamma =
        get_nested_param(params, "reinforcement_learning", "gamma", 0.9);
    double epsilon =
        get_nested_param(params, "reinforcement_learning", "epsilon", 0.5);
    double epsilon_decay = get_nested_param(params, "reinforcement_learning",
                                            "epsilon_decay", 0.97);
    double epsilon_min =
        get_nested_param(params, "reinforcement_learning", "epsilon_min", 0.05);
    double lr = get_nested_param(params, "reinforcement_learning", "lr", 0.001);
    size_t batch_size =
        get_nested_param(params, "reinforcement_learning", "batch_size", 128u);
    size_t memory_size = get_nested_param(params, "reinforcement_learning",
                                          "memory_size", 10000u);
    unsigned genome_length = static_cast<unsigned>(
        get_nested_param(params, "problem", "GENOME_LENGTH", 100u));
    unsigned jump_size = static_cast<unsigned>(
        get_nested_param(params, "problem", "JUMP_SIZE", 4u));

    std::vector<double> custom_mut_rates;
    std::vector<double> custom_sel_rates;
    if (params.contains("reinforcement_learning")) {
        auto& rl_params = params["reinforcement_learning"];
        if (rl_params.contains("mut_rates")) {
            custom_mut_rates =
                rl_params["mut_rates"].get<std::vector<double>>();
        }
        if (rl_params.contains("sel_rates")) {
            custom_sel_rates =
                rl_params["sel_rates"].get<std::vector<double>>();
        }
    }
    init_action_space(custom_mut_rates, custom_sel_rates);

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";
    std::cout << "Jump Problem (" << genome_length
              << " bits | Jump Size: " << jump_size << ")\n\n";

    JumpRLGA ga(pop_size, mut_rate, seed, genome_length, jump_size);
    JumpDQNAgent agent(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                       epsilon_decay, epsilon_min, lr, batch_size, memory_size,
                       seed);

    std::cout << "=== Jump RLGA (DQN) ===\n"
              << "Action space: " << N_ACTIONS << " | State dim: " << STATE_DIM
              << "\n\n";

    size_t total_gens = 0;
    agent.ga.init_population();

    auto start_time = std::chrono::high_resolution_clock::now();

    bool overall_solved = false;
    size_t solved_ep = 0;

    for (size_t ep = 0; ep < n_ep; ++ep) {
        agent.current_state = agent.ga.get_dqn_state();
        double total_reward = 0.0;
        size_t ep_gens = 0;
        bool solved = false;

        for (size_t gen = 0; gen < gen_per_ep; ++gen) {
            agent.step();
            ++ep_gens;
            total_reward += agent.latest_reward;

            if (agent.ga.check_halt(agent.ga.population())) {
                solved = true;
                break;
            }
        }

        total_gens += ep_gens;
        agent.sync_target();
        agent.decay_epsilon();
        agent.curr_ep++;

        std::cout << "Ep " << std::setw(3) << ep << " | gens=" << std::setw(4)
                  << ep_gens << " | reward=" << std::setw(8) << std::fixed
                  << std::setprecision(2) << total_reward
                  << " | best=" << std::setw(7) << std::setprecision(1)
                  << best_fitness_of(agent) << " | eps=" << std::setprecision(4)
                  << agent.epsilon << "\n";

        if (solved) {
            std::cout << "\nSolved at episode " << ep << " after " << ep_gens
                      << " gens this episode"
                      << " (" << total_gens << " total cumulative gens).\n";
            overall_solved = true;
            solved_ep = ep;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    json result;
    result["mode"] = "rlga";
    result["time_ms"] = duration;
    result["genome_length"] = genome_length;
    result["jump_size"] = jump_size;
    result["pop_size"] = pop_size;
    result["total_generations"] = total_gens;
    result["episodes_run"] = overall_solved ? solved_ep + 1 : n_ep;
    result["best_fitness"] = best_fitness_of(agent);
    result["solved"] = overall_solved;

    return result;
}

json main_par_rlga(const json& params) {
    unsigned n_threads = get_nested_param(params, "parallel", "n_threads", 4u);
    int pop_size =
        get_nested_param(params, "genetic_algorithm", "pop_size", 10000);
    double mut_rate =
        get_nested_param(params, "genetic_algorithm", "mut_rate", 0.01);
    size_t gen_per_ep =
        get_nested_param(params, "reinforcement_learning", "gen_per_ep", 100u);
    size_t n_ep =
        get_nested_param(params, "reinforcement_learning", "n_ep", 200u);
    double gamma =
        get_nested_param(params, "reinforcement_learning", "gamma", 0.9);
    double epsilon =
        get_nested_param(params, "reinforcement_learning", "epsilon", 0.5);
    double epsilon_decay = get_nested_param(params, "reinforcement_learning",
                                            "epsilon_decay", 0.97);
    double epsilon_min =
        get_nested_param(params, "reinforcement_learning", "epsilon_min", 0.05);
    double lr = get_nested_param(params, "reinforcement_learning", "lr", 0.001);
    size_t batch_size =
        get_nested_param(params, "reinforcement_learning", "batch_size", 128u);
    size_t memory_size = get_nested_param(params, "reinforcement_learning",
                                          "memory_size", 10000u);
    unsigned n_migrants =
        get_nested_param(params, "parallel", "n_migrants", 500u);
    double migration_probability =
        get_nested_param(params, "parallel", "migration_probability", 0.5);
    unsigned quorum = get_nested_param(params, "parallel", "quorum", 2u);
    unsigned genome_length = static_cast<unsigned>(
        get_nested_param(params, "problem", "GENOME_LENGTH", 100u));
    unsigned jump_size = static_cast<unsigned>(
        get_nested_param(params, "problem", "JUMP_SIZE", 4u));

    std::vector<double> custom_mut_rates;
    std::vector<double> custom_sel_rates;
    if (params.contains("reinforcement_learning")) {
        auto& rl_params = params["reinforcement_learning"];
        if (rl_params.contains("mut_rates")) {
            custom_mut_rates =
                rl_params["mut_rates"].get<std::vector<double>>();
        }
        if (rl_params.contains("sel_rates")) {
            custom_sel_rates =
                rl_params["sel_rates"].get<std::vector<double>>();
        }
    }
    init_action_space(custom_mut_rates, custom_sel_rates);

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";
    std::cout << "Jump Problem (" << genome_length
              << " bits | Jump Size: " << jump_size << ")\n\n";

    std::vector<unsigned> seeds;
    std::random_device rd;
    for (unsigned i = 0; i < n_threads; ++i)
        seeds.push_back(rd());

    std::vector<JumpDQNAgent> agents;
    agents.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; ++i) {
        JumpRLGA ga(pop_size, mut_rate, seeds[i], genome_length, jump_size);
        ga.init_population();
        agents.emplace_back(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                            epsilon_decay, epsilon_min, lr, batch_size,
                            memory_size, seeds[i]);
    }

    RL_IslandModel<JumpDQNAgent> model(n_threads, migration_probability,
                                       n_migrants, quorum, std::move(agents),
                                       seeds);

    std::cout << "=== Parallel Jump RLGA island model (DQN) ===\n"
              << "Threads=" << n_threads << " | quorum=" << quorum
              << " | pop_per_island=" << pop_size << "\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    model.island_model_run();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    std::cout << "\n--- Results ---\n";
    auto populations = model.populations();
    auto gens = model.generations();
    unsigned min_gens_solved = std::numeric_limits<unsigned>::max();
    unsigned solved_count = 0;
    double overall_best = 0.0;

    json islands_data = json::array();

    for (unsigned i = 0; i < populations.size(); ++i) {
        auto& pop = populations[i];
        if (pop.empty()) {
            std::cout << "Island " << i << ": empty\n";
            continue;
        }
        auto best = std::max_element(
            pop.begin(), pop.end(),
            [](const auto& a, const auto& b) { return a.fitness < b.fitness; });

        // Check if this island reached the optimal solution
        bool solved = model.island_solved(i);
        std::cout << "Island " << i << " best fitness: " << best->fitness
                  << " | generations: " << gens[i]
                  << (solved ? " [SOLVED]" : " [did not solve]") << "\n";

        if (best->fitness > overall_best)
            overall_best = best->fitness;

        if (solved) {
            ++solved_count;
            min_gens_solved = std::min(min_gens_solved, gens[i]);
        }

        json island_info;
        island_info["island_id"] = i;
        island_info["best_fitness"] = best->fitness;
        island_info["generations"] = gens[i];
        island_info["solved"] = solved;
        islands_data.push_back(island_info);
    }

    std::cout << "\n--- Summary ---\n";
    if (solved_count > 0) {
        std::cout << "Islands solved: " << solved_count << " / "
                  << populations.size() << "\n"
                  << "Minimum generations to solution: " << min_gens_solved
                  << "\n";
    } else {
        std::cout << "No island reached the optimal solution.\n";
    }

    json result;
    result["mode"] = "par_rlga";
    result["time_ms"] = duration;
    result["genome_length"] = genome_length;
    result["jump_size"] = jump_size;
    result["island_pop"] = pop_size;
    result["n_migrants"] = n_migrants;
    result["n_threads"] = n_threads;
    result["overall_best_fitness"] = overall_best;
    result["solved_count"] = solved_count;
    if (solved_count > 0) {
        result["min_gens_solved"] = min_gens_solved;
    }
    result["islands"] = islands_data;

    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " [rlga|par_rlga] [params.json] [output.jsonl]\n";
        return 1;
    }

    std::string mode = argv[1];
    json params = load_params(argv[2]);
    std::string output_file = argv[3];

    json result;
    if (mode == "rlga")
        result = main_rlga(params);
    else if (mode == "par_rlga")
        result = main_par_rlga(params);
    else {
        std::cerr << "Invalid argument. Use 'rlga' or 'par_rlga'.\n";
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
