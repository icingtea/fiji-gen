// deceptive_royal_road_problem_rl.cpp
// DQN agent guiding a Sequential/Parallel GA for the Deceptive Royal Road
// Problem. Genome = 128-bit string via std::vector<uint8_t>. Block Fitness: If
// fully ones, +16. Else, +(8 - number_of_ones). Run with:
// ./deceptive_royal_road_problem_rl [rlga|par_rlga]

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

#include <fstream>
#include <map>
#include <sstream>
#include <string>

// ============================================================
// Utility: Load Parameters from File
// ============================================================
std::map<std::string, std::string> load_params(const std::string& filename) {
    std::map<std::string, std::string> params;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open parameter file " << filename
                  << ". Using defaults.\n";
        return params;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
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
// Global Problem Parameters
// ============================================================
// MAX_FITNESS calculated dynamically at runtime!

// ============================================================
// Action space — identical to existing configurations: 27
// ============================================================

enum class SelectionMethod : int { ELITISM = 0, ROULETTE = 1, RANK = 2 };

struct Action {
    SelectionMethod method;
    double selection_rate;
    double mutation_rate;
};

std::vector<Action> ACTION_SPACE;
int N_ACTIONS = 0;

void init_action_space(const std::vector<double>& custom_mut_rates) {
    ACTION_SPACE.clear();
    const std::vector<SelectionMethod> methods = {SelectionMethod::ELITISM,
                                                  SelectionMethod::ROULETTE,
                                                  SelectionMethod::RANK};
    const std::vector<double> sel_rates = {0.3, 0.6, 0.9};
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
// DQNNet — (same MLP architecture as baseline)
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
// DeceptiveRoyalRoadRLGA — RLGA<std::vector<uint8_t>, int>
// ============================================================

class DeceptiveRoyalRoadRLGA : public RLGA<std::vector<uint8_t>, int> {
  public:
    std::mt19937 rng;

    // Current GA parameters, set by apply_action()
    SelectionMethod current_method = SelectionMethod::ELITISM;
    double current_selection_rate = 0.5;
    double current_mutation_rate = 0.01;

    unsigned GENOME_LENGTH;
    unsigned BLOCK_SIZE;
    unsigned NUM_BLOCKS;
    double MAX_FITNESS;

    std::uniform_real_distribution<double> prob_dist{0.0, 1.0};
    std::uniform_int_distribution<uint8_t> bit_dist{0, 1};

    explicit DeceptiveRoyalRoadRLGA(int pop_size, double mut_rate,
                                    unsigned seed, unsigned genome_length,
                                    unsigned block_size)
        : RLGA(pop_size, mut_rate, seed), rng(seed),
          GENOME_LENGTH(genome_length), BLOCK_SIZE(block_size),
          NUM_BLOCKS(genome_length / block_size),
          MAX_FITNESS(static_cast<double>((genome_length / block_size) *
                                          (block_size * 2))),
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
        double fitness = 0.0;

        for (unsigned i = 0; i < NUM_BLOCKS; ++i) {
            unsigned ones = 0;
            for (unsigned j = 0; j < BLOCK_SIZE; ++j) {
                if (ind.genome[i * BLOCK_SIZE + j] == 1) {
                    ones++;
                }
            }

            if (ones == BLOCK_SIZE) {
                fitness += static_cast<double>(BLOCK_SIZE * 2); // Global peak
            } else {
                fitness += static_cast<double>(BLOCK_SIZE - ones); // Local trap
            }
        }
        return fitness;
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

    // Individual<std::vector<uint8_t>> crossover(
    //     Pairing<std::vector<uint8_t>>& pair) override {

    //     Individual<std::vector<uint8_t>> child;
    //     child.genome.resize(GENOME_LENGTH);

    //     std::uniform_int_distribution<unsigned> cut_dist(1, GENOME_LENGTH -
    //     1); unsigned cut = cut_dist(rng);

    //     const auto& p1 = population_[pair.parent_1_index].genome;
    //     const auto& p2 = population_[pair.parent_2_index].genome;

    //     std::copy(p1.begin(), p1.begin() + cut, child.genome.begin());
    //     std::copy(p2.begin() + cut, p2.end(), child.genome.begin() + cut);

    //     return child;
    // }

    Individual<std::vector<uint8_t>> crossover(
        Pairing<std::vector<uint8_t>>& pair) override {

        Individual<std::vector<uint8_t>> child;
        child.genome.resize(GENOME_LENGTH);

        const auto& p1 = population_[pair.parent_1_index].genome;
        const auto& p2 = population_[pair.parent_2_index].genome;

        // Number of blocks
        unsigned num_blocks = NUM_BLOCKS;

        // Choose cut in block space (not bit space)
        std::uniform_int_distribution<unsigned> block_cut_dist(1,
                                                               num_blocks - 1);
        unsigned block_cut = block_cut_dist(rng);

        // Convert block cut to bit index
        unsigned cut = block_cut * BLOCK_SIZE;

        // Copy whole blocks
        std::copy(p1.begin(), p1.begin() + cut, child.genome.begin());
        std::copy(p2.begin() + cut, p2.end(), child.genome.begin() + cut);

        return child;
    }

    void mutate(Individual<std::vector<uint8_t>>& ind) override {

        // Block-level mutation
        for (unsigned b = 0; b < NUM_BLOCKS; ++b) {

            if (prob_dist(rng) < current_mutation_rate) {

                // Flip entire block
                unsigned start = b * BLOCK_SIZE;
                unsigned end = start + BLOCK_SIZE;

                for (unsigned i = start; i < end; ++i) {
                    ind.genome[i] ^= 1;
                }
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
// DeceptiveRoyalRoadDQNAgent
// ============================================================

class DeceptiveRoyalRoadDQNAgent : public Agent<int, DeceptiveRoyalRoadRLGA> {
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

    explicit DeceptiveRoyalRoadDQNAgent(
        DeceptiveRoyalRoadRLGA&& ga_in, size_t gen_per_ep, size_t n_ep,
        double gamma, double epsilon, double epsilon_decay, double epsilon_min,
        double lr, size_t batch_size, size_t memory_size, unsigned rng_seed)
        : Agent<int, DeceptiveRoyalRoadRLGA>(std::move(ga_in), gen_per_ep, n_ep,
                                             gamma, epsilon, epsilon_decay,
                                             epsilon_min, lr),
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

    void post_step() override {
        update();
    }
};

static double best_fitness_of(DeceptiveRoyalRoadDQNAgent& agent) {
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

double get_param(const std::map<std::string, std::string>& params,
                 const std::string& key, double def) {
    auto it = params.find(key);
    if (it != params.end())
        return std::stod(it->second);
    return def;
}

int get_param(const std::map<std::string, std::string>& params,
              const std::string& key, int def) {
    auto it = params.find(key);
    if (it != params.end())
        return std::stoi(it->second);
    return def;
}

int main_rlga(const std::map<std::string, std::string>& params) {
    int pop_size = get_param(params, "pop_size", 10000);
    double mut_rate = get_param(params, "mut_rate", 0.01);
    unsigned seed = get_param(params, "seed", 42);
    size_t gen_per_ep = get_param(params, "gen_per_ep", 100);
    size_t n_ep = get_param(params, "n_ep", 200);
    double gamma = get_param(params, "gamma", 0.9);
    double epsilon = get_param(params, "epsilon", 0.5);
    double epsilon_decay = get_param(params, "epsilon_decay", 0.97);
    double epsilon_min = get_param(params, "epsilon_min", 0.05);
    double lr = get_param(params, "lr", 0.001);
    size_t batch_size = get_param(params, "batch_size", 128);
    size_t memory_size = get_param(params, "memory_size", 10000);
    unsigned genome_length =
        static_cast<unsigned>(get_param(params, "GENOME_LENGTH", 128));
    unsigned block_size =
        static_cast<unsigned>(get_param(params, "BLOCK_SIZE", 8));

    std::vector<double> custom_mut_rates;
    if (params.count("mut_rates")) {
        std::istringstream iss(params.at("mut_rates"));
        double m;
        while (iss >> m)
            custom_mut_rates.push_back(m);
    }
    init_action_space(custom_mut_rates);

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";
    std::cout << "Deceptive Royal Road Problem (" << genome_length
              << " bits | Block Size: " << block_size << ")\n\n";

    DeceptiveRoyalRoadRLGA ga(pop_size, mut_rate, seed, genome_length,
                              block_size);
    DeceptiveRoyalRoadDQNAgent agent(std::move(ga), gen_per_ep, n_ep, gamma,
                                     epsilon, epsilon_decay, epsilon_min, lr,
                                     batch_size, memory_size, seed);

    std::cout << "=== Deceptive Royal Road RLGA (DQN) ===\n"
              << "Action space: " << N_ACTIONS << " | State dim: " << STATE_DIM
              << "\n\n";

    size_t total_gens = 0;
    agent.ga.init_population();
    for (size_t ep = 0; ep < n_ep; ++ep) {
        // agent.ga.init_population();
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
            break;
        }
    }

    return 0;
}

// ============================================================
// main_par_rlga — parallel DQN via RL_IslandModel
// ============================================================

int main_par_rlga(const std::map<std::string, std::string>& params) {
    unsigned n_threads = get_param(params, "n_threads", 4);
    int pop_size = get_param(params, "pop_size", 10000);
    double mut_rate = get_param(params, "mut_rate", 0.01);
    size_t gen_per_ep = get_param(params, "gen_per_ep", 100);
    size_t n_ep = get_param(params, "n_ep", 200);
    double gamma = get_param(params, "gamma", 0.9);
    double epsilon = get_param(params, "epsilon", 0.5);
    double epsilon_decay = get_param(params, "epsilon_decay", 0.97);
    double epsilon_min = get_param(params, "epsilon_min", 0.05);
    double lr = get_param(params, "lr", 0.001);
    size_t batch_size = get_param(params, "batch_size", 128);
    size_t memory_size = get_param(params, "memory_size", 10000);
    unsigned n_migrants = get_param(params, "n_migrants", 500);
    double migration_probability =
        get_param(params, "migration_probability", 0.5);
    unsigned quorum = get_param(params, "quorum", 2);
    unsigned genome_length =
        static_cast<unsigned>(get_param(params, "GENOME_LENGTH", 128));
    unsigned block_size =
        static_cast<unsigned>(get_param(params, "BLOCK_SIZE", 8));

    std::vector<double> custom_mut_rates;
    if (params.count("mut_rates")) {
        std::istringstream iss(params.at("mut_rates"));
        double m;
        while (iss >> m)
            custom_mut_rates.push_back(m);
    }
    init_action_space(custom_mut_rates);

    std::cout << "Using device: "
              << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";
    std::cout << "Deceptive Royal Road Problem (" << genome_length
              << " bits | Block Size: " << block_size << ")\n\n";

    std::vector<unsigned> seeds;
    for (unsigned i = 0; i < n_threads; ++i)
        seeds.push_back(1234u + i * 1337u);

    std::vector<DeceptiveRoyalRoadDQNAgent> agents;
    agents.reserve(n_threads);
    for (unsigned i = 0; i < n_threads; ++i) {
        DeceptiveRoyalRoadRLGA ga(pop_size, mut_rate, seeds[i], genome_length,
                                  block_size);
        ga.init_population();
        agents.emplace_back(std::move(ga), gen_per_ep, n_ep, gamma, epsilon,
                            epsilon_decay, epsilon_min, lr, batch_size,
                            memory_size, seeds[i]);

        agents.back().epsilon_decay =
            epsilon_decay; // rescale to per-step for island model
    }

    RL_IslandModel<DeceptiveRoyalRoadDQNAgent> model(
        n_threads, migration_probability, n_migrants, quorum, std::move(agents),
        seeds);

    std::cout
        << "=== Parallel Deceptive Royal Road RLGA island model (DQN) ===\n"
        << "Threads=" << n_threads << " | quorum=" << quorum
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
        std::cout << "Island " << i << " best fitness: " << best->fitness
                  << " (generations: " << model.generations()[i] << ")\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " [rlga|par_rlga] [params.txt (optional)]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::map<std::string, std::string> params;

    if (argc >= 3) {
        params = load_params(argv[2]);
    }

    if (mode == "rlga")
        return main_rlga(params);
    else if (mode == "par_rlga")
        return main_par_rlga(params);
    else {
        std::cerr << "Invalid argument. Use 'rlga' or 'par_rlga'.\n";
        return 1;
    }
}
