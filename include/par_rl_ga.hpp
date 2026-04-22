#pragma once
#include "ga.hpp"
#include "rl_ga.hpp"
#include "par_ga.hpp"
#include <atomic>
#include <cassert>
#include <concepts>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>


// RL_Island: standalone RL island (does NOT inherit Island<GAType>).
// Mirrors Island<GAType> migration logic but delegates stepping to an
// Agent<ActionType, GAType>.  A must be a concrete subclass of
// Agent<ActionType, GAType> (e.g. RoyalRoadQAgent).
template <typename A>
class RL_Island {
  public:
    using GAType = std::remove_cvref_t<decltype(std::declval<A&>().ga)>;

    size_t id;
    size_t n_migrants;
    size_t quorum;
    std::atomic<unsigned>& done_counter;
    A agent;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

    explicit RL_Island(
        A&& agent_in,
        double migration_probability,
        unsigned island_id,
        unsigned n_migrants,
        unsigned quorum,
        std::atomic<unsigned>& done_counter,
        std::vector<std::unique_ptr<migrant_buffer<GAType>>>& migrant_buffers,
        unsigned rng_seed)
        : agent(std::move(agent_in)),
          migration_probability_(migration_probability),
          id(island_id),
          n_migrants(n_migrants),
          migrant_buffers_(migrant_buffers),
          self_migrant_buffer_(*migrant_buffers_[id]),
          quorum(quorum),
          done_counter(done_counter),
          neighbor_migrant_buffer_(
              *migrant_buffers_[(id + 1) % migrant_buffers_.size()]),
          rng(rng_seed),
          dist(0.0, 1.0) {}

    void island_run() {
        bool counted = false;
        size_t n_ep = agent.n_ep;
        size_t gen_per_ep = agent.gen_per_ep;

        for (size_t ep = 0; ep < n_ep && !counted; ++ep) {
            // Run one episode: gen_per_ep steps (generations)
            for (size_t step = 0; step < gen_per_ep; ++step) {
                // Check global halt before each step
                if (done_counter.load(std::memory_order_relaxed) >= quorum) {
                    return;
                }

                island_step(true);

                // Epsilon decay happens inside agent.step() already.
                // Check if this island reached the optimum
                if (!counted &&
                    agent.ga.check_halt(agent.ga.population())) {
                    unsigned expected_done =
                        done_counter.load(std::memory_order_relaxed);
                    while (expected_done < quorum) {
                        if (done_counter.compare_exchange_weak(
                                expected_done, expected_done + 1,
                                std::memory_order_acq_rel,
                                std::memory_order_relaxed)) {
                            counted = true;
                            break;
                        }
                    }
                    if (counted) break;
                }
            }

            

            // After each episode: sync target network + decay epsilon
            agent.sync_target();
            agent.decay_epsilon();
            agent.curr_ep++;

            // Drain migrant buffers between episodes
            receive_migrants();

            if (done_counter.load(std::memory_order_relaxed) >= quorum) {
                break;
            }
        }

        // If this island finished all episodes without hitting quorum,
        // still count it as done so the model can terminate.
        if (!counted) {
            unsigned expected_done =
                done_counter.load(std::memory_order_relaxed);
            while (expected_done < quorum) {
                if (done_counter.compare_exchange_weak(
                        expected_done, expected_done + 1,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    break;
                }
            }
        }
    }

  private:
    double migration_probability_;
    std::vector<std::unique_ptr<migrant_buffer<GAType>>>& migrant_buffers_;
    migrant_buffer<GAType>& self_migrant_buffer_;
    migrant_buffer<GAType>& neighbor_migrant_buffer_;

    // island_step: perform one generation (step) on this island.
    // Epsilon decay is handled inside agent.step().
    void island_step(bool active) {
        receive_migrants();
        if (active) {
            if (agent.ga.generation % 100 == 0) {
                const auto& pop = agent.ga.population();
                if (!pop.empty()) {
                    auto best_it = std::max_element(pop.begin(), pop.end(), [](const auto& a, const auto& b) {
                        return a.fitness < b.fitness;
                    });
                    double total_fit = 0.0;
                    for (const auto& ind : pop) total_fit += ind.fitness;
                    double avg_fit = total_fit / pop.size();
                    std::cout << "[ Island " << id
                              << " | ep " << agent.curr_ep
                              << " | gen " << agent.ga.generation
                              << " ] Best: " << best_it->fitness
                              << " | Avg: " << avg_fit
                              << " | eps: " << agent.epsilon << "\n";
                }
            }
            agent.step();   // selects action, runs GA step, decays epsilon
            send_migrants();
        }
    }

    void send_migrants() {
        if (dist(rng) < migration_probability_ &&
            !neighbor_migrant_buffer_.full.load(std::memory_order_acquire)) {
            neighbor_migrant_buffer_.buffer.clear();

            auto& population = agent.ga.population();
            unsigned pop_sz = static_cast<unsigned>(population.size());
            unsigned round_n_migrants =
                (pop_sz > 1) ? std::min(static_cast<unsigned>(n_migrants), pop_sz - 1) : 0;
            if (round_n_migrants == 0)
                return;

            // Build an index vector and do a partial Fisher-Yates shuffle
            // to pick round_n_migrants random (distinct) positions.
            std::vector<unsigned> indices(pop_sz);
            std::iota(indices.begin(), indices.end(), 0u);
            for (unsigned i = 0; i < round_n_migrants; ++i) {
                std::uniform_int_distribution<unsigned> pick(i, pop_sz - 1);
                std::swap(indices[i], indices[pick(rng)]);
            }

            // Sort the selected indices descending so erasing by position
            // doesn't shift indices we haven't processed yet.
            std::sort(indices.begin(), indices.begin() + round_n_migrants,
                      std::greater<unsigned>());

            for (unsigned i = 0; i < round_n_migrants; ++i) {
                unsigned idx = indices[i];
                neighbor_migrant_buffer_.buffer.push_back(
                    std::move(population[idx]));
                population.erase(population.begin() + idx);
            }

            neighbor_migrant_buffer_.full.store(true,
                                               std::memory_order_release);
        }
    }

    void receive_migrants() {
        if (self_migrant_buffer_.full.load(std::memory_order_acquire)) {
            auto& population = agent.ga.population();
            auto& buf = self_migrant_buffer_.buffer;

            for (auto& individual : buf) {
                population.push_back(std::move(individual));
            }

            buf.clear();
            self_migrant_buffer_.full.store(false, std::memory_order_release);
        }
    }
};

// RL_IslandModel: manages a ring of RL_Island instances running in parallel.
// Caller constructs the concrete agents (one per island) and passes them in.
template <typename A>
class RL_IslandModel {
  public:
    using GAType = typename RL_Island<A>::GAType;

    unsigned n_threads;
    double migration_probability;
    unsigned n_migrants;
    std::atomic<unsigned> done_counter{0};
    std::vector<std::unique_ptr<migrant_buffer<GAType>>> migrant_buffers;

    explicit RL_IslandModel(unsigned n_threads, double migration_probability,
                            unsigned n_migrants, unsigned quorum,
                            std::vector<A>&& agents,
                            std::vector<unsigned> rng_seeds)
        : n_threads(n_threads),
          migration_probability(migration_probability),
          n_migrants(n_migrants) {

        assert(agents.size() == n_threads);
        assert(rng_seeds.size() == n_threads);

        migrant_buffers.reserve(n_threads);
        for (unsigned i = 0; i < n_threads; i++) {
            migrant_buffers.push_back(
                std::make_unique<migrant_buffer<GAType>>());
        }

        // Reserve so that emplace_back never reallocates — RL_Island holds
        // references that must not dangle.
        islands_.reserve(n_threads);
        for (unsigned i = 0; i < n_threads; i++) {
            islands_.emplace_back(std::move(agents[i]), migration_probability,
                                  i, n_migrants, quorum, done_counter,
                                  migrant_buffers, rng_seeds[i]);
        }
    }

    bool island_model_run() {
        try {
            std::vector<std::thread> threads;

            for (unsigned i = 0; i < n_threads; i++) {
                threads.emplace_back(&RL_Island<A>::island_run,
                                     &islands_[i]);
            }

            for (auto& t : threads) {
                t.join();
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::vector<std::vector<Individual<typename GAType::genome_type>>>
    populations() {
        std::vector<std::vector<Individual<typename GAType::genome_type>>>
            result;
        for (auto& island : islands_) {
            result.push_back(island.agent.ga.population());
        }
        return result;
    }

    std::vector<unsigned> generations() {
        std::vector<unsigned> gens;
        for (auto& island : islands_) {
            gens.push_back(island.agent.ga.generation);
        }
        return gens;
    }

    // Returns true if island i reached the optimal solution (check_halt == true).
    bool island_solved(unsigned i) {
        auto& pop = islands_[i].agent.ga.population();
        return islands_[i].agent.ga.check_halt(
            const_cast<std::vector<Individual<typename GAType::genome_type>>&>(pop));
    }

  private:
    std::vector<RL_Island<A>> islands_;
};
