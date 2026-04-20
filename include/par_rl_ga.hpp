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

template <typename GAType>
concept IsGA = requires(GAType& ga) {
    typename GAType::genome_type;
    { ga.step() };
    { ga.init_population() };
    { ga.population() };
    { ga.check_halt(ga.population()) } -> std::convertible_to<bool>;
    { ga.generation };
};

template <typename A, IsGA GAType> class RL_Island : Island {
    public:
        using Island<GAType>::id;
        using Island<GAType>::n_migrants;
        using Island<GAType>::quorum;
        using Island<GAType>::done_counter;
        using Island<GAType>::rng;
        using Island<GAType>::dist;

        Agent<A, GAType> agent;

        explicit RL_Island (
            Agent<A, GAType>&& Agent, double migration_probability, unsigned island_id,
            unsigned n_migrants, unsigned quorum,
            std::atomic<unsigned>& done_counter,
            std::vector<std::unique_ptr<migrant_buffer<GAType>>>& migrant_buffers,
            unsigned rng_seed
        )
            : agent(std::move(agent)), migration_probability(migration_probability),
            id(island_id), n_migrants(n_migrants),
            migrant_buffers_(migrant_buffers),
            self_migrant_buffer_(*migrant_buffers_[id]), quorum(quorum),
            done_counter(done_counter),
            neighbor_migrant_buffer_(
              *migrant_buffers_[(id + 1) % migrant_buffers_.size()]
            ),
            rng(rng_seed), dist(0.0, 1.0) {}

        void island_run() override {
            bool counted = false;
            while (true) {
                island_step(!counted);
            
                if (!counted && agent.ga.check_halt(agent.ga.population())) {
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
                }
            
                if (done_counter.load(std::memory_order_relaxed) >= quorum) {
                    break;
                }
            }
        }

    private:
        double migration_probability;
        std::vector<std::unique_ptr<migrant_buffer<GAType>>>& migrant_buffers_;
        migrant_buffer<GAType>& self_migrant_buffer_;
        migrant_buffer<GAType>& neighbor_migrant_buffer_;

        void island_step(bool active) override {

            receive_migrants();
            if (active) {
                // TEMP
                std::cout << "[ gen " << agent.ga.generation << " | id " << id << " ] "
                          << std::endl;
                agent.step();
                send_migrants();
            }
        };

        void send_migrants() override {
            if (dist(rng) < migration_probability &&
                !neighbor_migrant_buffer_.full.load(std::memory_order_acquire)) {
                neighbor_migrant_buffer_.buffer.clear();

                unsigned round_n_migrants =
                    (agent.ga.pop_size > 1) ? std::min(n_migrants, agent.ga.pop_size - 1) : 0;
                if (round_n_migrants == 0)
                    return;

                agent.ga.partition_by_fitness(round_n_migrants);
                auto& population = agent.ga.population();

                for (unsigned i = 0; i < round_n_migrants; i++) {
                    neighbor_migrant_buffer_.buffer.push_back(
                        std::move(population[i]));
                }

                population.erase(population.begin(),
                                 population.begin() + round_n_migrants);
                neighbor_migrant_buffer_.full.store(true,
                                                    std::memory_order_release);
            }
        }

        void receive_migrants() override {
            if (self_migrant_buffer_.full.load(std::memory_order_acquire)) {
                auto& population = agent.ga.population();
                auto& buf = self_migrant_buffer_.buffer;

                for (auto& individual : buf) {
                    population.push_back(std::move(individual));
                }

                buf.clear();
                self_migrant_buffer_.full.store(false, std::memory_order_release);
            }
        };
};
