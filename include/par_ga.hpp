#pragma once
#include "ga.hpp"
#include <atomic>
#include <cassert>
#include <concepts>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

template <typename GAType>
concept IsGA = requires(GAType ga) {
    typename GAType::genome_type;
    { ga.step() };
    { ga.init_population() };
    { ga.population() };
    { ga.check_halt(ga.population()) } -> std::convertible_to<bool>;
    { ga.generation };
};

template <IsGA GAType> struct migrant_buffer {
    using T = typename GAType::genome_type;
    std::vector<Individual<T>> buffer;
    std::atomic<bool> full{false};
};

template <IsGA GAType> class Island {
  public:
    unsigned id;
    unsigned n_migrants;
    unsigned quorum;
    std::atomic<unsigned>& done_counter;
    GAType ga;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

    explicit Island(
        GAType&& ga, double migration_probability, unsigned island_id,
        unsigned n_migrants, unsigned quorum,
        std::atomic<unsigned>& done_counter,
        std::vector<std::unique_ptr<migrant_buffer<GAType>>>& migrant_buffers,
        unsigned rng_seed)
        : ga(std::move(ga)), migration_probability(migration_probability),
          id(island_id), n_migrants(n_migrants),
          migrant_buffers_(migrant_buffers),
          self_migrant_buffer_(*migrant_buffers_[id]), quorum(quorum),
          done_counter(done_counter),
          neighbor_migrant_buffer_(
              *migrant_buffers_[(id + 1) % migrant_buffers_.size()]),
          rng(rng_seed), dist(0.0, 1.0) {}

    void island_run() {
        bool counted = false;
        while (true) {
            island_step(!counted);

            if (!counted && ga.check_halt(ga.population())) {
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

    void island_step(bool active) {

        receive_migrants();
        if (active) {
            // TEMP
            std::cout << "[ gen " << ga.generation << " | id " << id << " ] "
                      << std::endl;
            ga.step();
            send_migrants();
        }
    };

    void send_migrants() {
        if (dist(rng) < migration_probability &&
            !neighbor_migrant_buffer_.full.load(std::memory_order_acquire)) {
            neighbor_migrant_buffer_.buffer.clear();

            unsigned round_n_migrants =
                (ga.pop_size > 1) ? std::min(n_migrants, ga.pop_size - 1) : 0;
            if (round_n_migrants == 0)
                return;

            ga.partition_by_fitness(round_n_migrants);
            auto& population = ga.population();

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

    void receive_migrants() {
        if (self_migrant_buffer_.full.load(std::memory_order_acquire)) {
            auto& population = ga.population();
            auto& buf = self_migrant_buffer_.buffer;

            for (auto& individual : buf) {
                population.push_back(std::move(individual));
            }

            buf.clear();
            self_migrant_buffer_.full.store(false, std::memory_order_release);
        }
    };
};

template <IsGA GAType> class IslandModel {
  public:
    using T = typename GAType::genome_type;

    unsigned n_threads;
    double migration_probability;
    unsigned n_migrants;
    std::atomic<unsigned> done_counter{0};
    std::vector<std::unique_ptr<migrant_buffer<GAType>>> migrant_buffers;

    explicit IslandModel(unsigned n_threads, double migration_probability,
                         unsigned n_migrants, unsigned pop_size,
                         double mut_rate, unsigned quorum,
                         std::vector<unsigned> rng_seeds)
        : n_threads(n_threads), migration_probability(migration_probability),
          n_migrants(n_migrants) {

        migrant_buffers.reserve(n_threads);
        for (unsigned i = 0; i < n_threads; i++) {
            migrant_buffers.push_back(
                std::make_unique<migrant_buffer<GAType>>());
        }

        assert(rng_seeds.size() == n_threads);
        for (unsigned i = 0; i < n_threads; i++) {
            GAType ga(pop_size, mut_rate, rng_seeds[i]);
            ga.init_population();

            islands_.emplace_back(std::move(ga), migration_probability, i,
                                  n_migrants, quorum, done_counter,
                                  migrant_buffers, rng_seeds[i]);
        }
    }

    bool island_model_run() {
        try {
            std::vector<std::thread> threads;

            for (unsigned i = 0; i < n_threads; i++) {
                threads.emplace_back(&Island<GAType>::island_run, &islands_[i]);
            }

            for (auto& t : threads) {
                t.join();
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    std::vector<std::vector<Individual<T>>> populations() {
        std::vector<std::vector<Individual<T>>> populations;
        for (Island<GAType>& island : islands_) {
            populations.push_back(island.ga.population());
        }

        return populations;
    }

  private:
    std::vector<Island<GAType>> islands_;
};