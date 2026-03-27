#pragma once
#include <thread>
#include <vector>
#include <barrier>
#include <atomic>
#include <cassert>
#include <concepts>
#include <functional>
#include "ga.hpp"

template <typename GAType>
concept IsGA = requires(GAType ga) {
  typename GAType::genome_type;
  { ga.step() };
  { ga.init_population() };
  { ga.population() };
  { ga.check_halt(ga.population()) } -> std::convertible_to<bool>;
  { ga.generation };
};

template <IsGA GAType>
class Island {
  public:
    explicit Island(
      GAType&& ga,
      unsigned migration_interval,
      std::barrier<std::function<void()>>& barrier,
      std::atomic<bool>& global_done
    )
      : ga(std::move(ga)),
        migration_interval(migration_interval),
        barrier_(barrier),
        global_done_(global_done)
    {}

    void island_run() {
      try {
        while (true) {
          island_step();

          if (ga.check_halt(ga.population())) {
            global_done_.store(true, std::memory_order_relaxed);
          }

          barrier_.arrive_and_wait();

          if (global_done_.load(std::memory_order_relaxed)) break;
        }
      } catch (...) {
        global_done_.store(true, std::memory_order_relaxed);
        barrier_.arrive_and_wait();
      }
    }

    GAType ga;

  private:
    unsigned migration_interval;
    std::barrier<std::function<void()>>& barrier_;
    std::atomic<bool>& global_done_;

    void island_step() {
      ga.step();

      if (ga.generation > 0 &&
          ga.generation % migration_interval == 0) {
        barrier_.arrive_and_wait();
      }
    }
};

template <IsGA GAType>
class IslandModel {
  public:
    unsigned n_threads;
    unsigned migration_interval;
    unsigned n_migrants;

    explicit IslandModel(
      unsigned n_threads,
      unsigned migration_interval,
      unsigned n_migrants,
      unsigned pop_size,
      double mut_rate,
      unsigned rng_seed
    )
      : n_threads(n_threads),
        migration_interval(migration_interval),
        n_migrants(n_migrants),
        barrier_(n_threads, [this]() { migrate(); })
    {
      for (unsigned i = 0; i < n_threads; i++) {
        GAType ga(pop_size, mut_rate, rng_seed + i * 1337);
        ga.init_population();

        islands_.emplace_back(
          std::move(ga),
          migration_interval,
          barrier_,
          global_done_
        );
      }
    }

    void island_model_run() {
      std::vector<std::thread> threads;

      for (unsigned i = 0; i < n_threads; i++) {
        threads.emplace_back(&Island<GAType>::island_run, &islands_[i]);
      }

      for (auto& t : threads) {
        t.join();
      }
    }

  private:
    using T = typename GAType::genome_type;

    std::vector<Island<GAType>> islands_;
    std::barrier<std::function<void()>> barrier_;
    std::atomic<bool> global_done_ = false;

    void migrate() {
      std::vector<std::vector<Individual<T>>> migrants(n_threads);

      for (unsigned i = 0; i < n_threads; i++) {
        auto& src = islands_[i].ga.population();

        assert(src.size() >= n_migrants);

        islands_[i].ga.partition_by_fitness(n_migrants);

        migrants[i].insert(
          migrants[i].end(),
          std::make_move_iterator(src.begin()),
          std::make_move_iterator(src.begin() + n_migrants)
        );

        src.erase(src.begin(), src.begin() + n_migrants);
      }

      for (unsigned i = 0; i < n_threads; i++) {
        auto& dst = islands_[(i + 1) % n_threads].ga.population();

        dst.insert(
          dst.end(),
          std::make_move_iterator(migrants[i].begin()),
          std::make_move_iterator(migrants[i].end())
        );
      }
    }
};