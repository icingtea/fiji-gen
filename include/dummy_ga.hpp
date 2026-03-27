#pragma once
#include "ga.hpp"
#include <random>
#include <algorithm>

class DummyGA : public GA<int> {
  public:
    using Base = GA<int>;
    using Base::GA;

    std::mt19937 rng;
    std::uniform_int_distribution<int> dist{-10, 10};

    explicit DummyGA(unsigned pop_size, double mut_rate, unsigned seed)
      : Base(pop_size, mut_rate, seed), rng(seed) {}

    void init_population() override {
      population_.clear();
      for (unsigned i = 0; i < pop_size; i++) {
        Individual<int> ind;
        ind.genome = dist(rng);
        ind.fitness = 0;
        population_.push_back(ind);
      }
      update_fitness();
    }

    double compute_fitness(Individual<int>* ind) override {
      return static_cast<double>(ind->genome);
    }

    std::vector<Pairing<int>> select_parents() override {
      std::sort(population_.begin(), population_.end(),
        [](auto& a, auto& b){ return a.fitness > b.fitness; });

      std::vector<Pairing<int>> pairs;

      for (size_t i = 0; i + 1 < population_.size() / 2; i += 2) {
        pairs.push_back(Pairing<int>{
          population_[i],
          population_[i + 1]
        });
      }

      return pairs;
    }

    Individual<int> crossover(Pairing<int>* pair) override {
      Individual<int> child;
      child.genome = (pair->parent_1.genome + pair->parent_2.genome) / 2;
      return child;
    }

    void mutate(Individual<int>* ind) override {
      if ((double)rng() / rng.max() < mut_rate) {
        ind->genome += dist(rng);
      }
    }

    void drop_individuals(std::vector<Individual<int>>& population) override {
      if (population.size() > pop_size) {
        std::sort(population.begin(), population.end(),
          [](auto& a, auto& b){ return a.fitness > b.fitness; });

        population.resize(pop_size);
      }
    }

    bool check_halt(std::vector<Individual<int>>& population) override {
      auto best = std::max_element(population.begin(), population.end(),
        [](auto& a, auto& b){ return a.fitness < b.fitness; });

      return best->genome >= 1000;
    }
};