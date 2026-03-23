#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

template <typename T> struct Individual {
    T genome;
    double fitness;
};

template <typename T> struct Pairing {
    size_t parent_1_index;
    size_t parent_2_index;
};

template <typename T> class GA {
  public:
    using genome_type = T;

    size_t init_pop_size;
    double mut_rate;
    double selection_rate;
    size_t rng_seed;
    size_t pop_size;

    int generation = 0;

    explicit GA(int pop_size, double mut_rate, unsigned seed)
        : init_pop_size(pop_size), pop_size(pop_size), mut_rate(mut_rate),
          rng_seed(seed) {}

    virtual ~GA() = default;

    virtual void init_population() = 0;
    virtual double compute_fitness(Individual<T>& individual) = 0;
    virtual std::vector<Pairing<T>> select_parents(
        double selection_rate,
        std::vector<Individual<T>>& population) = 0;
    virtual Individual<T> crossover(Pairing<T>& pair) = 0;
    virtual void mutate(Individual<T>& individual) = 0;
    virtual void drop_individuals(std::vector<Individual<T>>& population) = 0;
    virtual bool check_halt(std::vector<Individual<T>>& population) = 0;

    void step() {
        update_fitness();
        drop_individuals(population_);

        pairings_ = std::move(select_parents(population_));

        for (auto& pair : pairings_) {
            Individual<T> child = crossover(pair);
            mutate(child);

            child.fitness = compute_fitness(child);
            population_.push_back(std::move(child));
        }

        // TEMP
        std::cout << "[ gen" << generation << " ] " << std::endl;

        generation++;
        pop_size = population_.size();
    }

    void partition_by_fitness(unsigned n, bool front = true) {
        if (n > population_.size())
            n = population_.size();

        if (front) {
            std::nth_element(
                population_.begin(), population_.begin() + n, population_.end(),
                [](auto& a, auto& b) { return a.fitness > b.fitness; });
        } else {
            std::nth_element(
                population_.begin(), population_.end() - n, population_.end(),
                [](auto& a, auto& b) { return a.fitness > b.fitness; });
        }
    }

    std::vector<Individual<T>>& population() { return population_; }

    bool run() {
        try {
            while (true) {
                step();

                if (check_halt(population_)) {
                    break;
                }
            }
        } catch (...) {
            return false;
        }

        return true;
    }

  protected:
    std::vector<Individual<T>> population_;
    std::vector<Pairing<T>> pairings_;

    void update_fitness() {
        for (auto& ind : population_) {
            ind.fitness = compute_fitness(ind);
        }
    }
};