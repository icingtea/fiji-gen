#pragma once
#include <vector>

template <typename T>
struct Individual {
  T genome;
  double fitness;
};

template <typename T>
struct Pairing {
  Individual<T>* parent_1;
  Individual<T>* parent_2;
};

template <typename T>
class GA {
  public:
    const int pop_size;
    const double mut_rate;
    const unsigned rng_seed;

    int generation = 0;

    explicit GA(int pop_size, double mut_rate, unsigned seed):
      pop_size(pop_size),
      mut_rate(mut_rate),
      rng_seed(seed)
    {}
    
    virtual ~GA() = default;
        
    virtual void init_population() = 0;
    virtual double compute_fitness(Individual<T>* individual) = 0;
    virtual std::vector<Pairing<T>> select_parents() = 0;
    virtual Individual<T> crossover(Pairing<T>* pair) = 0;
    virtual void mutate(Individual<T>* individual) = 0;
    virtual void drop_individuals(std::vector<Individual<T>>& population) = 0;
    virtual bool check_halt(Individual<T>* population) = 0;

    void run() {
      while (!conclude_) {
        step();
        conclude_ = check_halt(&population_);
      }
    }

  private:
    std::vector<Individual<T>> population_;
    std::vector<Pairing<T>> pairings_;
    bool conclude_ = false;

    void update_fitness() {
      for (Individual<T>& ind : population_) {
       ind.fitness = compute_fitness(&ind); 
      }
    }

    void step() {
      pairings_ = std::move(select_parents());
      for (Pairing<T>& pair : pairings_) {
        Individual<T> child = crossover(&pair);
        mutate(&child);
        population_.push_back(std::move(child));
      }

      update_fitness();
      drop_individuals(population_);      
      
      generation++;
    }
};

