#include <iostream>
#include "par_ga.hpp"
#include "dummy_ga.hpp"

int main() {
  constexpr unsigned n_threads = 4;
  constexpr unsigned migration_interval = 5;
  constexpr unsigned n_migrants = 2;
  constexpr unsigned pop_size = 50;
  constexpr double mut_rate = 0.3;
  constexpr unsigned seed = 42;

  IslandModel<DummyGA> model(
    n_threads,
    migration_interval,
    n_migrants,
    pop_size,
    mut_rate,
    seed
  );

  model.island_model_run();

  std::cout << "Finished without deadlocking 🎉\n";
}