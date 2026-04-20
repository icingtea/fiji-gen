#pragma once
#include "ga.hpp"
#include <concepts>
#include <limits>
#include <span>

template <typename GAType>
concept IsGA = requires(GAType& ga) {
    typename GAType::genome_type;
    { ga.step() };
    { ga.init_population() };
    { ga.population() };
    { ga.check_halt(ga.population()) } -> std::convertible_to<bool>;
    { ga.generation };
};

struct GAState {
    size_t generation;
    size_t pop_size;
    size_t curr_ep;
    size_t n_ep;
    double best_fitness;
    double avg_fitness;
    double epsilon;
};

template <typename A, IsGA GAType>
class Agent {
    public:
        GAType ga;
        std::vector<A> actions;
        size_t gen_per_ep;
        size_t n_ep;
        size_t curr_ep;
        double gamma;
        double epsilon;
        double epsilon_decay;
        double epsilon_min;
        double alpha;
        GAState old_state;
        GAState new_state;
        double latest_reward;

        explicit Agent(
            GAType&& ga,
            size_t gen_per_ep,
            size_t n_ep,
            double gamma,
            double epsilon,
            double epsilon_decay,
            double epsilon_min,
            double alpha
        ) : ga(std::move(ga)),
            gen_per_ep(gen_per_ep),
            n_ep(n_ep),
            curr_ep(0),
            gamma(gamma),
            epsilon(epsilon),
            epsilon_decay(epsilon_decay),
            epsilon_min(epsilon_min),
            alpha(alpha),
            latest_reward(0.0) {}

        virtual ~Agent() = default;

        virtual A select_action() = 0;
        virtual void update() = 0;
        virtual void post_step() = 0;

        GAState get_state() {
            auto& pop = ga.population();
            double best = std::numeric_limits<double>::lowest();
            double total = 0.0;
            for (const auto& ind : pop) {
                best = std::max(best, ind.fitness);
                total += ind.fitness;
            }
            double avg = pop.empty() ? 0.0 : total / pop.size();

            return GAState{
                .generation   = ga.generation,
                .pop_size     = pop.size(),
                .curr_ep      = curr_ep,
                .n_ep         = n_ep,
                .best_fitness = best,
                .avg_fitness  = avg,
                .epsilon      = epsilon
            };
        }

        void step() {
            A action = select_action();
            old_state = get_state();
            latest_reward = ga.reward_step(action);
            new_state = get_state();
            post_step();
        }
};


template <typename T, typename A> class RLGA : public GA<T> {
    public:
        using GA<T>::update_fitness;
        using GA<T>::drop_individuals;
        using GA<T>::select_parents;
        using GA<T>::crossover;
        using GA<T>::mutate;
        using GA<T>::compute_fitness;

        using GA<T>::population_;
        using GA<T>::pairings_;
        using GA<T>::generation;
        using GA<T>::pop_size;

        virtual ~RLGA() = default;

        virtual double calculate_reward(
            std::span<const Individual<T>> old_population,
            std::vector<Individual<T>>& new_population,
            std::vector<Individual<T>>& children,
            std::vector<Pairing<T>>& pairings
        ) = 0;

        double reward_step(A action) {
            apply_action(action);
            update_fitness();
            drop_individuals(population_);

            pairings_ = std::move(select_parents(population_));
            std::vector<Individual<T>> children;

            size_t parent_count = population_.size();
            population_.reserve(parent_count + pairings_.size());
            std::span<const Individual<T>> parents(population_.data(), parent_count);

            for (auto& pair : pairings_) {
                Individual<T> child = crossover(pair);
                mutate(child);

                child.fitness = compute_fitness(child);
                children.push_back(child);
                population_.push_back(std::move(child));
            }

            // TEMP
            std::cout << "[ gen " << generation << " ]" << std::endl;

            generation++;
            pop_size = population_.size();

            return calculate_reward(parents, population_, children, pairings_);
        }

        virtual void apply_action(A action) = 0;
};