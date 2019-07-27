# Current entry point for python project

import numpy as np

from common.constants import DEFAULT_HYPERPARAMETER_CACHE_FILE, \
                             DEFAULT_LATENT_VECTOR, \
                             DEFAULT_CORRELATION_SUMMARY_FILE, \
                             EPS
from common.simulation import SimulationProxy, play_1_1
from common.simulate_agent import simulate_level_with_human, replay_level_with_human
from evolution import evolve
from evolution.evolve import default_hyperparameters, bad_hyperparameters
from gan import generator_client
from common.plotting import plot_to_file, _get_unique_file
from evolution.human_evaluation.hyperparameter_random_search \
    import PopulationGenerator, \
           HyperparameterCache, \
           human_evaluate_hyperparameters, \
           dummy_evaluate_hyperparameters
from timeit import default_timer as timer
from evolution.human_evaluation.hyperparameter_random_search import evaluate_level
import random
import json

### Testing Level Playing ###

def test_1_1():
    play_1_1()

def test_gan():
    generator_client.load_generator()
    latent_vector = np.random.uniform(-1, 1, 32)
    level = generator_client.apply_generator(latent_vector)
    print("Play level once:")
    simulate_level_with_human(level)
    print("Latent vector:\n", latent_vector)
    print("Evaluate level:")
    return evaluate_level(level)

### Testing Level Fitness ###

def test_fitness(random_latent_vector=True):
    generator_client.load_generator()
    if random_latent_vector:
        latent_vector = np.random.uniform(-1, 1, 32)
    else:
        latent_vector = DEFAULT_LATENT_VECTOR
    level = generator_client.apply_generator(latent_vector)
    fitness = evolve._multiple_run_fitness(level, default_hyperparameters)
    return latent_vector, fitness

def test_json_level(json_fname):
    SimulationProxy.from_json_file(json_fname, human_tested=True).invoke()

### Testing CMA ###

def test_evolution(hyperparameters = default_hyperparameters):
    generator_client.load_generator()
    level = evolve.run(hyperparameters, 5)
    print(level.get_data())
    replay_level_with_human(level)
    
def timing_run(hp, max_iterations):
    generator_client.load_generator()
    print("Iterations:", max_iterations)
    start = timer()
    level, avgs, mins = evolve.run(hp, max_iterations, return_fitnesses=True)
    end = timer()
    print("Time taken for run:", end - start, "(s)")
    return level, avgs, mins

def plot_run_fitness(hp, max_iterations):
    level, avgs, mins = timing_run(hp, max_iterations)
    generation_numbers = [i for i in range(len(avgs))]
    
    for name, ys in zip(["Average", "Minimum"], [avgs, mins]): 
        plot_to_file(title = f"{name} Fitness Value per Generation",
                     xs = generation_numbers, 
                     ys = ys,
                     xlabel = "Generation Number",
                     ylabel = f"{name} Fitness",
                     file_path = f"results/plots/fitness/{name}_cma_fitness_per_generation.png")
        with open(_get_unique_file(f"results/data/fitness/{name}_cma_fitness_per_generation.txt"), 'w') as json_file:
            json.dump(ys, json_file)
    return level

def test_correlation(hp1,
                     hp2,
                     small_iterations = 5,
                     big_iterations = 20):
    generator_client.load_generator()
    assert hp1 != hp2
    print("Iterations: ", small_iterations)
    small_levels = [{"level": evolve.run(hp1, small_iterations), 
                     "hyperparameter": hp1,
                     "iterations": small_iterations}, 
                    {"level": evolve.run(hp2, small_iterations), 
                     "hyperparameter": hp2,
                     "iterations": small_iterations}]
    print("Iterations: ", big_iterations)
    big_levels = [{"level": evolve.run(hp1, big_iterations), 
                   "hyperparameter": hp1,
                   "iterations": big_iterations}, 
                  {"level": evolve.run(hp2, big_iterations),
                   "hyperparameter": hp2,
                   "iterations": big_iterations}]
    
    swap_small_iterations = random.randint(0, 1) == 0
    swap_big_iterations = random.randint(0, 1) == 0
    
    if swap_small_iterations:
        print("SwapSmall")
        small_levels[0], small_levels[1] = small_levels[1], small_levels[0]
    if swap_big_iterations:
        print("SwapBig")
        big_levels[0], big_levels[1] = big_levels[1], big_levels[0]
        
    small_results = dict()
    big_results = dict()
        
    for data in small_levels:
        level, hp = data["level"], data["hyperparameter"]
        print("Evaluate the following level:")
        small_results[hp] = evaluate_level(level)
        
    for data in big_levels:
        level, hp = data["level"], data["hyperparameter"]
        print("Evaluate the following level:")
        big_results[hp] = evaluate_level(level)
        
    hp1_worse_for_small_iterations = small_results[hp1] < small_results[hp2]
    hp1_worse_for_big_iterations = big_results[hp1] < big_results[hp2]
    
    has_correlation = hp1_worse_for_small_iterations == hp1_worse_for_big_iterations
    correlation_value = 1 if has_correlation else 0
    with open(DEFAULT_CORRELATION_SUMMARY_FILE, 'a') as summary_file:
        print(correlation_value, file = summary_file)
        
def correlation_percentage():
    total = 0
    correlations = 0
    with open(DEFAULT_CORRELATION_SUMMARY_FILE, 'r') as summary_file:
        for line in summary_file:
            correlated = int(line.strip())
            assert type(correlated) == int
            assert correlated == 0 or correlated == 1
            total += 1
            correlations += correlated
    assert total > 0
    return 100.0 * float(correlations) / total

def correlation_test_script(iterations):
    print("Just a quick run to test that it works on your system")
    test_correlation(default_hyperparameters, bad_hyperparameters, 1, 2)
    print("Okay now we're goint to test things for real")
    for i in range(iterations):
        print("Iteration ", i)
        test_correlation(default_hyperparameters, bad_hyperparameters, 3, 10)
    pct = correlation_percentage()
    print(f"The % of times that a hyperparameter was better than the other, regardless of max_iters: {pct}%")
    return pct
            
### Testing Hyperparameter Training ###
       
def test_tuning(evaluation = human_evaluate_hyperparameters):
    population_generator = PopulationGenerator(evaluator = evaluation,
                                               population_size = 3,
                                               num_mutations_per_candidate = 5,
                                               step_size = 10.0,
                                               adaptive_step = True)
    cache = HyperparameterCache(generator = population_generator, storage_file = DEFAULT_HYPERPARAMETER_CACHE_FILE)
    return cache

def num_steps_to_optima(num_generations, 
                        evaluation,
                        optimal_fitness):
    population_generator = PopulationGenerator(evaluator = evaluation, 
                                               population_size = 4,
                                               num_mutations_per_candidate = 4,
                                               step_size = 5.0,
                                               adaptive_step = True)
    cache = HyperparameterCache(generator = population_generator, 
                                storage_file = "results/data/hyperparameters/hyperparameter_optima_check_cache.json")
    cache.reset()
    for generation in range(num_generations):
        candidate, fitness = cache.best()
        if abs(optimal_fitness - fitness) < EPS:
            print(f"Reached optima at generation {generation}")
            return generation
    print("Never got close to optima")
    return None


def plot_tuning(num_generations, evaluation):
    generation_numbers = []
    y_axis_names = ["Fitness",
             "0th Position",
             "1st Position",
             "2nd Position",
             "Hyper-Parameter Magnitude",
             "Hyper-Parameter Sum"]
    fitnesses, hp0s, hp1s, hp2s, magnitudes, sums = [ [] for _ in range(len(y_axis_names)) ]
    
    population_generator = PopulationGenerator(evaluator = evaluation, 
                                               population_size = 4,
                                               num_mutations_per_candidate = 4,
                                               step_size = 5.0,
                                               adaptive_step = True)
    cache = HyperparameterCache(generator = population_generator, 
                                storage_file = "results/data/hyperparameters/hyperparameter_plotting_cache.json")
    cache.reset()
    for generation in range(num_generations):
        candidate, fitness = cache.best()
        
        generation_numbers.append(generation)
        fitnesses.append(fitness)
        hp0s.append(candidate[0])
        hp1s.append(candidate[1])
        hp2s.append(candidate[2])
        sums.append(sum(candidate))
        magnitudes.append(np.linalg.norm(candidate))
        
        cache.get_next_generation()
    
    for y_axis_name, y_values in zip(y_axis_names, [fitnesses, hp0s, hp1s, hp2s, magnitudes, sums]):
        plot_to_file(title = f"{y_axis_name} Values per Generation",
                     xs = generation_numbers,
                     ys = y_values,
                     xlabel = "Generation Number",
                     ylabel = y_axis_name,
                     file_path = f"results/plots/hyperparameters/{y_axis_name}_per_random_search_generation.png")
    return cache

### Experiment Below ###

if __name__ == '__main__':
    plot_run_fitness(default_hyperparameters, 2)
