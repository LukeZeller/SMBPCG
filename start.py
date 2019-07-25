# Current entry point for python project

import numpy as np

from common.constants import DEFAULT_HYPERPARAMETER_CACHE_FILE, \
                             DEFAULT_LATENT_VECTOR, \
                             DEFAULT_CORRELATION_SUMMARY_FILE
from common.simulation import SimulationProxy, play_1_1
from common.simulate_agent import simulate_level_with_human, replay_level_with_human
from evolution import evolve
from evolution.evolve import Hyperparameters
from gan import generator_client
import matplotlib.pyplot as plt
from evolution.human_evaluation.hyperparameter_random_search \
    import PopulationGenerator, HyperparameterCache
from timeit import default_timer as timer
from evolution.human_evaluation.hyperparameter_random_search import evaluate_level
from random import randint

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
    fitness = evolve._multiple_run_fitness(level, Hyperparameters())
    return latent_vector, fitness

def test_json_level(json_fname):
    SimulationProxy.from_json_file(json_fname, human_tested=True).invoke()

### Testing CMA ###

def test_evolution(hyperparameters = Hyperparameters()):
    generator_client.load_generator()
    level = evolve.run(hyperparameters)
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

def plot_run_fitness(hp):
    level, avgs, mins = timing_run(hp)
    generation_numbers = [i for i in range(len(avgs))]
    for name, ys in zip(["avgs", "mins"], [avgs, mins]): 
        fig, ax = plt.subplots()
        ax.plot(generation_numbers, ys)
        ax.set_title("Level fitness value per generation (" + name + ")")
        fig.show()
    return level

def test_correlation(hp1,
                     hp2,
                     small_iterations = 5,
                     big_iterations = 20):
    assert hp1 != hp2
    small_levels = [{"level": evolve.run(hp1, small_iterations), 
                     "hyperparameter": hp1,
                     "iterations": small_iterations}, 
                    {"level": evolve.run(hp2, small_iterations), 
                     "hyperparameter": hp2,
                     "iterations": small_iterations}]
    big_levels = [{"level": evolve.run(hp1, big_iterations), 
                   "hyperparameter": hp1,
                   "iterations": big_iterations}, 
                    {"level": evolve.run(hp2, big_iterations),
                     "hyperparameter": hp2,
                     "iterations": big_iterations}]
    
    swap_small_iterations = randint(0, 1) == 0
    swap_big_iterations = randint(0, 1) == 0
    
    if swap_small_iterations:
        small_levels[0], small_levels[1] = small_levels[1], small_levels[0]
    if swap_big_iterations:
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
            
### Testing Hyperparameter Training ###
       
def test_tuning():
    mock_evaluation = lambda hp: hp[0]
    population_generator = PopulationGenerator(evaluator = mock_evaluation,
                                               population_size = 2)
    cache = HyperparameterCache(generator = population_generator, storage_file = DEFAULT_HYPERPARAMETER_CACHE_FILE)
    return cache

def plot_tuning(num_generations):
    mock_evaluation = lambda hp: -(hp[0] ** 2 + hp[1] ** 2 + hp[2] ** 2)

    generation_numbers = []
    y_axis_names = ["fitness",
             "0th position", 
             "1st position", 
             "2nd position", 
             "magnitude", 
             "sums"]
    fitnesses, hp0s, hp1s, hp2s, magnitudes, sums = [ [] for _ in range(len(y_axis_names)) ]
    
    population_generator = PopulationGenerator(evaluator = mock_evaluation)
    cache = HyperparameterCache(generator = population_generator, storage_file = DEFAULT_HYPERPARAMETER_CACHE_FILE)
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
        fig, ax = plt.subplots()
        ax.plot(generation_numbers, y_values)
        ax.set_title(y_axis_name)
        fig.show()
    return cache

### Experiment Below ###

if __name__ == '__main__':
    pass
    