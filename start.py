# Current entry point for python project

import numpy as np

from common.constants import DEFAULT_HYPERPARAMETER_CACHE_FILE, \
                             DEFAULT_LATENT_VECTOR, \
                             DEFAULT_CORRELATION_SUMMARY_FILE, \
                             EPS, \
                             DEFAULT_LEVEL_ROOT_DIRECTORY
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
from common.level import load_level_from_ascii_str, level_to_ascii_str, level_to_jpg
import random
import json

### Testing Level Generation and Playing ###

def test_1_1():
    play_1_1()
    
def random_level():
    generator_client.load_generator()
    latent_vector = np.random.uniform(-1, 1, 32)
    print("Latent vector:\n", latent_vector)
    level = generator_client.apply_generator(latent_vector)
    return level

def test_gan():
    level = random_level()
    print("Play level once:")
    simulate_level_with_human(level)
    print("Evaluate level:")
    return evaluate_level(level)

### Test Textual level Representation ###
    
def test_level_to_ascii_and_back():
    level = random_level()
    ascii_repr = level_to_ascii_str(level)
    level_copy = load_level_from_ascii_str(ascii_repr)
    assert level_copy.width == level.width
    assert level_copy.height == level.height
    assert ascii_repr == level_to_ascii_str(level_copy)
    print("Checks passed!")

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
        print("Swap")
        small_levels[0], small_levels[1] = small_levels[1], small_levels[0]
    if swap_big_iterations:
        print("Swap")
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
                                               population_size = 5,
                                               num_mutations_per_candidate = 1,
                                               step_size = 10.0,
                                               adaptive_step = True)
    cache = HyperparameterCache(generator = population_generator, storage_file = DEFAULT_HYPERPARAMETER_CACHE_FILE)
    return cache

def num_steps_to_optima(num_generations, 
                        evaluation,
                        optimal_fitness):
    population_generator = PopulationGenerator(evaluator = evaluation, 
                                               population_size = 3,
                                               num_mutations_per_candidate = 2,
                                               step_size = 10.0,
                                               adaptive_step = True)
    cache = HyperparameterCache(generator = population_generator, 
                                storage_file = "results/data/hyperparameters/hyperparameter_optima_check_cache.json")
    cache.reset()
    min_dist = float('inf')
    for generation in range(num_generations):
        candidate, fitness = cache.best()
        distance = abs(optimal_fitness - fitness)
        if distance < EPS:
            print(f"Reached optima at generation {generation}")
            return generation
        min_dist = min(min_dist, distance)
        if generation == 12:
            print(min_dist)
        cache.get_next_generation()
    print(f"Never got close to optima, closest point was {min_dist} away")
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

def hp_random_search_script(iterations):
    print("Initial generation")
    cache = test_tuning()
    for i in range(iterations):
        print("Iteration: ", i)
        cache.get_next_generation()
        if cache.stop():
            print("Ending early because radius is too small")
            break
    print("Best hyperparameters: ", cache.best())
    return cache.best()

## Pipeline
    
def save_latent_vector(lv, name):
    root_dir = DEFAULT_LEVEL_ROOT_DIRECTORY
    with open(_get_unique_file(f"{root_dir}/latent_vectors/{name}.txt"), 'w') as lv_file:
        lv_as_string = " ".join([str(elem) for elem in lv])
        print(lv_as_string, file = lv_file)

def save_level(level, name, is_pre_lstm):
    root_dir = DEFAULT_LEVEL_ROOT_DIRECTORY
    lstm_dir = "prelstm" if is_pre_lstm else "postlstm"
    level_to_jpg(level, 
                 _get_unique_file(f"{root_dir}/level_images/{lstm_dir}/{name}"),
                 trim_buffer = False)
    text = level_to_ascii_str(level)
    with open(_get_unique_file(f"{root_dir}/level_asciis/{lstm_dir}/{name}.txt"), 'w') as level_file:
        print(text, file = level_file)
    
def generate_best_level_for_hyperparameters(hp, cma_iterations):
    level, latent_vector, fitness = evolve.run(default_hyperparameters, 
                                      cma_iterations, 
                                      return_fitnesses = False,
                                      return_level_properties = True)
    fitness = int(fitness)
    identifier = f"{fitness}_{cma_iterations}"
    
    save_latent_vector(latent_vector, identifier)
    
    save_level(level, identifier, is_pre_lstm = True)
    
    level_as_text = level_to_ascii_str(level)
    cleaned_level_as_text = level_as_text ## RHS should be: lstm_client.apply_generator(level_as_text)
    cleaned_level = load_level_from_ascii_str(cleaned_level_as_text)
    
    save_level(cleaned_level, identifier, is_pre_lstm = False)
        
    return cleaned_level

### Experiment Below ###

if __name__ == '__main__':
    res = generate_best_level_for_hyperparameters(default_hyperparameters, 1)
