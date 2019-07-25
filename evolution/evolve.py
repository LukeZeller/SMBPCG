import functools
import math
import os
import cma

from common.constants import DEBUG_PRINT, INF
from evolution.level_difficulty.difficulty \
    import calculate_difficulty_for_failure, calculate_difficulty_for_success
from common.simulate_agent \
    import simulate_level_with_astar
from gan import generator_client
from typing import NamedTuple
from multiprocessing import Pool
from time import time

class Hyperparameters(NamedTuple):
    SUCCESS_COEFFICIENT : float = 0.0025
    FAILURE_PERCENTAGE_COEFFICIENT : float = 1.0000
    ALL_FAILURE_COEFFICIENT : float = 1.0000

default_hyperparameters = Hyperparameters()
bad_hyperparameters = Hyperparameters(0.1, 0.2, 0.3)

# Number of times the A* agent is invoked on each sample during evolution
TRIALS_PER_SAMPLE = 10
MAX_ITERS = 50
PARALLELIZE_ITERATIONS = True

### WARNING: CONSTRUCTION ZONE ###

cma_es = None

# Sample fitness function based on EvalutionInfo information
def _fitness(level):
    info = simulate_level_with_astar(level)
    passed = info.level_passed()
    if DEBUG_PRINT:
        print("Passed:", passed)
    if not passed:
        return _fitness_failure(info, level)
    else:
        return _fitness_success(info, level)
    
def _fitness_success(eval_info, level):
    assert eval_info.level_passed()
    difficulty = calculate_difficulty_for_success(eval_info, level)
    fitness = -difficulty # Negate difficulty since fitness is minimized
    return fitness
    
def _fitness_failure(eval_info, level):
    assert not eval_info.level_passed()
    difficulty = calculate_difficulty_for_failure(eval_info)
    fitness = difficulty
    return fitness

def _multiple_run_fitness(level, hp):
    """ 
    First, astar agent is run TRIALS_PER_SAMPLE times on the given level
    
    If the agent passes the level at least once, we use the agent's eval info to
        calculate the success fitness using the '_fitness' function
        Afterwards we weigh this success fitness along with a reward for each
        time that the astar agent failed
        We add these weighted values to get the final level fitness value
    If the astar agent always fails, we find the average fitness value of
        failure and weigh it to get the final level fitness value
    """
    success_info = None
    sum_failure_fitness = 0
    failure_cnt = 0

    for _ in range(TRIALS_PER_SAMPLE):
        info = simulate_level_with_astar(level)
        if DEBUG_PRINT:
            print("Level passed:", info.level_passed())
        if info.level_passed():
            success_info = info
        else:
            failure_cnt += 1
            sum_failure_fitness += _fitness_failure(info, level)
            
    all_failures = failure_cnt == TRIALS_PER_SAMPLE
    if all_failures:
        average_failure_fitness = float(sum_failure_fitness) / failure_cnt
        weighted_fitness_value = hp.ALL_FAILURE_COEFFICIENT * average_failure_fitness
        if DEBUG_PRINT:
            print("All failed")
            print("Average failure fitness:", average_failure_fitness)
    else:
        failure_percentage = float(failure_cnt) / TRIALS_PER_SAMPLE
        success_fitness = _fitness_success(success_info, level)
        weighted_fitness_value = hp.SUCCESS_COEFFICIENT * success_fitness - \
                                 hp.FAILURE_PERCENTAGE_COEFFICIENT * failure_percentage
        if DEBUG_PRINT:
            print("Not all failed")
            print("Failure percentage:", failure_percentage)
            print("Success fitness:", success_fitness)
    return weighted_fitness_value
    
def _latent_vector_fitness(latent_vector, hp):
    generator_client.load_generator()
    level = generator_client.apply_generator(latent_vector)
    return _multiple_run_fitness(level, hp)

def run(hyperparameters, max_iterations = MAX_ITERS, return_fitnesses = False):
    fitness = functools.partial(_latent_vector_fitness, hp = hyperparameters)
    
    cma_es = cma.CMAEvolutionStrategy([0] * 32, 1 / math.sqrt(32), {'maxiter':max_iterations})

    avg_fits = []
    min_fits = []
    gen_itr = 0

    print("Pool sz: " + str(os.cpu_count()))

    p_sz = 0
    while not cma_es.stop():
        population = cma_es.ask()
        p_sz = len(population)
        print("Population size: ", p_sz)
        best_lv = None
        best_fitness = INF
        
        start_total_time = time()
        print(" ---- Generation " + str(gen_itr) + " ----")
        
        start_fitness_time = time()
        if PARALLELIZE_ITERATIONS:
            with Pool() as pool:
                fits = list(pool.map(fitness, population))
        else:
            fits = list(map(fitness, population))
        print("Time for fitness: ", time() - start_fitness_time)
        
        start_append_time = time()
        avg_fits.append(sum(fits) / len(fits))
        min_fits.append(min(fits))
        print("Time for append: ", time() - start_append_time)
        
        start_tell_time = time()
        cma_es.tell(population, fits)
        print("Time for tell: ", time() - start_tell_time)

        start_best_find_time = time()
        for i in range(p_sz):
            if fits[i] <= best_fitness:
                best_lv = population[i]
                best_fitness = fits[i]
        print("Time for best fitness find: ", time() - start_best_find_time)
        
        gen_itr += 1
        print("Total loop time: ", time() - start_total_time)

    if DEBUG_PRINT:
        print("ALL GEN AVG FITS: " + str(avg_fits))
        
        print(" ---- CMA RESULTS --- ")
        print(" --- From Framework --- ")
        cma_es.result_pretty()
        print(" --- From Recalculation --- ")
        cma_es.best.get()
        print(type(cma_es.best.get()))

    best_lv_f = cma_es.best.get()[0]

    if DEBUG_PRINT:
        print("Best Latent Vector (According to framework): " + ', '.join(str(best_lv_f).split()))
        print("Corresponding fitness: " + str(fitness(best_lv_f)))
    
        print("Best Latent Vector (According to manual bookkeeping): " + ', '.join(str(best_lv).split()))
        print("Corresponding fitness: " + str(fitness(best_lv)))
        print("Saved best fitness: " + str(best_fitness))
    print("Type of best_lv_f: ", type(best_lv_f))   
    if return_fitnesses:
        return generator_client.apply_generator(best_lv_f), avg_fits, min_fits
    else:
        return generator_client.apply_generator(best_lv_f)
