import functools
import math
from multiprocessing import Pool
import os
import cma

from common.constants import DEBUG_PRINT, INF
from evolution.level_difficulty.difficulty \
    import calculate_difficulty_for_failure, calculate_difficulty_for_success
from common.simulate_agent import simulate_level_with_astar
from gan import generator_client

from typing import NamedTuple

class Hyperparameters(NamedTuple):
    SUCCESS_COEFFICIENT : float = 0.0025
    FAILURE_COEFFICIENT : float = 1.0000
    ALL_FAILURE_COEFFICIENT : float = 1.0000

# Number of times the A* agent is invoked on each sample during evolution
TRIALS_PER_SAMPLE = 1
MAX_ITERS = 1

# TODO: Refactor into class for cleanliness

### WARNING: CONSTRUCTION ZONE ###

cma_es = None

# Sample fitness function based on EvalutionInfo information
def _fitness(level, hp, ret_passed_bool = False):
    info = simulate_level_with_astar(level)
    passed = info.level_passed()
    
    if not passed:
        difficulty = calculate_difficulty_for_failure(info)
    else:
        difficulty = calculate_difficulty_for_success(info, level)
    fitness = -difficulty if passed else difficulty
    
    if ret_passed_bool:
        return fitness, passed
    else:
        return fitness

def _multiple_run_fitness(level, hp):
    min_fit = INF
    fits = []
    passed = []

    for t_itr in range(TRIALS_PER_SAMPLE):
        trial_fit, level_passed = _fitness(level, hp, ret_passed_bool = True)
        
        if DEBUG_PRINT:
            print("Trial " + str(t_itr) + " Fitness: " + str(trial_fit))
    
        fits.append(trial_fit)
        passed.append(1 if level_passed else 0)
        min_fit = min(trial_fit, min_fit)

    passed_cnt = sum(passed)
    failed_pct = float(TRIALS_PER_SAMPLE - passed_cnt) / TRIALS_PER_SAMPLE
    ### TEST ###
    if DEBUG_PRINT:
        print("PASSED COUNT: " + str(passed_cnt))
    
    if passed_cnt > 0:
        avg_passed_fit = sum([fits[i] * passed[i] for i in range(TRIALS_PER_SAMPLE)]) / passed_cnt
        
        scaled_fitness_value = hp.SUCCESS_COEFFICIENT * avg_passed_fit
        scaled_repeat_reward = -1 * hp.FAILURE_COEFFICIENT * failed_pct
        if DEBUG_PRINT:
            print("AVG PASSED FIT: " + str(avg_passed_fit))
        return scaled_fitness_value + scaled_repeat_reward
    else:
        avg_fit = sum(fits) / TRIALS_PER_SAMPLE
        if DEBUG_PRINT:
            print("AVG FIT: " + str(avg_fit))
        scaled_fitness_value = hp.ALL_FAILURE_COEFFICIENT * avg_fit
        return scaled_fitness_value
    
def _latent_vector_fitness(latent_vector, hp):
    generator_client.load_generator()
    level = generator_client.apply_generator(latent_vector)
    return _multiple_run_fitness(level, hp)

def run(hyperparameters):
    fitness = functools.partial(_latent_vector_fitness, hp = hyperparameters)
    
    cma_es = cma.CMAEvolutionStrategy([0] * 32, 1 / math.sqrt(32), {'maxiter':MAX_ITERS})

    avg_fits = []
    gen_itr = 0

    print("Pool sz: " + str(os.cpu_count()))

    p_sz = 0
    while not cma_es.stop():
        population = cma_es.ask()
        p_sz = len(population)
        best_lv = None
        best_fitness = INF
        with Pool() as p:
            fits = list(map(fitness, population))
            if DEBUG_PRINT:
                print(" ---- Generation " + str(gen_itr) + " ----")
                print("GEN FITS: " + str(fits))
                print("GEN AVG: " + str(sum(fits) / len(fits)))
            avg_fits.append(sum(fits) / len(fits))
            cma_es.tell(population, fits)
            for i in range(p_sz):
                if fits[i] <= best_fitness:
                    best_lv = population[i]
                    best_fitness = fits[i]
            gen_itr += 1

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
    return generator_client.apply_generator(best_lv_f)
