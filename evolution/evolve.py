import functools
import math
from multiprocessing import Pool
import os
import time

import cma

from common import constants
from common.simulation import SimulationProxy
from evolution.feasible_shifts import number_of_shifts_and_jumps
from gan import generate


HARD_JUMP_WEIGHT = 10
MEDIUM_JUMP_WEIGHT = 5
EASY_JUMP_WEIGHT = 3
TRIVIAL_JUMP_WEIGHT = 1
PENALTY_FOR_FAILURE = 1000

REPEAT_REWARD = 400

# Number of times the A* agent is invoked on each sample during evolution
TRIALS_PER_SAMPLE = 20
MAX_ITERS = 30
PARALLELIZE_TRIALS = False

# TODO: Generalize and make magic numbers into parameters w/ defaults
# TODO: Refactor into class for cleanliness

### WARNING: CONSTRUCTION ZONE ###

cma_es = None

def _fitness(latent_vector):
    generate.load_generator()
    level = generate.apply_generator(latent_vector)

    min_fit = constants.INF
    fits = []
    passed = []

    if PARALLELIZE_TRIALS:
        with Pool() as p:
            fit_fn_with_info = functools.partial(_fitness_function, ret_passed_bool = True)
            fit_data = p.map(fit_fn_with_info, [level for _ in range(TRIALS_PER_SAMPLE)])
            fits = [elem[0] for elem in fit_data]
            passed = [1 if elem[1] else 0 for elem in fit_data]

            ### TEST OUTPUT ###
            print("Trial fitnesses: " + str(fits))
            print("Passed trial indicators: " + str(passed))
    else:
        for t_itr in range(TRIALS_PER_SAMPLE):
            trial_fit, level_passed = _fitness_function(level, True)
            
            ### TEST ###
            print("Trial " + str(t_itr) + " Fitness: " + str(trial_fit))
        
            fits.append(trial_fit)
            passed.append(1 if level_passed else 0)
            min_fit = min(trial_fit, min_fit)

    passed_cnt = sum(passed)
    ### TEST ###
    print("PASSED COUNT: " + str(passed_cnt))
    
    if passed_cnt > 0:
        avg_passed_fit = sum([fits[i] * passed[i] for i in range(TRIALS_PER_SAMPLE)]) / passed_cnt
        pct_failed = 1 - passed_cnt / TRIALS_PER_SAMPLE
        avg_passed_fit -= REPEAT_REWARD * pct_failed
        print("AVG PASSED FIT: " + str(avg_passed_fit))
        return avg_passed_fit
    else:
        avg_fit = sum(fits) / TRIALS_PER_SAMPLE
        print("AVG FIT: " + str(avg_fit))
        return avg_fit

# Sample fitness function based on EvalutionInfo information
def _fitness_function(level, ret_passed_bool = False):
    sim_proxy = SimulationProxy(level)
    sim_proxy.invoke()
    info = sim_proxy.eval_info
    passed = info.level_passed()
    
    if not passed:
        difficulty = _calculate_difficulty_for_failure(info)
    else:
        difficulty = _calculate_difficulty_for_success(info, level)
    fitness = -difficulty
    
    if ret_passed_bool:
        return fitness, passed
    else:
        return fitness
    
def _calculate_difficulty_for_failure(info):
    fraction_of_level_completed = float(info.lengthOfLevelPassedPhys) / constants.LEVEL_LENGTH
    return 1 - fraction_of_level_completed

def _calculate_difficulty_for_success(info, level):
    num_shifts, num_jumps = number_of_shifts_and_jumps(info, level)
    print("Shifts: ", num_shifts)
    print("Num jumps: ", num_jumps)
    average_number_of_shifts_per_jump = float(num_shifts) / num_jumps
    # The more that the jumps can be shifted, the easier the level is
    return 1 / average_number_of_shifts_per_jump

def run():
    cma_es = cma.CMAEvolutionStrategy([0] * 32, 1 / math.sqrt(32), {'maxiter':MAX_ITERS})

    avg_fits = []
    #cma_es.optimize(_fitness)
    gen_itr = 0

    print("Pool sz: " + str(os.cpu_count()))

    p_sz = 0
    while not cma_es.stop():
        population = cma_es.ask()
        p_sz = len(population)
        best_lv = None
        best_fitness = constants.INF
        with Pool() as p:
            print(" ---- Generation " + str(gen_itr) + " ----")
            fits = p.map(_fitness, population)
            print("GEN FITS: " + str(fits))
            print("GEN AVG: " + str(sum(fits) / len(fits)))
            avg_fits.append(sum(fits) / len(fits))
            cma_es.tell(population, fits)
            for i in range(p_sz):
                if fits[i] <= best_fitness:
                    best_lv = population[i]
                    best_fitness = fits[i]
            gen_itr += 1

    print("ALL GEN AVG FITS: " + str(avg_fits))
    
    print(" ---- CMA RESULTS --- ")
    print(" --- From Framework --- ")
    cma_es.result_pretty()
    print(" --- From Recalculation --- ")
    cma_es.best.get()
    print(type(cma_es.best.get()))

    best_lv_f = cma_es.best.get()[0]

    print("Best Latent Vector (According to framework): " + ', '.join(str(best_lv_f).split()))
    print("Corresponding fitness: " + str(_fitness(best_lv_f)))

    print("Best Latent Vector (According to manual bookkeeping): " + ', '.join(str(best_lv).split()))
    print("Corresponding fitness: " + str(_fitness(best_lv)))
    print("Saved best fitness: " + str(best_fitness))
    
    return generate.apply_generator(best_lv_f)
