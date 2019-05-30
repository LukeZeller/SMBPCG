import functools
import math
from multiprocessing import Pool
import os

import cma

from common import constants
from common.level import Level
from common.simulation import SimulationProxy
from gan import generate

HARD_JUMP_WEIGHT = 10
MEDIUM_JUMP_WEIGHT = 5
EASY_JUMP_WEIGHT = 3
TRIVIAL_JUMP_WEIGHT = 1

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

    penalty = 0
    
    if not sim_proxy.eval_info.level_passed():
        penalty = 1 - sim_proxy.eval_info.lengthOfLevelPassedPhys / constants.LEVEL_LENGTH

        ### TEST ###
        '''
        print(sim_proxy.eval_info.lengthOfLevelPassedPhys)
        print(sim_proxy.eval_info.toString())
        print(sim_proxy.eval_info.marioRanOutOfTime)
        print(sim_proxy.eval_info.marioDiedToFall)
        print(sim_proxy.eval_info.marioDiedToEnemy)
        '''

        if sim_proxy.eval_info.marioRanOutOfTime:
            penalty *= 100
        elif sim_proxy.eval_info.marioDiedToFall:
            penalty *= 50
        elif sim_proxy.eval_info.marioDiedToEnemy:
            penalty *= 20
        else:
            raise RuntimeError( ("If mario failed, it should have been due to  running out "
                                 "of time, running into an enemy, or falling into a hole!")
            )
        
    fitness = (
        - sim_proxy.eval_info.trivialJumpActionsPerformed * TRIVIAL_JUMP_WEIGHT
        - sim_proxy.eval_info.easyJumpActionsPerformed * EASY_JUMP_WEIGHT
        - sim_proxy.eval_info.mediumJumpActionsPerformed * MEDIUM_JUMP_WEIGHT
        - sim_proxy.eval_info.hardJumpActionsPerformed * HARD_JUMP_WEIGHT
        + penalty )

    if ret_passed_bool:
        return fitness, sim_proxy.eval_info.level_passed()
    else:
        return fitness

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
            print("GEN AVG: " + str(sum(fits)/len(fits)))
            avg_fits.append(sum(fits)/len(fits))
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
