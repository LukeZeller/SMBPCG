import math

import cma

from common.level import Level
from common.simulation import SimulationProxy
from gan import generate


LEVEL_LENGTH = 704

HARD_JUMP_WEIGHT = 10
MEDIUM_JUMP_WEIGHT = 5
EASY_JUMP_WEIGHT = 3
TRIVIAL_JUMP_WEIGHT = 1

# TODO: Generalize and make magic numbers into parameters w/ defaults
# TODO: Refactor into class for cleanliness

cma_es = None

def _fitness(latent_vector):
    generate.load_generator()
    level = generate.apply_generator(latent_vector)
    return _fitness_function(level)

# Sample fitness function based on EvalutionInfo information
def _fitness_function(level):
    sim_proxy = SimulationProxy(level)
    sim_proxy.invoke()

    if sim_proxy.get_distance_passed() < LEVEL_LENGTH:
        return - sim_proxy.get_distance_passed() / LEVEL_LENGTH
    return (
        - sim_proxy.get_trivial_jump_cnt() * TRIVIAL_JUMP_WEIGHT
        - sim_proxy.get_easy_jump_cnt() * EASY_JUMP_WEIGHT
        - sim_proxy.get_medium_jump_cnt() * MEDIUM_JUMP_WEIGHT
        - sim_proxy.get_hard_jump_cnt() * HARD_JUMP_WEIGHT
    )
    

def run():
    cma_es = cma.CMAEvolutionStrategy([0] * 32, 1 / math.sqrt(32), {'maxiter':1000})
    cma_es.optimize(_fitness)

    print(" ---- CMA RESULTS --- ")
    print(" --- From Framework --- ")
    cma_es.result_pretty()
    print(" --- From Recalculation --- ")
    best_lv = cma_es.best.get()[0]
    print("Best Latent Vector: " + str(best_lv))
    print("Best Fitness: " + str(_fitness(best_lv)))

    return generate.apply_generator(best_lv)
