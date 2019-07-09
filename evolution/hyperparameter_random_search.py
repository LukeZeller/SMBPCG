import numpy as np
from evolution.evolve import Hyperparameters, run
from evolution.rubric import input_rubric, rubric_score
from common.agents import create_human_agent
from common.simulation import SimulationProxy

DIMENSION = len(Hyperparameters._fields)
POPULATION_SIZE = 16
NUM_BEST_CANDIDATES = 4
STEP_SIZE = 1.0

def random_hyperparameters():
    vec = np.random.uniform(size = DIMENSION)
    vec /= np.sum(vec)
    return Hyperparameters(*vec)

def point_on_random_sphere(dimension, radius = 1):
    vec = np.random.normal(size = dimension)
    vec /= np.linalg.norm(vec)
    vec *= radius
    return vec

def mutated(hp, step_size = 1):
    delta = point_on_random_sphere(DIMENSION, step_size)
    new_coordinates = []
    for i, val in enumerate(hp):
        new_coordinates.append(val + delta[i])
    return Hyperparameters(*new_coordinates)

def evaluate_level(level):
    SimulationProxy(level = level, 
                        agent = create_human_agent(), 
                        visualize = True).invokeTillStopped()
    rubric = input_rubric()
    return rubric_score(rubric)

def evaluate_hyperparameters(hp):
    print("Evaluating: ", hp)
    level = run(hp)
    return evaluate_level(level)

def best_of(population, num):
    population_fitness = [(evaluate_hyperparameters(candidate), candidate) 
                    for candidate in population]
    return sorted(population_fitness, key = lambda pf: pf[1])[:num][1]

def find_optimal_hyperparameters(num_iter):
    population = [random_hyperparameters() for _ in range(POPULATION_SIZE)]
    for _ in range(num_iter):
        for _ in range(POPULATION_SIZE):
            best_of_population = best_of(population, NUM_BEST_CANDIDATES)
            population = []
            for candidate in best_of_population:
                for _ in range(POPULATION_SIZE / NUM_BEST_CANDIDATES - NUM_BEST_CANDIDATES):
                    population.append(mutated(candidate, STEP_SIZE))
                population.append(candidate)
            assert len(population) == POPULATION_SIZE
    print("Best of population")
    return best_of(population, 1)
    
        
                
        
        
        
        
    