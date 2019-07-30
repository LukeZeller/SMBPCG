import numpy as np
from common.constants import DEBUG_PRINT, CMA_ITERATIONS_FOR_HUMAN_EVALUATION, EPS
from evolution.evolve import Hyperparameters, run
from evolution.human_evaluation.rubric import input_rubric, rubric_score
from common.agents import create_human_agent
from common.simulation import SimulationProxy
import json
import os

DIMENSION = len(Hyperparameters._fields)

## Evaluation Functions ## 

def evaluate_level(level):
    SimulationProxy(level = level, 
                    agent = create_human_agent(), 
                    visualize = True).invokeTillStopped()
    rubric = input_rubric()
    return rubric_score(rubric)

def human_evaluate_hyperparameters(hp):
    print("Evaluating: ", hp)
    level = run(hp, max_iterations = CMA_ITERATIONS_FOR_HUMAN_EVALUATION)
    return evaluate_level(level)

""" Hard-coded evaluator w/ an optima at (3, 4, 5) """
def dummy_evaluate_hyperparameters(hp):
    return -( (hp[0] - 3) ** 2 + (hp[1] - 4) ** 2 + (hp[2] - 5) ** 2)

## Random Search Helper Functions ##

def random_hyperparameters():
    vec = np.random.uniform(low = -1.0, high = 1.0, size = DIMENSION)
    return Hyperparameters(*vec)

def point_on_random_sphere(dimension, radius = 1):
    vec = np.random.normal(size = dimension)
    vec /= np.linalg.norm(vec)
    vec *= radius
    return vec

def mutated(hp, step_size = 1):
    delta = point_on_random_sphere(DIMENSION, step_size)
    new_coordinates = np.zeros(DIMENSION)
    for i, val in enumerate(hp):
        new_coordinates[i] = val + delta[i]
    return Hyperparameters(*new_coordinates)

def best_of(population_fitness, num):
    return sorted(population_fitness, key = lambda pf: pf[1], reverse = True)[:num]

### Class the packages parameters to determine how populations are generated ###

class PopulationGenerator:
    def __init__(self, 
                 evaluator,
                 population_size = 4,
                 num_mutations_per_candidate = 4,
                 step_size = 0.1,
                 adaptive_step = False):
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_mutations_per_candidate = num_mutations_per_candidate
        self.step_size = step_size
        self.adaptive_step = adaptive_step
        self.iteration = 0
    
    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            candidate = random_hyperparameters()
            fitness = self.evaluator(candidate)
            population.append((candidate, fitness))
        self.iteration = 0
        return population
    
    def radius(self):
        if not self.adaptive_step:
            return self.step_size
        else:
            return self.step_size / (1.0 + self.iteration)
    
    def next_population(self, population):
        assert len(population) * (self.num_mutations_per_candidate + 1) >= self.population_size
        
        possible_candidates = []
        for candidate, fitness in population:
            for  _ in range(self.num_mutations_per_candidate):
                mutated_candidate = mutated(candidate, self.radius())
                mutated_candidate_fitness = self.evaluator(mutated_candidate)
                possible_candidates.append((mutated_candidate, 
                                            mutated_candidate_fitness))
            possible_candidates.append((candidate, fitness))
        if DEBUG_PRINT:
            print("Iteration: ", self.iteration, "Radius: ", self.radius())
        self.iteration += 1
        return best_of(possible_candidates, self.population_size)
    
    def stop(self):
        return self.radius() <= EPS
    
### Class that handles how human evaluation results are stored via json ###    
    
CURRENT = "CURRENT"
OLD = "OLD"
AVERAGE = "AVERAGE"
class HyperparameterCache:
    def __init__(self, 
                 storage_file,
                 generator):
        self.storage_file = storage_file
        self.population_generator = generator
        if not os.path.isfile(self.storage_file):
            self._initialize()
        else:
            self._load()
    
    def _load(self):
        with open(self.storage_file, 'r') as json_file:
            self.ratings = HyperparameterCache._decode(json.load(json_file))
        
    def _save(self):
        self._save_current_average()
        with open(self.storage_file, 'w') as json_file:
            json.dump(HyperparameterCache._encode(self.ratings), json_file)
            
    def _save_current_average(self):
        current_sum = 0.0
        for hp, rating in self.ratings[CURRENT]:
            current_sum += rating
        self.ratings[AVERAGE].append(current_sum / len(self.ratings[CURRENT]))
        
    def _initialize(self):
        self.ratings = {CURRENT : [], OLD : [], AVERAGE : []}
        self.ratings[CURRENT] = self.population_generator.initial_population()
        self._save()
        
    @staticmethod
    def _tuple_to_string(t):
        return " ".join(map(str, t))
    
    @staticmethod
    def _string_to_tuple(s):
        return Hyperparameters(*map(float, s.split()))
        
    @staticmethod
    def _encode(data):
        result = data.copy()
        result[CURRENT] = [(HyperparameterCache._tuple_to_string(hyperparameter), rating)
                              for hyperparameter, rating in data[CURRENT]]
        result[OLD] = [(HyperparameterCache._tuple_to_string(hyperparameter), rating)
                              for hyperparameter, rating in data[OLD]]
        return result
    
    @staticmethod        
    def _decode(data):
        result = data.copy()
        result[CURRENT] = [(HyperparameterCache._string_to_tuple(hyperparameter), rating)
                              for hyperparameter, rating in data[CURRENT]]
        result[OLD] = [(HyperparameterCache._string_to_tuple(hyperparameter), rating)
                              for hyperparameter, rating in data[OLD]]
        return result
    
    def get_next_generation(self):
        next_generation = self.population_generator.next_population(self.ratings[CURRENT])
        self.ratings[OLD] += self.ratings[CURRENT]
        self.ratings[CURRENT] = next_generation
        self._save()
        
    def best(self):
        return best_of(self.ratings[CURRENT], 1)[0]
        
    def __repr__(self):
        return json.dumps(self.ratings, 
                          sort_keys = True, 
                          indent = 2,
                          separators=(',', ': '))
    
    def reset(self):
        self._initialize()
        
    def stop(self):
        return self.population_generator.stop()
    
    def get_averages(self):
        return self.ratings[AVERAGE]
        
    