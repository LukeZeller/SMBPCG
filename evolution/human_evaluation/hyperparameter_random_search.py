import numpy as np
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
    level = run(hp)
    return evaluate_level(level)

## Random Search Helper Functions ##

def random_hyperparameters():
    vec = np.random.uniform(size = DIMENSION)
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
                 population_size = 4,
                 num_mutations_per_candidate = 4,
                 step_size = 0.1,
                 evaluator = None):
        self.dimension = len(Hyperparameters._fields)
        self.population_size = population_size
        self.num_mutations_per_candidate = num_mutations_per_candidate
        self.step_size = step_size
        self.evaluator = human_evaluate_hyperparameters \
                             if evaluator is None \
                             else evaluator
    
    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            candidate = random_hyperparameters()
            fitness = self.evaluator(candidate)
            population.append((candidate, fitness))
        return population
    
    def next_population(self, population):
        assert len(population) >= self.population_size
        
        possible_candidates = []
        for candidate, fitness in population:
            for  _ in range(self.num_mutations_per_candidate):
                mutated_candidate = mutated(candidate, self.step_size)
                mutated_candidate_fitness = self.evaluator(mutated_candidate)
                possible_candidates.append((mutated_candidate, 
                                            mutated_candidate_fitness))
            possible_candidates.append((candidate, fitness))
        return best_of(possible_candidates, self.population_size)
    
### Class that handles how human evaluation results are stored via json ###    
    
CURRENT = "CURRENT"
OLD = "OLD"
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
        with open(self.storage_file, 'w') as json_file:
            json.dump(HyperparameterCache._encode(self.ratings), json_file)
        
    def _initialize(self):
        self.ratings = {CURRENT : [], OLD : []}
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
    