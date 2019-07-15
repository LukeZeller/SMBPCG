# Current entry point for python project

import numpy as np

from common.simulation import SimulationProxy, play_1_1
from common.agents import create_human_agent, create_astar_agent, create_forced_agent
from evolution import evolve
from gan import generator_client
import matplotlib.pyplot as plt
from evolution.human_evaluation.hyperparameter_random_search \
    import PopulationGenerator, HyperparameterCache

def test_gan():
    generator_client.load_generator()

    lv = np.random.uniform(-1, 1, 32)

    level = generator_client.apply_generator(lv)
    print("Play level once")
    SimulationProxy(level, agent=create_human_agent(), visualize = True).invoke()

    # For testing purposes
    print(lv)
    print("Evaluate level:")
    return evaluate_level(level)

def test_fitness(random_latent_vector=True):
    generator_client.load_generator()
    if random_latent_vector:
        latent_vector = np.random.uniform(-1, 1, 32)
    else:
        latent_vector = [-0.78956354, 0.04543577, -0.96196604, 0.52659459, -0.12304981,
                         0.09152696, 0.04387067, -0.31702606, 0.16287384, 0.98019136,
                         -0.14670026, 0.69688305, 0.91131571, -0.23115624, 0.07971183,
                         0.94697882, -0.78124791, 0.1948184, 0.68505739, 0.7450125,
                         -0.8739045, -0.74168745, 0.55388925, 0.06871638, -0.27734117,
                         0.17328284, 0.30875873, 0.85229842, 0.47069057, -0.77601111,
                         0.83469813, 0.79881951]
    fitness = evolve._fitness(latent_vector)
    return latent_vector, fitness

def test_evolution(hyperparameters = evolve.Hyperparameters()):
    generator_client.load_generator()
    level = evolve.run()
    print(level.get_data())
    SimulationProxy(level = level, 
                    agent = create_human_agent(), 
                    visualize = True).invokeTillStopped()
       
def test_tuning():
    mock_evaluation = lambda hp: hp[0]
    popgen = PopulationGenerator(evaluator = mock_evaluation, 
                                 population_size = 2)
    cache = HyperparameterCache(generator = popgen)
    return cache

def plot_tuning():
    mock_evaluation = lambda hp: hp[0]
    
    xs, fitnesses, hp0s, hp1s, hp2s, magnitudes, sums = [], [], [], [], [], [], []
    names = ["fitness", 
             "0th position", 
             "1st position", 
             "2nd position", 
             "magnitude", 
             "sums"]
    
    popgen = PopulationGenerator(evaluator = mock_evaluation)
    cache = HyperparameterCache(generator = popgen)
    for generation in range(2):
        candidate, fitness = tuple(*cache.best())
        
        xs.append(generation)
        fitnesses.append(fitness)
        hp0s.append(candidate[0])
        hp1s.append(candidate[1])
        hp2s.append(candidate[2])
        sums.append(sum(candidate))
        magnitudes.append(np.linalg.norm(candidate))
        
        cache.get_next_generation()
    
    for i, ys in enumerate([fitnesses, hp0s, hp1s, hp2s, magnitudes, sums]):
        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.set_title(names[i])
        fig.show()
    return cache

def test_json_level(json_fname):
    SimulationProxy.from_json_file(json_fname, human_tested=True).invoke()

if __name__ == '__main__':
    cache = test_tuning()
    
    