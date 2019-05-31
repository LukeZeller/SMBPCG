# Current entry point for python project

import numpy as np
import cProfile

from common.simulation import SimulationProxy
from common.agents import create_human_agent, create_astar_agent, create_forced_agent
from evolution import evolve
from gan import generate

def test_gan():
    generate.load_generator()

    lv = np.random.uniform(-1, 1, 32)
    level = generate.apply_generator(lv)

    SimulationProxy(level, agent = create_human_agent()).invoke()

    # For testing purposes
    print(lv)

def test_fitness(random_latent_vector = True):
    generate.load_generator()
    if random_latent_vector:
        latent_vector = np.random.uniform(-1, 1, 32).tolist()
    else:
        latent_vector = [-0.78956354, 0.04543577, -0.96196604, 0.52659459, -0.12304981, 
                         0.09152696, 0.04387067, -0.31702606, 0.16287384, 0.98019136, 
                         -0.14670026, 0.69688305, 0.91131571, -0.23115624, 0.07971183, 
                         0.94697882, -0.78124791, 0.1948184, 0.68505739, 0.7450125, 
                         -0.8739045, -0.74168745, 0.55388925, 0.06871638, -0.27734117, 
                         0.17328284, 0.30875873, 0.85229842, 0.47069057, -0.77601111, 
                         0.83469813, 0.79881951]
    level = generate.apply_generator(latent_vector)
    fitness = evolve._fitness_function(level)
    return latent_vector, fitness


def test_evolution():
    level = evolve.run()
    print(level.get_data())
    while(True):
        SimulationProxy(level, agent = create_astar_agent()).invoke()


def test_json_level(json_fname):
    SimulationProxy.from_json_file(json_fname, human_tested=True).invoke()
    
if __name__ == '__main__':
    latent_vector, fitness = test_fitness(False)
    print(latent_vector)
    print("Fitness is: ", fitness)