# Current entry point for python project

import numpy as np

from common.simulation import SimulationProxy, play_1_1
from common.agents import create_human_agent, create_astar_agent, create_forced_agent
from evolution import evolve
from gan import generator_client
from evolution.hyperparameter_random_search import find_optimal_hyperparameters


def test_gan():
    generator_client.load_generator()

    lv = np.random.uniform(-1, 1, 32)

    level = generate.apply_generator(lv)
    SimulationProxy(level, agent=create_human_agent()).invoke()

    # For testing purposes
    print(lv)
    
    while True:
        SimulationProxy(level = level, 
                        agent = create_human_agent(), 
                        visualize = True).invoke()
        message = input("Enter STOP to end loop")
        if message == "STOP":
            break

def test_fitness(random_latent_vector=True):
    generate.load_generator()
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
    while(True):
        SimulationProxy(level = level, agent = create_human_agent(), visualize = True).invoke()
        message = input("Enter STOP to end loop")
        if message == "STOP":
            break
        
def test_tuning():
    best = find_optimal_hyperparameters(0)
    return best

def test_json_level(json_fname):
    SimulationProxy.from_json_file(json_fname, human_tested=True).invoke()

if __name__ == '__main__':
    hp = test_evolution()
    print(hp)
