# Current entry point for python project

import numpy as np

from common.simulation import SimulationProxy
from evolution import evolve
from gan import generate

def test_gan():
    generate.load_generator()

    latent_vector = np.random.uniform(-1, 1, 32)
    level = generate.apply_generator(latent_vector)

    sim_proxy = SimulationProxy(level, testing_mode=False)
    sim_proxy.set_visualization(True)
    sim_proxy.invoke()
    
    fitness = evolve._fitness_function(level)

    # For testing purposes
    return latent_vector, sim_proxy.eval_info, fitness


def test_evolution():
    level = evolve.run()
    print(level.get_data())
    while(True):
        SimulationProxy(level, testing_mode=True).invoke()


def test_json_level(json_fname):
    SimulationProxy.from_json_file(json_fname, testing_mode=True).invoke()
    
if __name__ == '__main__':
    # test_json_level("test.json")
    latent_vector, info, fitness = test_gan()
    print(latent_vector)
    print(info)
    print("Fitness is: ", fitness)