# Current entry point for python project

import numpy as np

from common.simulation import SimulationProxy
from evolution import evolve
from gan import generate

def test_gan():
    generate.load_generator()

    lv = np.random.uniform(-1, 1, 32)
    level = generate.apply_generator(lv)

    SimulationProxy(level, testing_mode=True).invoke()

    # For testing purposes
    print(lv)

    
def test_evolution():
    level = evolve.run()

    SimulationProxy(level, testing_mode=True).invoke()

if __name__ == '__main__':
    
    test_evolution()
