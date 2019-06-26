from common.simulation import SimulationProxy
from common.agents import create_forced_agent


def can_complete(level, moves, view=False):
    info = simulate_level_with_moves(level, moves, view)
    return info.level_passed()


def simulate_level_with_moves(level, moves, view=False):
    agent = create_forced_agent(moves)
    simulation_proxy = SimulationProxy(level, agent, view)
    simulation_proxy.invoke()
    return simulation_proxy.eval_info
