from common.simulation import SimulationProxy
from common.agents import create_forced_agent, create_astar_agent, create_human_agent

def can_complete_with_moves(level, moves, view=False):
    info = simulate_level_with_moves(level, moves, view)
    return info.level_passed()

def simulate_level_with_moves(level, moves, view=False):
    agent = create_forced_agent(moves)
    simulation_proxy = SimulationProxy(level, agent, view)
    simulation_proxy.invoke()
    return simulation_proxy.eval_info

def can_complete_with_astar(level, view=False):
    info = simulate_level_with_astar(level, view)
    return info.level_passed()

def simulate_level_with_astar(level, view=False):
    agent = create_astar_agent()
    simulation_proxy = SimulationProxy(level, agent, view)
    simulation_proxy.invoke()
    return simulation_proxy.eval_info

def can_complete_with_human(level, view=True):
    info = simulate_level_with_human(level, view)
    return info.level_passed()

def simulate_level_with_human(level, view=True):
    agent = create_human_agent()
    simulation_proxy = SimulationProxy(level, agent, view)
    simulation_proxy.invoke()
    return simulation_proxy.eval_info

def replay_level_with_human(level, view=True):
    agent = create_human_agent()
    simulation_proxy = SimulationProxy(level, agent, view)
    simulation_proxy.invokeTillStopped()
    return simulation_proxy.eval_info
