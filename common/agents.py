import jnius

# --- Java class names ---
ASTAR_AGENT_CLASS = 'competition.icegic.robin.AStarAgent'
HUMAN_AGENT_CLASS = 'ch.idsia.ai.agents.human.HumanKeyboardAgent'
FORCED_AGENT_CLASS = 'edu.unc.cs.smbpcg.simulator.ForcedActionAgent'
SIMULATION_HANDLER_CLASS = 'edu.unc.cs.smbpcg.simulator.SimulationHandler'

# Standard Library classes
_J_ArrayList = jnius.autoclass('java.util.ArrayList')
_J_Integer = jnius.autoclass('java.lang.Integer')

# Project classes
_J_AStarAgent = jnius.autoclass(ASTAR_AGENT_CLASS)
_J_HumanKeyboardAgent = jnius.autoclass(HUMAN_AGENT_CLASS)
_J_ForcedActionAgent = jnius.autoclass(FORCED_AGENT_CLASS)

_human_agents = [_J_HumanKeyboardAgent]

def create_astar_agent():
    return _J_AStarAgent()

def create_human_agent():
    return _J_HumanKeyboardAgent()

def create_forced_agent(moves):
    return _J_ForcedActionAgent(moves)

def is_human(agent):
    return any(map(lambda agent_type: isinstance(agent, agent_type), _human_agents))