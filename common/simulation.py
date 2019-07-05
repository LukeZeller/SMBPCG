from config import config_mgr
from common import constants
import common.agents
import common.level

config_mgr.setup_environment()
import jnius

# --- Java class names ---
EVALUATION_INFO_CLASS = 'ch.idsia.tools.EvaluationInfo'
LEVEL_CLASS = 'ch.idsia.mario.engine.level.Level'
LEVEL_PARSER_CLASS = 'ch.idsia.mario.engine.level.LevelParser'
SIMULATION_HANDLER_CLASS = 'edu.unc.cs.smbpcg.simulator.SimulationHandler'

# --- Java class definitions ---

# Standard Library classes
_J_ArrayList = jnius.autoclass('java.util.ArrayList')
_J_Integer = jnius.autoclass('java.lang.Integer')

# Project classes
_J_EvaluationInfo = jnius.autoclass(EVALUATION_INFO_CLASS)
_J_Level = jnius.autoclass(LEVEL_CLASS)
_J_LevelParser = jnius.autoclass(LEVEL_PARSER_CLASS)
_J_SimulationHandler = jnius.autoclass(SIMULATION_HANDLER_CLASS)


def play_1_1():
    level = _J_LevelParser.createLevelASCII(config_mgr.get_absolute_path('simulator/mario-1-1.txt'))
    handler = _J_SimulationHandler(level)
    handler.init()
    handler.invoke()


def proxy_test():
    return


java_level_cache = {}


def _get_java_level(level):
    if level in java_level_cache:
        return java_level_cache[level].copy()

    # j_level_arraylist is a Java ArrayList of rows in the level
    j_level_list = _J_ArrayList(level.height)
    for y in range(level.height):
        # Each row is a List of Integers representing tiles
        j_row_list = _J_ArrayList(level.width)
        for x in range(level.width):
            # Second argument expects java Object but python int converts to
            # java primitive int, so we must wrap in Integer
            j_row_list.add(_J_Integer(level.get_tile_int(x, y)))
        j_level_list.add(j_row_list)

    res = _J_LevelParser.createLevelJson(j_level_list)
    java_level_cache[level] = res
    return res.copy()


def _instantiated_simulation_handler(level, agent, visualize):
    if level is None:
        j_sim_handler = _J_SimulationHandler()
    else:
        j_sim_handler = _J_SimulationHandler(_get_java_level(level))

    j_sim_handler.setAgent(agent)
    j_sim_handler.setVisualization(visualize)
    j_sim_handler.setMaxFPS(not visualize)

    j_sim_handler.init()
    return j_sim_handler


class _EvaluationInfoProxy(object):
    def __init__(self, instance=None):
        self.__instance = instance

    def __getattr__(self, name):
        try:
            return getattr(self.__instance, name)
        except AttributeError:
            raise AttributeError(
                "Java EvaluationInfo class does not have attribute '" + name + "'."
            )

    def __str__(self):
        if self.has_instance():
            return self.__instance.toString()
        return "Unbound _EvaluationInfoProxy (no associated instance of EvaluationInfo).\n"

    def has_instance(self):
        return self.__instance is not None

    def set_instance(self, instance):
        self.__instance = instance

    def level_passed(self):
        return self.lengthOfLevelPassedPhys == constants.LEVEL_LENGTH

class SimulationProxy(object):
    @staticmethod
    def from_json_file(json_fname, human_tested=False):
        level = common.level.load_level_from_json(json_fname)
        agent = common.agents.create_astar_agent() if not human_tested else \
            common.agents.create_human_agent()
        return SimulationProxy(level, agent)

    def __init__(self, level=None, agent=None, visualize=None):
        self.__level = level
        self.__agent = agent if agent is not None else common.agents.create_astar_agent()
        self.__visualize = visualize if visualize is not None else \
            True if common.agents.is_human(agent) else False

        # Private instance variable for java EvaluationInfo object returned from simulation
        self.__eval_info_proxy = _EvaluationInfoProxy()
        # Instantiate SimulationHandler with given parameters
        self.__j_sim_handler = _instantiated_simulation_handler(self.__level, self.__agent, self.__visualize)

    # Forward eval_info requests to EvalutionInfoProxy after checking for simulation completion.
    # This additional check allows for a custom error message and potential additional
    # behaviors in the future
    def __getattr__(self, name):
        if name == 'eval_info':
            self.__check_completion()
            return self.__eval_info_proxy

    def set_level(self, level):
        self.__level = level
        self.__j_sim_handler.setLevel(_get_java_level(level))

    def set_visualize(self, visualize):
        self.__j_sim_handler = _instantiated_simulation_handler(self.__level,
                                                                self.__agent,
                                                                visualize)

    def invoke(self):
        self.__eval_info_proxy.set_instance(self.__j_sim_handler.invoke())

    def __check_completion(self):
        if not self.__eval_info_proxy.has_instance:
            raise RuntimeError(("Simulation must be run at least once before evaluation "
                                "information can be retrieved"))

    '''

    def get_distance_passed(self):
        self.__check_completion()
        return self.__j_eval_info.computeDistancePassed()

    def get_jump_cnt(self):
        self.__check_completion()
        return self.__j_eval_info.jumpActionsPerformed

    def get_trivial_jump_cnt(self):
        self.__check_completion()
        return self.__j_eval_info.trivialJumpActionsPerformed

    def get_easy_jump_cnt(self):
        self.__check_completion()
        return self.__j_eval_info.easyJumpActionsPerformed

    def get_medium_jump_cnt(self):
        self.__check_completion()
        return self.__j_eval_info.mediumJumpActionsPerformed

    def get_hard_jump_cnt(self):
        self.__check_completion()
        return self.__j_eval_info.hardJumpActionsPerformed

    '''


if __name__ == '__main__':
    play_1_1()
