from config import config_mgr

config_mgr.setup_environment()

import jnius

from common import constants
import common.level

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
    level = _J_LevelParser.createLevelASCII(config_mgr.get_absolute_path('simulator/Mario-1-1.txt'))
    handler = _J_SimulationHandler(level)
    handler.init()
    handler.invoke()

def proxy_test():
    return

def _get_java_level(level):
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

    return _J_LevelParser.createLevelJson(j_level_list)

class _EvaluationInfoProxy(object):
    def __init__(self, instance = None):
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
    def from_json_file(json_fname, testing_mode = False):
        level = common.level.load_level_from_json(json_fname)
        return SimulationProxy(level, testing_mode)
        
    # Setting testing_mode = True invokes simulation with human player instead of agent
    def __init__(self, level = None, testing_mode = False):
        # Private instance variable for java EvaluationInfo object returned from simulation 
        self.__eval_info_proxy = _EvaluationInfoProxy()

        # Instantiate SimulationHandler (with level if present)
        if level is None:
            self.__j_sim_handler = _J_SimulationHandler()
        else:
            self.__j_sim_handler = _J_SimulationHandler(_get_java_level(level))
        # Indicate that agent - not human player - should be used, and set default
        # EvaluationOptions for agent (run simulation at max FPS w/o visualization)
        self.__j_sim_handler.setHumanPlayer(testing_mode, True)
        self.__j_sim_handler.init()

    # Forward eval_info requests to EvalutionInfoProxy after checking for simulation completion.
    # This additional check allows for a custom error message and potential additional
    # behaviors in the future
    def __getattr__(self, name):
        if name == 'eval_info':
            self.__check_completion()
            return self.__eval_info_proxy
        
    def set_level(self, level):
        self.__j_sim_handler.setLevel(_get_java_level(level))
        
    def set_visualization(self, is_visualization):
        self.__j_sim_handler.setVisualization(is_visualization)
        self.__j_sim_handler.setMaxFPS(not is_visualization)

    def invoke(self):
        self.__eval_info_proxy.set_instance(self.__j_sim_handler.invoke())
    
    def __check_completion(self):
        if not self.__eval_info_proxy.has_instance:
            raise RuntimeError( ("Simulation must be run at least once before evaluation "
                                 "information can be retrieved")
            )

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
