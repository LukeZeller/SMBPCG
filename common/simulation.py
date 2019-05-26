from config import config_mgr

config_mgr.setup_environment()

import jnius

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

class SimulationProxy(object):
    # Setting testing_mode = True invokes simulation with human player instead of agent
    def __init__(self, level = None, testing_mode = False):
        # Private instance variable for java EvaluationInfo object returned from simulation 
        self.__j_eval_info = None

        # Instantiate SimulationHandler (with level if present)
        if level is None:
            self.j_sim_handler = _J_SimulationHandler()
        else:
            self.j_sim_handler = _J_SimulationHandler(_get_java_level(level))
        # Indicate that agent - not human player - should be used, and set default
        # EvaluationOptions for agent (run simulation at max FPS w/o visualization)
        self.j_sim_handler.setHumanPlayer(testing_mode, True)
        self.j_sim_handler.init()

    def set_level(level):
        self.j_sim_handler.setLevel(_get_java_level(level))

    def invoke(self):
        self.__j_eval_info = self.j_sim_handler.invoke()


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

    def __check_completion(self):
        if self.__j_eval_info is None:
            raise RuntimeError( ("Simulation must be run at least once before evaluation "
                                 "information can be retrieved")
            )

if __name__ == '__main__':
    play_1_1()
