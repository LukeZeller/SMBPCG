from evolution.level_difficulty.feasible_shifts import number_of_shifts_and_jumps
from common import constants
from common.constants import DEBUG_PRINT

def calculate_difficulty_for_failure(info):
    fraction_of_level_completed = float(info.lengthOfLevelPassedPhys) / constants.LEVEL_LENGTH
    return 1 - fraction_of_level_completed

def calculate_difficulty_for_success(info, level):
    num_shifts, num_jumps = number_of_shifts_and_jumps(info, level)
    if DEBUG_PRINT:
        print("Shifts: ", num_shifts)
        print("Num jumps: ", num_jumps)
    if num_jumps == 0:
        average_number_of_shifts_per_jump = float('inf')
    else:
        average_number_of_shifts_per_jump = float(num_shifts) / num_jumps
    # The more that the jumps can be shifted, the easier the level is
    return 1 / average_number_of_shifts_per_jump