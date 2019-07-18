from common.constants import KEY_JUMP, DEBUG_PRINT
from common.check_moves import can_complete_with_moves
from functools import partial
import time


def number_of_shifts_and_jumps(info, level):
    moves = info.marioMoves
    jump_starts, jump_ends = _get_jumps(moves)
    if DEBUG_PRINT:
        print("Before removing redundant")
        print(jump_starts, jump_ends)
    moves, jump_starts, jump_ends = _removed_redundant_jumps(level, 
                                                             moves, 
                                                             jump_starts,
                                                             jump_ends)
    if DEBUG_PRINT:
        print("After removing redundant")
        print(jump_starts, jump_ends)
    return _calculate_number_of_shifts_and_jumps(level, moves, jump_starts, jump_ends)


def _get_jumps(moves):
    jump_starts = []
    jump_ends = []
    jump_previously_pressed = False
    if DEBUG_PRINT:
        print("Moves size: ", moves.size())
    for frame in range(moves.size()):
        keys_pressed = moves.get(frame)
        jump_currently_pressed = keys_pressed.isPressed(KEY_JUMP)
        if jump_previously_pressed != jump_currently_pressed:
            if jump_currently_pressed:
                jump_starts.append(frame)
            else:
                jump_ends.append(frame)
        jump_previously_pressed = jump_currently_pressed
    if len(jump_ends) < len(jump_starts):
        jump_ends.append(moves.size())
    assert len(jump_starts) == len(jump_ends)
    return jump_starts, jump_ends


def _removed_redundant_jumps(level, moves, jump_starts, jump_ends):
    redundant_jump_indices = set()
    for i, (jump_start, jump_end) in enumerate(zip(jump_starts, jump_ends)):
        new_moves = moves.copy()
        new_moves.removeJumpsInRange(jump_start, jump_end)
        if can_complete_with_moves(level, new_moves, False):
            redundant_jump_indices.add(i)
            moves = new_moves.copy()
    jump_starts = [jump
                   for index, jump in enumerate(jump_starts)
                   if index not in redundant_jump_indices]
    jump_ends = [jump
                 for index, jump in enumerate(jump_ends)
                 if index not in redundant_jump_indices]
    return moves, jump_starts, jump_ends


def _calculate_number_of_shifts_and_jumps(level, moves, jump_starts, jump_ends):
    start_time = time.perf_counter()

    num_jumps = len(jump_starts)
    jump_index_to_num_shifts = partial(_number_of_shifts_for_specific_jump,
                                       level,
                                       moves,
                                       jump_starts,
                                       jump_ends)
    num_shifts_per_jump = map(jump_index_to_num_shifts, range(num_jumps))
    num_shifts = sum(num_shifts_per_jump)

    end_time = time.perf_counter()
    if DEBUG_PRINT:
        print("Time taken (s): ", end_time - start_time)
    return num_shifts, num_jumps


def _number_of_shifts_for_specific_jump(level, moves, jump_starts, jump_ends, jump_index):
    valid_shifts_for_jump = 0

    original_start = jump_starts[jump_index]
    original_end = jump_ends[jump_index]

    end_of_previous_jump = 0 if jump_index - 1 < 0 else jump_ends[jump_index - 1]
    start_of_next_jump = moves.size() if jump_index + 1 >= len(jump_starts) else jump_starts[jump_index + 1]

    valid_shifts_for_jump = 0
    for new_start in range(end_of_previous_jump, start_of_next_jump):
        for new_end in range(new_start + 1, start_of_next_jump):
            new_moves = moves.copy()
            new_moves.shiftedJump(original_start, original_end,
                                  new_start, new_end)
            if can_complete_with_moves(level, new_moves, False):
                valid_shifts_for_jump += 1
                break
    return valid_shifts_for_jump
