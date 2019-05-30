from common.constants import KEY_JUMP
from evolution.check_moves import can_complete

def number_of_feasible_shifts(info, level):
    moves = info.marioMoves
    jump_starts, jump_ends = _get_jumps(moves)
    print("Jump starts: ", jump_starts)
    print("Jump ends: ", jump_ends)
    _remove_redundant_jumps(moves, jump_starts, jump_ends)
    return 1

def _get_jumps(moves):
    jump_starts = []
    jump_ends = []
    jump_previously_pressed = False
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
        jump_ends.append(jump_starts[-1] + 1)
    assert len(jump_starts) == len(jump_ends)
    return jump_starts, jump_ends

def _remove_redundant_jumps(moves, jump_starts, jump_ends):
    pass
        