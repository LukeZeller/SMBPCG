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
    """ ArrayList<Integer> jumpStarts = new ArrayList<>();
        ArrayList<Integer> jumpEnds = new ArrayList<>();

        boolean jumpAlreadyPressed = false;
        for (int frame = 0; frame < moves.size(); frame++) {
            boolean jumpCurrentlyPressed = moves.get(frame)[Mario.KEY_JUMP];
            if (!jumpAlreadyPressed && jumpCurrentlyPressed) {
                jumpStarts.add(frame);
            }
            if (jumpAlreadyPressed && !jumpCurrentlyPressed) {
                jumpEnds.add(frame);
            }
            jumpAlreadyPressed = jumpCurrentlyPressed;
        }
        if (jumpStarts.size() > 0 && jumpEnds.size() < jumpStarts.size())
        {
            /*
             * If there is no release of the jump key before the level is complete, add a sentinel jump
             * release 1 frame after the last jump was started
             */
            int lastJumpStart = jumpStarts.get(jumpStarts.size() - 1);
            jumpEnds.add(lastJumpStart + 1);
        }
        if (jumpStarts.size() != jumpEnds.size()) {
            throw new RuntimeException("Mismatch between number of jump key presses and releases");
        }

        return new Pair<>(jumpStarts, jumpEnds);"""
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
        