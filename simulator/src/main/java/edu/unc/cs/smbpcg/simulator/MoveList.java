package edu.unc.cs.smbpcg.simulator;

import ch.idsia.mario.engine.sprites.Mario;

import java.util.ArrayList;

public class MoveList {
    private ArrayList<KeyPress> moves;

    public MoveList() {
        moves = new ArrayList<>();
    }

    public MoveList copy() {
        MoveList copied = new MoveList();
        for (KeyPress kp: moves) {
            copied.moves.add(kp.copy());
        }
        return copied;
    }

    public void addKeyPress(KeyPress pressed) {
        moves.add(new KeyPress(pressed));
    }

    public KeyPress get(int index) {
        return moves.get(index);
    }

    public void removeJumpsInRange(int start, int end) {
        for (int frame = start; frame < end; frame++){
            moves.get(frame).unsetKey(Mario.KEY_JUMP);
        }
    }

    public void addJumpsInRange(int start, int end) {
        for (int frame = start; frame < end; frame++){
            moves.get(frame).setKey(Mario.KEY_JUMP);
        }
    }

    public void shiftedJump(int oldStart, int oldEnd, int newStart, int newEnd) {
        removeJumpsInRange(oldStart, oldEnd);
        addJumpsInRange(newStart, newEnd);
    }

    public void rstrip(KeyPress toRemove) {
        while (!moves.isEmpty() && moves.get(moves.size() - 1).equals(toRemove)) {
            moves.remove(moves.size() - 1);
        }
    }

    public int size() {
        return moves.size();
    }

    @Override
    public String toString() {
        String result = "";
        for (int i = 0; i < moves.size(); i++)
        {
            KeyPress keysPressed = moves.get(i);
            result += "               ";
            result += keysPressed.isPressed(Mario.KEY_LEFT) ? "L" : " ";
            result += keysPressed.isPressed(Mario.KEY_RIGHT) ? "R" : " ";
            result += keysPressed.isPressed(Mario.KEY_DOWN) ? "D" : " ";
            result += keysPressed.isPressed(Mario.KEY_JUMP) ? "J" : " ";
            result += keysPressed.isPressed(Mario.KEY_SPEED) ? "S" : " ";
            if (i != moves.size() - 1)
                result += '\n';
        }
        return result;
    }

    public String viewJumps() {
        StringBuilder result = new StringBuilder();
        for (KeyPress kp: moves) {
            char c = kp.isPressed(Mario.KEY_JUMP) ? 'X' : '_';
            result.append(c);
        }
        return result.toString();
    }
}
