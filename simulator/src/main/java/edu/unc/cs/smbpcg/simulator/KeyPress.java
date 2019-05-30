package edu.unc.cs.smbpcg.simulator;

import ch.idsia.mario.environments.Environment;

import java.util.Arrays;

public class KeyPress {
    private boolean [] pressed;

    public KeyPress() {
        this.pressed = new boolean[Environment.numberOfButtons];
    }

    public KeyPress(KeyPress kp) {
        this(kp.pressed);
    }

    public KeyPress(boolean [] pressed) {
        if (pressed.length != Environment.numberOfButtons) {
            throw new RuntimeException("Invalid number of keys, does not match with environment");
        }
        if (pressed == null) {
            this.pressed = null;
        }
        else {
            this.pressed = pressed.clone();
        }
    }

    public void setKey(int key) {
        pressed[key] = true;
    }

    public void unsetKey(int key) {
        pressed[key] = false;
    }

    public boolean isPressed(int key) {
        return pressed[key];
    }

    public boolean isValid() {
        return pressed != null;
    }

    public boolean [] getPressed() {
        return pressed;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null)
            return false;
        if (this.getClass() != o.getClass())
            return false;
        KeyPress other = (KeyPress) o;
        return Arrays.equals(this.pressed, other.pressed);
    }
}
