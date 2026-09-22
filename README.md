# Train a World Model From Scratch

Run the interactive pendulum demo from the project root:

```bash
python src/envs/demo.py --env balance
```

Hold **A/D** or the **arrow keys** to push the cart, and release to stop applying
force. Keyboard pushes use the full 12 N so the cart can catch a falling pole.
Push toward the direction the pole is leaning, and release or reverse to avoid
overshooting.
The cart coasts and the pole swings freely under the original physics; balancing
is entirely manual. Press **R** to reset, **Tab** to switch to the car, or
**Q/Esc** to quit.

The pendulum demo runs at **half speed** to give you more time to react to the
falling pole. Drawing and keyboard input remain responsive; the simulation's
physics and training time step are unchanged. The car demo runs at normal speed.

These demos are interactive previews and do not save trajectories.
