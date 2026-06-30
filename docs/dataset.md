# Dataset Format

The dataset represents spatio-temporal depth data.

## Structure
- `X_raw.npy`: Contains raw depth frames of shape `(N, T, 8, 8)` in millimeters.
- `X_processed.npy`: Contains normalized frames (in range [0, 1]).
- `y.npy`: Contains integer labels corresponding to classes.

## Configurable Gestures
The gestures are defined in `configs/config.yaml`. Default gestures include:
- NONE
- LEFT
- RIGHT
- UP
- DOWN
- PUSH
- PULL

## Simulation
Due to the absence of the physical sensor, data is simulated via `MockGestureSensor` which places an artificial object in the 8x8 grid and moves it across the field of view over the 20-frame sequence.
