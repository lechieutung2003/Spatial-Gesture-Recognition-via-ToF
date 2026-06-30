# System Architecture

The Spatial Gesture Recognition system is designed for TinyML deployment on embedded systems (e.g., ESP32, STM32) using an 8x8 Time-of-Flight sensor.

## Data Flow Pipeline

1. **VL53L5CX Sensor**: Captures 8x8 depth matrices at ~15Hz.
2. **Preprocessing**: Normalizes distances to [0, 1] range (where 1 is closest) and clips invalid readings.
3. **Sequence Construction**: A sliding window or buffered sequence of T=20 frames is built.
4. **Model Inference**: A lightweight CNN-GRU model processes the (20, 8, 8) input and outputs softmax probabilities across 7 classes.
5. **Post-processing**: Selects the argmax of the probabilities to determine the gesture.

## Hardware Abstraction Layer
To allow offline development and testing, a `DepthSensor` base class defines the HAL interface. `MockGestureSensor` generates synthetic datasets mimicking the sensor without hardware being attached.
