# Model Architecture

To achieve a balance between spatial recognition (recognizing the location of the hand) and temporal recognition (understanding the movement over time), a hybrid CNN-GRU architecture is used.

## Architecture Pipeline
- **Input**: `(20, 8, 8)` sequence.
- **Reshape**: `(20, 8, 8, 1)` for channel dimension.
- **TimeDistributed CNN 1**: 8 filters, `3x3` -> Extracts initial spatial features.
- **TimeDistributed MaxPooling**: `2x2` -> Reduces resolution to `4x4`.
- **TimeDistributed CNN 2**: 16 filters, `3x3`.
- **TimeDistributed MaxPooling**: `2x2` -> Reduces resolution to `2x2`.
- **Flatten**
- **GRU**: 32 units -> Models the temporal sequence across 20 frames.
- **Dense**: 16 units with ReLU.
- **Dense**: 7 units with Softmax (Classification).

## Size and Constraints
- The model is extremely lightweight, containing fewer than 15,000 parameters.
- Designed specifically to map well to TFLite and deployable with low memory overhead.
