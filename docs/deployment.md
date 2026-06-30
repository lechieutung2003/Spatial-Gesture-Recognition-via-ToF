# TFLite Micro Deployment

The Keras model is converted into a `.tflite` FlatBuffer representation.

## Export Process
- Run `python scripts/export_tflite.py` to generate `float32`, `float16`, and `int8` versions of the model.
- Note: Pure INT8 quantization for GRU models often fails because Keras translates RNNs using `TensorList` ops, which require TensorFlow Select Ops (not available in TFLite Micro). For this project, we rely on the `float32` version.

## C++ Header
- Run `python scripts/export_cc.py` to convert the `float32` model into a C-style array (`gesture_model_data.h`).
- This header can be directly included in your embedded C++ project.

## Inference API
- `firmware/include/inference.h` provides an integration interface:
  - `init_inference()`: Initializes the TFLM interpreter, allocates the tensor arena, and loads the model.
  - `predict_gesture()`: Takes a flat float array of length `20 * 8 * 8` and returns the predicted class index.
