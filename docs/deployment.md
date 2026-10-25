# TFLite Micro Deployment

The Keras model is converted into optimized `.tflite` FlatBuffer representations and embedded directly into C++ firmware headers.

---

## Model Conversion & Quantization

Run the export script to generate all model variants:
```bash
python scripts/export_tflite.py
```

### Unrolled GRU for Pure TFLM Compatibility
Standard Keras RNNs often compile to dynamic `TensorList` control-flow operations that necessitate the TensorFlow Select (Flex) delegate—unsupported in bare-metal TensorFlow Lite Micro. 

By unrolling the temporal recurrence (`unroll=True`) across the fixed 20-frame window, the model generates a static execution graph containing only standard built-in TFLite kernels (`Conv2D`, `MaxPool2D`, `FullyConnected`, `Logistic`, `Tanh`, `Add`, `Mul`). This enables:
- **Zero Flex Op Dependencies**: 100% compliant with standard TFLite Micro runtime.
- **Quantization Support**: Direct post-training quantization to `Float16` and full `Int8`.

---

## C++ Firmware Integration

### 1. Generate C Array
```bash
python scripts/export_cc.py
```
This generates `firmware/include/gesture_model_data.h`, exposing `g_gesture_model_data` aligned in read-only Flash memory.

### 2. C++ Inference API (`firmware/include/inference.h`)
- `init_inference()`: Initializes the static tensor arena (48 KB) and loads the TFLite Micro model.
- `push_sensor_frame(depth_frame)`: Ingests an incoming 8×8 depth frame into the ring buffer and normalizes distance values. Returns `true` when a full 20-frame sequence is ready.
- `predict_gesture(sequence_data, length, &confidence)`: Directly runs inference on a normalized sequence.
- `predict_current_window(&confidence, &profile)`: Chronologically aligns the circular sliding window and returns the predicted gesture class with runtime latency profiling.
