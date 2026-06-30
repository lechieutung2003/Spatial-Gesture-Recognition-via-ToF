# Benchmarking

Due to the unavailability of the target embedded hardware, on-device latency and tensor arena sizes were not physically measured. However, local profiling gives us key insights.

## Model Sizes
- **Float32 TFLite Model**: ~43 KB
- **Float16 TFLite Model**: ~23 KB

## Accuracy (Synthetic Data)
- **Train/Val/Test**: 100% Accuracy (due to the deterministic nature of the mock dataset).
- **Precision/Recall**: 1.00 for all classes.

## Limitations
- **Int8 Quantization**: GRU unrolling limitation in TF Lite Micro prevents pure Int8 execution without custom ops.
- **Latency**: Embedded latency is not measured because the target hardware is unavailable.
