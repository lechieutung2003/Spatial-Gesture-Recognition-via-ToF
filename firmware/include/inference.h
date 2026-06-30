#ifndef INFERENCE_H
#define INFERENCE_H

#include <cstdint>

// Predict gesture from a (20, 8, 8) sequence.
// Sequence should be preprocessed: normalized [0, 1] floats.
// If using int8 quantization, caller must map floats to int8, 
// or this function should handle the quantization based on input.
// We assume the model expects float32 for simplicity, unless compiled for INT8.

#define SEQUENCE_LENGTH 20
#define SENSOR_WIDTH 8
#define SENSOR_HEIGHT 8

bool init_inference();
int predict_gesture(const float* sequence_data, int length, float* confidence);

#endif
