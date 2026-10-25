#ifndef INFERENCE_H
#define INFERENCE_H

#include <cstdint>
#include <cstddef>

#define SEQUENCE_LENGTH 20
#define SENSOR_WIDTH 8
#define SENSOR_HEIGHT 8
#define SENSOR_FRAME_SIZE (SENSOR_WIDTH * SENSOR_HEIGHT)
#define NUM_CLASSES 7

// Gesture class enumeration
enum GestureClass {
    GESTURE_NONE = 0,
    GESTURE_LEFT = 1,
    GESTURE_RIGHT = 2,
    GESTURE_UP = 3,
    GESTURE_DOWN = 4,
    GESTURE_PUSH = 5,
    GESTURE_PULL = 6
};

// Pipeline performance profile metrics
struct PipelineProfile {
    float preprocess_time_ms;
    float inference_time_ms;
    float total_pipeline_time_ms;
    size_t arena_used_bytes;
};

// API Functions
bool init_inference();
bool push_sensor_frame(const uint16_t depth_frame[SENSOR_HEIGHT][SENSOR_WIDTH]);
int predict_gesture(const float* sequence_data, int length, float* confidence);
int predict_current_window(float* confidence, PipelineProfile* profile);
const char* gesture_to_string(int gesture_id);

#endif // INFERENCE_H
