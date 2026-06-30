#ifndef SENSOR_VL53L5CX_H
#define SENSOR_VL53L5CX_H

#include <cstdint>

class DepthSensor {
public:
    virtual bool begin() = 0;
    virtual bool readFrame(uint16_t depth[8][8]) = 0;
    virtual ~DepthSensor() = default;
};

// Real implementation using the ST VL53L5CX Ultra Lite Driver (ULD)
class VL53L5CXSensor : public DepthSensor {
public:
    VL53L5CXSensor(int sda_pin, int scl_pin);
    virtual ~VL53L5CXSensor();

    bool begin() override;
    bool readFrame(uint16_t depth[8][8]) override;

private:
    int _sda;
    int _scl;
    bool _is_initialized;
    // In a real project, we would hold a reference to the ULD configuration struct:
    // VL53L5CX_Configuration dev;
};

#endif
