# Spatial Gesture Recognition via Time-of-Flight (ToF)

A TinyML project for recognizing touchless hand gestures using an 8x8 ToF sensor (e.g., VL53L5CX). 

## 🚀 Project Overview
This project provides a complete edge AI pipeline:
1. Spatio-temporal dataset generation (Mock HAL).
2. Lightweight CNN-GRU model training.
3. TensorFlow Lite export.
4. C++ TFLite Micro interface.

## 🛠️ Features
- **Privacy-Compliant**: Relies entirely on 8x8 depth maps; no RGB cameras or PII.
- **Hardware Abstraction**: `MockGestureSensor` allows pure software testing without physical hardware.
- **Lightweight Model**: Designed specifically for low RAM/Flash footprint on MCUs.

## 📂 Architecture & Documentation
For detailed architecture, model, and benchmarking notes, please read the documentation in the `docs/` folder:
- [System Architecture](docs/architecture.md)
- [Dataset](docs/dataset.md)
- [Model Architecture](docs/model.md)
- [Deployment](docs/deployment.md)
- [Benchmarking](docs/benchmarking.md)

## 🔧 Installation & Reproducibility

**1. Clone the repository and setup environment:**
```bash
python -m venv .venv
# Activate venv based on your OS
pip install -r requirements.txt
```

**2. Generate Mock Dataset:**
```bash
python scripts/generate_mock_dataset.py
```
*Generates synthetic 20-frame gesture sequences.*

**3. Train the Model:**
```bash
python scripts/train.py
```
*Trains the CNN-GRU model and saves the best version to `models/best_model.keras`.*

**4. Export to TFLite:**
```bash
python scripts/export_tflite.py
```
*Creates float32 and float16 TFLite models. (Note: Int8 is attempted but currently limited by Keras GRU unrolling support in TFLite).*

**5. Generate C++ Header for TFLM:**
```bash
python scripts/export_cc.py
```
*Converts the `.tflite` model to a C-byte array in `firmware/include/gesture_model_data.h`.*

## ⚠️ Current Status & Limitations
- **Hardware Integration Status**: Simulated via HAL. Physical VL53L5CX not connected.
- **Embedded Testing Status**: The C++ inference framework is written, but on-device execution is pending a target compiler (e.g., ESP-IDF, STM32Cube) and physical board.
- **Latency Measurements**: Embedded latency not measured because target hardware is unavailable.
