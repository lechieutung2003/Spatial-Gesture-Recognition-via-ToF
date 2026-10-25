import os
import sys
import numpy as np
import tensorflow as tf

def representative_dataset():
    # Load representative data for int8 quantization calibration
    # We use a small subset of the training data
    X = np.load("data/processed/X_processed.npy")
    # Take 100 random samples
    indices = np.random.choice(len(X), 100, replace=False)
    for i in indices:
        # yield input as float32
        yield [X[i:i+1].astype(np.float32)]

def main():
    model_path = "models/best_model.keras"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 1. Standard Float32 TFLite (Pure TFLITE_BUILTINS, zero SELECT_TF_OPS)
    print("Exporting Float32 TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_model_f32 = converter.convert()
    with open("models/gesture_model_f32.tflite", "wb") as f:
        f.write(tflite_model_f32)
        
    # 2. Float16 Quantization
    print("Exporting Float16 TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    tflite_model_f16 = converter.convert()
    with open("models/gesture_model_f16.tflite", "wb") as f:
        f.write(tflite_model_f16)
        
    # 3. Int8 Dynamic Range & Weight Quantization
    print("Exporting Int8 TFLite model...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model_int8 = converter.convert()
        with open("models/gesture_model_int8.tflite", "wb") as f:
            f.write(tflite_model_int8)
        print("Int8 export successful.")
    except Exception as e:
        print("Int8 export failed:", e)
            
    # Print sizes
    print("\nModel Sizes:")
    f32_size = os.path.getsize("models/gesture_model_f32.tflite") / 1024
    f16_size = os.path.getsize("models/gesture_model_f16.tflite") / 1024
    print(f"Float32: {f32_size:.2f} KB")
    print(f"Float16: {f16_size:.2f} KB")
    if os.path.exists("models/gesture_model_int8.tflite"):
        int8_size = os.path.getsize("models/gesture_model_int8.tflite") / 1024
        print(f"Int8 (Full): {int8_size:.2f} KB")
    elif os.path.exists("models/gesture_model_int8_fallback.tflite"):
        int8_size = os.path.getsize("models/gesture_model_int8_fallback.tflite") / 1024
        print(f"Int8 (Fallback): {int8_size:.2f} KB")
        
if __name__ == "__main__":
    main()
