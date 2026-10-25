import tensorflow as tf
from tensorflow.keras import layers, models

def create_lightweight_model(sequence_length=20, num_classes=7, unroll=True):
    """
    Creates a lightweight Spatio-Temporal model suitable for TinyML.
    
    Architecture:
    Input (T, 8, 8) 
    -> Reshape to (T, 8, 8, 1)
    -> TimeDistributed(Conv2D) to extract spatial features from each frame
    -> TimeDistributed(MaxPooling2D) or TimeDistributed(Flatten)
    -> GRU (Temporal modeling, unrolled for pure TFLite Micro compatibility)
    -> Dense (Classification)
    
    This keeps parameters low and is fully compatible with TFLite Micro without Select TF Ops.
    """
    inputs = layers.Input(shape=(sequence_length, 8, 8))
    
    # Add channel dimension
    x = layers.Reshape((sequence_length, 8, 8, 1))(inputs)
    
    # Spatial feature extraction per frame
    # We use a very small CNN because the resolution is only 8x8
    x = layers.TimeDistributed(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x) # Now (T, 4, 4, 8)
    
    x = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x) # Now (T, 2, 2, 16)
    
    x = layers.TimeDistributed(layers.Flatten())(x) # Now (T, 64)
    
    # Temporal modeling
    # An unrolled GRU avoids Flex/TensorList ops in TFLite Micro
    x = layers.GRU(32, return_sequences=False, unroll=unroll)(x)
    
    # Classification
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
