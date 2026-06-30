import tensorflow as tf
from tensorflow.keras import layers, models

def create_lightweight_model(sequence_length=20, num_classes=7):
    """
    Creates a lightweight Spatio-Temporal model suitable for TinyML.
    
    Architecture:
    Input (T, 8, 8) 
    -> Reshape to (T, 8, 8, 1)
    -> TimeDistributed(Conv2D) to extract spatial features from each frame
    -> TimeDistributed(MaxPooling2D) or TimeDistributed(Flatten)
    -> GRU (Temporal modeling)
    -> Dense (Classification)
    
    This keeps parameters low and is compatible with TFLite Micro.
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
    # A small GRU is sufficient for 20-frame sequences
    x = layers.GRU(32, return_sequences=False)(x)
    
    # Classification
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
