import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.gesture_model import create_lightweight_model

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config["dataset"]["processed_dir"]
    classes = config["classes"]
    seq_len = config["dataset"]["sequence_length"]
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    learning_rate = config["model"]["learning_rate"]
    
    # Load data
    X = np.load(os.path.join(processed_dir, "X_processed.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))
    
    print(f"Loaded {len(X)} samples. Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for c, count in zip(unique, counts):
        print(f"  {classes[c]}: {count}")
        
    # Split dataset (Recording/session level simulated by splitting whole sequences)
    # Stratified split to ensure balanced classes
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create model
    model = create_lightweight_model(sequence_length=seq_len, num_classes=len(classes))
    model.summary()
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/best_model.keras", 
                                                    save_best_only=True, 
                                                    monitor='val_accuracy', 
                                                    mode='max')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Evaluate
    print("Evaluating on test set...")
    model = tf.keras.models.load_model("models/best_model.keras")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    np.save("models/confusion_matrix.npy", cm)
    print("Saved confusion matrix to models/confusion_matrix.npy")
    
if __name__ == "__main__":
    main()
