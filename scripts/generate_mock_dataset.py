import os
import sys
import yaml
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.sensor import MockGestureSensor
from src.preprocessing.preprocessor import DepthPreprocessor

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    classes = config["classes"]
    seq_len = config["dataset"]["sequence_length"]
    raw_dir = config["dataset"]["raw_dir"]
    processed_dir = config["dataset"]["processed_dir"]
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    sensor = MockGestureSensor(fps=15)
    sensor.begin()
    
    preprocessor = DepthPreprocessor()
    
    samples_per_class = 150 # Total samples per gesture class
    
    print("Generating mock dataset...")
    
    all_raw_data = []
    all_processed_data = []
    all_labels = []
    
    for class_idx, label in enumerate(classes):
        print(f"Generating {samples_per_class} samples for {label}...")
        for sample_idx in range(samples_per_class):
            sensor.set_gesture(label, seq_len)
            
            sequence = []
            for _ in range(seq_len):
                frame = sensor.read_frame()
                sequence.append(frame)
                
            sequence = np.array(sequence) # (T, 8, 8)
            processed_seq = preprocessor.process_sequence(sequence)
            
            all_raw_data.append(sequence)
            all_processed_data.append(processed_seq)
            all_labels.append(class_idx)
            
    # Save datasets
    X_raw = np.array(all_raw_data)
    X_processed = np.array(all_processed_data)
    y = np.array(all_labels)
    
    np.save(os.path.join(raw_dir, "X_raw.npy"), X_raw)
    np.save(os.path.join(raw_dir, "y.npy"), y)
    
    np.save(os.path.join(processed_dir, "X_processed.npy"), X_processed)
    np.save(os.path.join(processed_dir, "y.npy"), y)
    
    print(f"Dataset generated.")
    print(f"Raw shape: {X_raw.shape}, Processed shape: {X_processed.shape}, Labels shape: {y.shape}")

if __name__ == "__main__":
    main()
