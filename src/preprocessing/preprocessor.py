import numpy as np

class DepthPreprocessor:
    """Preprocesses 8x8 depth frames for model input."""
    
    def __init__(self, max_distance=4000.0, invalid_value=0.0):
        self.max_distance = float(max_distance)
        self.invalid_value = float(invalid_value)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single 8x8 depth frame.
        1. Replace invalid readings (e.g. 0 or very large numbers) with max_distance
        2. Clip to max_distance
        3. Normalize to [0, 1] range (where 0 is max_distance, 1 is 0mm - i.e., closeness)
           Normalizing such that closer objects have higher values usually helps CNNs.
        """
        frame_clean = np.copy(frame)
        
        # Handle invalid values (often 0 means invalid in some ToF sensors)
        frame_clean[frame_clean == self.invalid_value] = self.max_distance
        
        # Clip
        frame_clean = np.clip(frame_clean, 0, self.max_distance)
        
        # Normalize (closer = higher value)
        normalized = 1.0 - (frame_clean / self.max_distance)
        
        return normalized.astype(np.float32)
        
    def process_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Process a sequence of shape (T, 8, 8).
        """
        processed = np.zeros_like(sequence, dtype=np.float32)
        for i in range(sequence.shape[0]):
            processed[i] = self.process_frame(sequence[i])
        return processed
