import unittest
import numpy as np
import tensorflow as tf
from src.preprocessing.preprocessor import DepthPreprocessor
from src.data.sensor import MockGestureSensor

class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DepthPreprocessor(max_distance=4000.0)
        self.sensor = MockGestureSensor(resolution=(8, 8), fps=15)
        self.sensor.begin()

    def test_preprocessor_normalization(self):
        """Verify distance normalization: 0mm -> 1.0, 4000mm -> 0.0"""
        raw_frame = np.array([[0, 2000], [4000, 5000]], dtype=np.float32)
        # Pad to 8x8
        padded = np.pad(raw_frame, ((0, 6), (0, 6)), mode='constant', constant_values=1000)
        processed = self.preprocessor.process_frame(padded)
        
        # 0 is invalid -> mapped to max_distance -> 0.0
        self.assertAlmostEqual(processed[0, 0], 0.0, places=3)
        # 2000mm -> 1 - (2000/4000) = 0.5
        self.assertAlmostEqual(processed[0, 1], 0.5, places=3)
        # 4000mm -> 1 - (4000/4000) = 0.0
        self.assertAlmostEqual(processed[1, 0], 0.0, places=3)
        # 5000mm clipped to 4000 -> 0.0
        self.assertAlmostEqual(processed[1, 1], 0.0, places=3)

    def test_mock_sensor_gesture_generation(self):
        """Verify gesture frame generation produces expected sequence length."""
        self.sensor.set_gesture("RIGHT", sequence_length=20)
        frames = []
        for _ in range(20):
            frame = self.sensor.read_frame()
            frames.append(frame)
        self.assertEqual(len(frames), 20)
        sequence = np.array(frames)
        self.assertEqual(sequence.shape, (20, 8, 8))

    def test_tflite_model_inference(self):
        """Verify TFLite model runs inference and returns valid softmax distribution."""
        model_path = "models/gesture_model_f32.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        dummy_input = np.random.rand(1, 20, 8, 8).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(output.shape, (1, 7))
        # Sum of softmax probabilities should be ~1.0
        self.assertAlmostEqual(float(np.sum(output)), 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
