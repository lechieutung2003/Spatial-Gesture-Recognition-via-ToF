import abc
import numpy as np
import time

class DepthSensor(abc.ABC):
    """Abstract Hardware Abstraction Layer for 8x8 ToF Sensors."""
    
    @abc.abstractmethod
    def begin(self) -> bool:
        """Initialize the sensor."""
        pass
    
    @abc.abstractmethod
    def read_frame(self) -> np.ndarray:
        """
        Read a single frame from the sensor.
        Returns:
            np.ndarray: An 8x8 numpy array of depth values (in mm).
                        Should return None or raise Exception on failure.
        """
        pass

class MockDepthSensor(DepthSensor):
    """Mock sensor that generates random/synthetic depth frames."""
    
    def __init__(self, resolution=(8, 8), fps=15):
        self.resolution = resolution
        self.fps = fps
        self.is_initialized = False
        
    def begin(self) -> bool:
        self.is_initialized = True
        return True
        
    def read_frame(self) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Sensor not initialized. Call begin() first.")
        
        # Simulate sensor delay (can be commented out to speed up data generation if needed)
        # time.sleep(1.0 / self.fps)
        
        # Generate a baseline flat surface around 1000mm with some noise
        base_depth = 1000
        noise = np.random.normal(0, 10, self.resolution)
        frame = np.clip(base_depth + noise, 0, 4000) # VL53L5CX typical max range 4000mm
        return frame.astype(np.float32)

class MockGestureSensor(MockDepthSensor):
    """Mock sensor that can generate synthetic gesture patterns over time."""
    
    def __init__(self, resolution=(8, 8), fps=15):
        super().__init__(resolution, fps)
        self.current_gesture = "NONE"
        self.gesture_frame_idx = 0
        self.sequence_length = 20
        
    def set_gesture(self, gesture: str, sequence_length: int = 20):
        self.current_gesture = gesture
        self.sequence_length = sequence_length
        self.gesture_frame_idx = 0
        
    def read_frame(self) -> np.ndarray:
        frame = super().read_frame()
        
        if self.current_gesture == "NONE":
            return frame
            
        # Add synthetic object (closer distance)
        progress = self.gesture_frame_idx / max(1, self.sequence_length - 1)
        
        obj_depth = 300 # Object is at 300mm
        
        row, col = 3, 3 # Default center
        
        if self.current_gesture == "LEFT":
            col = int(progress * 7)
        elif self.current_gesture == "RIGHT":
            col = 7 - int(progress * 7)
        elif self.current_gesture == "UP":
            row = int(progress * 7)
        elif self.current_gesture == "DOWN":
            row = 7 - int(progress * 7)
        elif self.current_gesture == "PUSH":
            obj_depth = 600 - int(progress * 400) # Moves from 600 to 200
        elif self.current_gesture == "PULL":
            obj_depth = 200 + int(progress * 400) # Moves from 200 to 600
            
        # Draw a 2x2 object
        r_start = max(0, row - 1)
        r_end = min(8, row + 2)
        c_start = max(0, col - 1)
        c_end = min(8, col + 2)
        
        frame[r_start:r_end, c_start:c_end] = obj_depth
        
        self.gesture_frame_idx += 1
        if self.gesture_frame_idx >= self.sequence_length:
            self.current_gesture = "NONE" # reset
            
        return frame

class VL53L5CXSerialSensor(DepthSensor):
    """
    Real sensor implementation that reads VL53L5CX data over a Serial port.
    Assumes an MCU (like Arduino/ESP32) is reading the sensor via I2C and
    streaming the 8x8 depth array over UART as comma-separated values or binary.
    """
    
    def __init__(self, port="COM3", baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.is_initialized = False
        
    def begin(self) -> bool:
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.is_initialized = True
            print(f"Connected to VL53L5CX on {self.port} at {self.baudrate} baud.")
            return True
        except ImportError:
            print("Error: pyserial is not installed. Run 'pip install pyserial'")
            return False
        except Exception as e:
            print(f"Failed to connect to sensor on {self.port}: {e}")
            return False
            
    def read_frame(self) -> np.ndarray:
        if not self.is_initialized:
            raise RuntimeError("Sensor not initialized. Call begin() first.")
            
        try:
            # Read a line from serial (expecting 64 comma-separated integer values)
            line = self.serial.readline().decode('utf-8').strip()
            
            if not line:
                return None
                
            # Parse the CSV string into an 8x8 numpy array
            values = [float(x) for x in line.split(',') if x]
            if len(values) == 64:
                frame = np.array(values, dtype=np.float32).reshape((8, 8))
                return frame
            else:
                # Malformed frame
                return None
                
        except Exception as e:
            print(f"Error reading from serial sensor: {e}")
            return None
