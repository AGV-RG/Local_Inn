import json
import numpy as np

class ConfigJSON():
    """
    Helper to save and load configuration dictionaries to JSON files.
    Used in train.py to save normalization parameters.
    """
    def __init__(self) -> None:
        self.d = {}
    
    def load_file(self, filename):
        with open(filename, 'r') as f:
            self.d = json.load(f)
    
    def save_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.d, f, ensure_ascii=False, indent=4)
            
            
class DataProcessor():
    """
    Helper for normalizing robot poses and LiDAR data.
    Used in train.py for data pre-processing.
    """
    def __init__(self) -> None:
        pass
    
    def two_pi_warp(self, angles):
        """
        Wraps angles to the range [0, 2*pi).
        Useful for fixing theta wrap-around issues (e.g. -0.1 -> 6.18).
        """
        twp_pi = 2 * np.pi
        return (angles + twp_pi) % (twp_pi)
    
    def data_normalize(self, data):
        """
        Min-Max normalization.
        Returns: (normalized_data, [max_val, min_val])
        """
        data_min = np.min(data)
        data = data - data_min
        data_max = np.max(data)
        data = data / data_max
        return data, [data_max, data_min]
    
    def runtime_normalize(self, data, params):
        """
        Normalizes data using pre-calculated [max, min] params.
        Used for LiDAR scans where we set fixed ranges (0 to 30m).
        """
        return (data - params[1]) / params[0]
    
    def de_normalize(self, data, params):
        """
        Converts normalized data back to real-world units.
        Used during evaluation to calculate errors in meters/radians.
        """
        return data * params[0] + params[1]