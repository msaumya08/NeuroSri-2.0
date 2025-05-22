import numpy as np
from scipy import signal
from typing import Dict, List
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eeg_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Define constants
SAMPLING_RATE = 255  # Hz

class SignalProcessor:
    def __init__(self):
        self.sampling_rate = SAMPLING_RATE
        # Define frequency bands directly in the class
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        logger.info("SignalProcessor initialized with frequency bands")
        
    def apply_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
        """Apply bandpass filter to EEG data."""
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=0)
        
    def extract_frequency_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency band powers for each channel."""
        features = {}
        
        # Apply window function
        window = signal.windows.hann(data.shape[0])
        windowed_data = data * window[:, np.newaxis]
        
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(windowed_data, 
                                fs=self.sampling_rate,
                                nperseg=min(256, data.shape[0]),
                                axis=0)
                                
        # Extract band powers
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(psd[freq_mask], axis=0)
            features[band_name] = band_power
            
        return features
        
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess EEG data with filtering and artifact removal."""
        try:
            # Ensure data is contiguous float32 array with correct shape
            data = np.ascontiguousarray(data, dtype=np.float32)
            
            # Ensure 2D array with shape (samples, channels)
            if len(data.shape) == 1:
                data = data.reshape(-1, 2)
            elif len(data.shape) == 2 and data.shape[1] != 2:
                data = data.T
            
            logger.debug(f"Data shape after reshaping: {data.shape}")
            
            # Remove DC offset
            data = data - np.mean(data, axis=0)
            
            # Apply notch filter for power line interference (50/60 Hz)
            notch_freq = 50  # Adjust to 60 if needed
            q = 30.0
            w0 = notch_freq / (self.sampling_rate/2)
            b, a = signal.iirnotch(w0, q)
            data = signal.filtfilt(b, a, data, axis=0)
            
            # Apply bandpass filter (0.5-100 Hz)
            data = self.apply_bandpass_filter(data, 0.5, 100)
            
            return data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            raise
        
    def get_band_power(self, data: np.ndarray, band: str) -> float:
        """Calculate power in a specific frequency band."""
        try:
            # Ensure data is contiguous float32 array
            data = np.ascontiguousarray(data, dtype=np.float32)
            
            # Ensure 1D array for single channel analysis
            if len(data.shape) > 1:
                if data.shape[1] == 1:
                    data = data.ravel()
                elif data.shape[0] == 1:
                    data = data.ravel()
                else:
                    raise ValueError(f"Input shape {data.shape} not supported for band power calculation")
            
            # Get frequency band limits
            low_freq, high_freq = self.freq_bands[band]
            
            # Calculate power spectrum
            freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, len(data)))
            
            # Find indices corresponding to frequency band
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # Calculate average power in band
            band_power = np.mean(psd[idx_band])
            
            return float(band_power)
            
        except Exception as e:
            logger.error(f"Error calculating {band} band power: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return 0.0
        
    def compute_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all features for emotion classification."""
        try:
            logging.info(f"Computing features from data shape: {data.shape}")
            
            # Preprocess the data
            processed_data = self.preprocess_data(data)
            logging.info(f"Preprocessed data shape: {processed_data.shape}")
            
            # Extract frequency features
            freq_features = self.extract_frequency_features(processed_data)
            
            # Log shapes of all features
            for name, feature in freq_features.items():
                logging.info(f"Feature {name} shape: {feature.shape}")
            
            # Compute additional features
            variance = np.var(processed_data, axis=0)
            hjorth_mob = self._hjorth_mobility(processed_data)
            hjorth_comp = self._hjorth_complexity(processed_data)
            
            features = {
                **freq_features,
                'variance': variance,
                'hjorth_mobility': hjorth_mob,
                'hjorth_complexity': hjorth_comp
            }
            
            # Verify all features have the expected shape
            for name, feature in features.items():
                if not isinstance(feature, np.ndarray):
                    logging.error(f"Feature {name} is not a numpy array: {type(feature)}")
                else:
                    logging.info(f"Final feature {name} shape: {feature.shape}")
            
            return features
            
        except Exception as e:
            logging.error(f"Error in compute_features: {e}")
            raise
        
    def _hjorth_mobility(self, data: np.ndarray) -> np.ndarray:
        """Compute Hjorth mobility parameter."""
        diff = np.diff(data, axis=0)
        return np.std(diff, axis=0) / np.std(data, axis=0)
        
    def _hjorth_complexity(self, data: np.ndarray) -> np.ndarray:
        """Compute Hjorth complexity parameter."""
        diff1 = np.diff(data, axis=0)
        diff2 = np.diff(diff1, axis=0)
        return (np.std(diff2, axis=0) * np.std(data, axis=0)) / (np.std(diff1, axis=0) ** 2) 