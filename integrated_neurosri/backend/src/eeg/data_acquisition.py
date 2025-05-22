import numpy as np
from pylsl import StreamInlet, resolve_streams
import sys
import time
from typing import Optional, Tuple
import logging
from collections import deque

logger = logging.getLogger(__name__)

class EEGStream:
    def __init__(self, buffer_size=250):  # 1 second at 250Hz
        self.inlet = None
        self.running = False
        self.current_emotion = "neutral"
        self.current_confidence = 0.5
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        
    def connect(self) -> bool:
        """Connect to an existing LSL stream."""
        try:
            logger.info("Looking for LSL stream...")
            max_retries = 10
            retry_count = 0
            
            while retry_count < max_retries:
                streams = resolve_streams()
                if streams:
                    self.inlet = StreamInlet(streams[0])
                    logger.info("Connected to LSL stream")
                    self.running = True
                    return True
                    
                logger.info("Stream not found, retrying...")
                time.sleep(1)
                retry_count += 1
                
            logger.error("No LSL stream found after maximum retries")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to LSL stream: {e}")
            return False
            
    def get_data(self, duration: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get EEG data with buffering."""
        if not self.inlet:
            return None
            
        try:
            # Get new samples
            while len(self.buffer) < self.buffer_size:
                sample, timestamp = self.inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                self.buffer.append(sample)
            
            # Return if we have enough data
            if len(self.buffer) >= 10:  # Minimum required samples
                data = np.array(list(self.buffer))
                return data, time.time()
            
            return None
            
        except Exception as e:
            logger.error(f"Error acquiring data: {e}")
            return None
            
    def disconnect(self):
        """Disconnect from the LSL stream."""
        self.running = False
        if self.inlet:
            self.inlet = None
        self.buffer.clear()
        logger.info("Disconnected from LSL stream") 