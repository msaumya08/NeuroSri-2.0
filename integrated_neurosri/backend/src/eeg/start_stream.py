import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_eeg_stream():
    # Create a new stream info
    stream_info = StreamInfo(
        name='EEGStream',
        type='EEG',
        channel_count=2,
        nominal_srate=250,
        channel_format='float32',
        source_id='eeg_sim_123'
    )

    # Create outlet
    outlet = StreamOutlet(stream_info)
    logger.info("LSL Stream created. Starting to send data...")

    try:
        while True:
            # Simulate EEG data (2 channels)
            eeg_data = np.random.normal(0, 1, 2)
            outlet.push_sample(eeg_data)
            time.sleep(1.0/250)  # 250 Hz sampling rate

    except KeyboardInterrupt:
        logger.info("Stream stopped by user")
    except Exception as e:
        logger.error(f"Error in stream: {e}")

if __name__ == "__main__":
    start_eeg_stream() 