import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.chords.chords import detect_hardware, parse_data

def main():
    try:
        # Detect CHORDS hardware
        serial_conn = detect_hardware()
        if serial_conn is None:
            logger.error("Failed to detect CHORDS hardware!")
            return 1

        # Start LSL stream
        logger.info("Starting LSL stream...")
        parse_data(
            ser=serial_conn,
            lsl_flag=True,  # Enable LSL streaming
            csv_flag=False,  # Disable CSV logging
            verbose=True,    # Enable verbose output
            run_time=None,   # Run indefinitely
            inverted=False   # Normal signal polarity
        )
        
        return 0
    except Exception as e:
        logger.error(f"Error starting LSL stream: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 