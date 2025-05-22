import os
import subprocess
import sys
import time
import signal
import logging
import threading
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integrated_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent
EEG_DIR = ROOT_DIR / 'eeg_processing'
BACKEND_DIR = ROOT_DIR / 'backend'
FRONTEND_DIR = ROOT_DIR / 'frontend'
CONNECTOR_DIR = ROOT_DIR / 'connector'

# Import questionnaire API if available
try:
    sys.path.append(str(EEG_DIR))
    from questionnaire_api import start_in_thread as start_questionnaire_api
    QUESTIONNAIRE_API_AVAILABLE = True
except ImportError:
    logger.warning("Questionnaire API module not found. Questionnaire functionality will be disabled.")
    QUESTIONNAIRE_API_AVAILABLE = False

def check_dependencies():
    """Check if all required files and dependencies exist"""
    logger.info("Checking dependencies...")
    
    # Check EEG processing dependencies
    if not (EEG_DIR / 'realtime_prediction.py').exists():
        logger.error("realtime_prediction.py not found!")
        return False
    
    if not (ROOT_DIR / 'model.pth').exists():
        logger.error("model.pth not found! Please ensure you have the trained model.")
        return False
    
    if not (ROOT_DIR / 'scaler.joblib').exists():
        logger.error("scaler.joblib not found! Please ensure you have the scaler file.")
        return False
    
    # Check NeuroSri-2.0 backend
    if not (BACKEND_DIR / 'server.py').exists():
        logger.error("server.py not found!")
        return False
    
    # Check connector
    if not (CONNECTOR_DIR / 'eeg_chatbot_bridge.py').exists():
        logger.error("eeg_chatbot_bridge.py not found!")
        return False
    
    # Check frontend
    if not (FRONTEND_DIR / 'package.json').exists():
        logger.error("Frontend package.json not found!")
        return False
    
    logger.info("All dependencies are present.")
    return True

def copy_model_files():
    """Copy model files to the expected locations"""
    try:
        import shutil
        
        # Copy model and scaler to EEG processing directory if they exist
        if (ROOT_DIR / 'model.pth').exists():
            shutil.copy(ROOT_DIR / 'model.pth', EEG_DIR / 'model.pth')
            logger.info("Copied model.pth to EEG processing directory")
        
        if (ROOT_DIR / 'scaler.joblib').exists():
            shutil.copy(ROOT_DIR / 'scaler.joblib', EEG_DIR / 'scaler.joblib')
            logger.info("Copied scaler.joblib to EEG processing directory")
        
        # Create empty eeg_data.csv if it doesn't exist
        if not (ROOT_DIR / 'eeg_data.csv').exists():
            with open(ROOT_DIR / 'eeg_data.csv', 'w') as f:
                f.write("timestamp,ch1,ch2,ch3\n")
            logger.info("Created empty eeg_data.csv file")
        
        # Create prediction_output.json if it doesn't exist
        if not (ROOT_DIR / 'prediction_output.json').exists():
            with open(ROOT_DIR / 'prediction_output.json', 'w') as f:
                import json
                from datetime import datetime
                json.dump({
                    "mental_state": "Normal/baseline state",
                    "confidence": 0.5,
                    "timestamp": datetime.now().isoformat(),
                    "eeg_data": [0, 0, 0],
                    "counseling_response": "Initializing system...",
                    "possible_disorders": [],
                    "healing_techniques": [],
                    "alert_level": "normal"
                }, f)
            logger.info("Created initial prediction_output.json file")
            
        # Create questionnaire_results directory if it doesn't exist
        os.makedirs(ROOT_DIR / 'questionnaire_results', exist_ok=True)
        logger.info("Created questionnaire_results directory if it didn't exist")
            
    except Exception as e:
        logger.error(f"Error copying model files: {str(e)}")
        return False
    
    return True

def start_eeg_processing():
    """Start the EEG data processing and real-time prediction"""
    logger.info("Starting EEG data processing...")
    
    os.chdir(EEG_DIR)
    eeg_process = subprocess.Popen(
        [sys.executable, 'realtime_prediction.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    os.chdir(ROOT_DIR)
    
    logger.info("EEG processing started with PID: %d", eeg_process.pid)
    return eeg_process

def start_neurosri_backend():
    """Start the NeuroSri-2.0 backend server"""
    logger.info("Starting NeuroSri-2.0 backend server...")
    
    os.chdir(BACKEND_DIR)
    server_process = subprocess.Popen(
        [sys.executable, 'server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    os.chdir(ROOT_DIR)
    
    logger.info("NeuroSri-2.0 backend server started with PID: %d", server_process.pid)
    return server_process

def start_bridge():
    """Start the EEG to NeuroSri-2.0 bridge"""
    logger.info("Starting EEG to NeuroSri-2.0 bridge...")
    
    os.chdir(CONNECTOR_DIR)
    bridge_process = subprocess.Popen(
        [sys.executable, 'eeg_chatbot_bridge.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    os.chdir(ROOT_DIR)
    
    logger.info("EEG to NeuroSri-2.0 bridge started with PID: %d", bridge_process.pid)
    return bridge_process

def start_questionnaire_server():
    """Start the questionnaire API server"""
    logger.info("Starting questionnaire API server...")
    
    if not QUESTIONNAIRE_API_AVAILABLE:
        logger.warning("Questionnaire API not available. Skipping.")
        return None
    
    try:
        # Start in a thread using the imported function
        thread = start_questionnaire_api()
        logger.info("Questionnaire API server started in background thread")
        return thread
    except Exception as e:
        logger.error(f"Error starting questionnaire API server: {str(e)}")
        return None

def start_frontend():
    """Start the frontend development server"""
    logger.info("Starting frontend development server...")
    
    os.chdir(FRONTEND_DIR)
    # Check if npm is installed
    try:
        subprocess.run(['npm', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("npm is not installed. Please install Node.js and npm to run the frontend.")
        os.chdir(ROOT_DIR)
        return None
    
    # Install dependencies if node_modules doesn't exist
    if not (FRONTEND_DIR / 'node_modules').exists():
        logger.info("Installing frontend dependencies...")
        try:
            subprocess.run(['npm', 'install'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Frontend dependencies installed successfully")
        except subprocess.SubprocessError as e:
            logger.error(f"Error installing frontend dependencies: {str(e)}")
            os.chdir(ROOT_DIR)
            return None
    
    # Start the frontend development server
    frontend_process = subprocess.Popen(
        ['npm', 'run', 'dev'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True  # Required for npm commands on Windows
    )
    
    os.chdir(ROOT_DIR)
    logger.info("Frontend started with PID: %d", frontend_process.pid)
    
    # Open the frontend in the browser after a short delay
    def open_browser():
        time.sleep(5)  # Wait for frontend to initialize
        try:
            import webbrowser
            webbrowser.open('http://localhost:5173')
            logger.info("Opened frontend in browser")
        except Exception as e:
            logger.error(f"Error opening browser: {str(e)}")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    return frontend_process

def monitor_processes(processes):
    """Monitor running processes and their output"""
    try:
        logger.info("Monitoring processes. Press Ctrl+C to stop...")
        
        # Create non-blocking stdout/stderr readers
        from queue import Queue
        from threading import Thread

        def enqueue_output(out, queue, process_name, is_error=False):
            for line in iter(out.readline, ''):
                if line:
                    queue.put((process_name, line.strip(), is_error))
            out.close()

        # Create queues and start reader threads
        output_queue = Queue()
        for name, process in processes.items():
            if process and hasattr(process, 'stdout') and process.stdout:
                Thread(target=enqueue_output, args=(process.stdout, output_queue, name, False), daemon=True).start()
            if process and hasattr(process, 'stderr') and process.stderr:
                Thread(target=enqueue_output, args=(process.stderr, output_queue, name, True), daemon=True).start()

        # Monitor until keyboard interrupt
        while True:
            # Check if any process has terminated
            for name, process in list(processes.items()):
                if process and hasattr(process, 'poll') and process.poll() is not None:
                    logger.error(f"Process {name} (PID: {process.pid}) terminated unexpectedly with code {process.poll()}")
                    # Remove the terminated process from the dict
                    del processes[name]
                    
                    # If all processes have terminated, exit the loop
                    if not any(p for p in processes.values() if p and hasattr(p, 'poll')):
                        logger.error("All processes have terminated. Exiting.")
                        return

            # Process any output
            try:
                while not output_queue.empty():
                    name, line, is_error = output_queue.get_nowait()
                    if is_error:
                        logger.error(f"[{name}] {line}")
                    else:
                        logger.info(f"[{name}] {line}")
            except Exception as e:
                logger.error(f"Error processing output: {str(e)}")
                
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
        shutdown_processes(processes)

def shutdown_processes(processes):
    """Shutdown all running processes"""
    logger.info("Shutting down all processes...")
    
    for name, process in list(processes.items()):
        if process and hasattr(process, 'poll') and process.poll() is None:  # If process is still running
            logger.info(f"Terminating {name} (PID: {process.pid})...")
            try:
                # Try graceful shutdown first
                if sys.platform == 'win32':
                    try:
                        process.terminate()
                    except Exception:
                        pass
                else:
                    try:
                        process.send_signal(signal.SIGTERM)
                    except Exception:
                        pass
                
                # Give the process a short time to terminate on its own
                try:
                    process.wait(timeout=1)
                    logger.info(f"{name} gracefully terminated.")
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    try:
                        process.kill()
                        logger.info(f"{name} forcefully terminated.")
                    except Exception as e:
                        logger.error(f"Error killing {name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error terminating {name}: {str(e)}")

def main():
    """Main function to start all system components"""
    import threading
    
    logger.info("="*50)
    logger.info("Starting Integrated NeuroSri System")
    logger.info("Features:")
    logger.info("- Real-time EEG analysis")
    logger.info("- Mental state classification")
    logger.info("- Potential mental disorder detection")
    logger.info("- Personalized counseling responses")
    logger.info("- Interactive web interface")
    logger.info("- Mental health technique recommendations")
    logger.info("- Interactive disorder questionnaires")
    logger.info("="*50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        return 1
    
    # Copy model files to expected locations
    if not copy_model_files():
        logger.error("Failed to copy model files. Exiting.")
        return 1
    
    # Dictionary to keep track of all processes
    processes = {}
    
    # Start EEG processing
    eeg_process = start_eeg_processing()
    processes["eeg_processing"] = eeg_process
    
    # Give the EEG processing time to initialize
    time.sleep(1)
    
    # Start NeuroSri-2.0 backend
    neurosri_process = start_neurosri_backend()
    processes["neurosri_backend"] = neurosri_process
    
    # Give the backend time to initialize
    time.sleep(1)
    
    # Start questionnaire API server
    questionnaire_thread = start_questionnaire_server()
    
    # Start the EEG to NeuroSri-2.0 bridge
    bridge_process = start_bridge()
    processes["eeg_bridge"] = bridge_process
    
    # Start frontend
    frontend_process = start_frontend()
    processes["frontend"] = frontend_process
    
    # Monitor processes
    monitor_processes(processes)
    
    return 0

if __name__ == "__main__":
    main() 