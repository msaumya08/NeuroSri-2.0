import os
import subprocess
import sys
import time
import signal
import logging
from pathlib import Path
import shutil
import traceback
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Start NeuroSri integrated system')
parser.add_argument('--disable-ble', action='store_true', help='Disable BLE connectivity')
args = parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Get the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent
CHATBOT_DIR = ROOT_DIR / 'Chatbot' / 'NeuroSri-2.0'
INTEGRATED_DIR = ROOT_DIR / 'integrated_neurosri' / 'backend'

# Add questionnaire API import
try:
    from questionnaire_api import start_in_thread as start_questionnaire_api
    QUESTIONNAIRE_API_AVAILABLE = True
except ImportError:
    logger.warning("Questionnaire API module not found. Questionnaire functionality will be disabled.")
    QUESTIONNAIRE_API_AVAILABLE = False

def check_dependencies():
    """Check if all required files and dependencies exist"""
    logger.info("Checking dependencies...")
    
    # Check if integrated server exists
    if not (INTEGRATED_DIR / 'server.py').exists():
        logger.error("Integrated server not found!")
        return False
    
    # Check if model files exist
    if not (ROOT_DIR / 'model.pth').exists():
        logger.warning("model.pth not found! EEG prediction will use default values.")
    
    if not (ROOT_DIR / 'scaler.joblib').exists():
        logger.warning("scaler.joblib not found! EEG prediction will use default values.")
    
    # Check if disorder questionnaire module exists
    if not (ROOT_DIR / 'disorder_questionnaire.py').exists():
        logger.warning("disorder_questionnaire.py not found! Mental disorder detection will be limited.")
    
    logger.info("All dependencies are present.")
    return True

def copy_integrated_server():
    """Copy the integrated server to the chatbot directory"""
    logger.info("Copying integrated server to chatbot directory...")
    
    try:
        # Ensure the target directory exists
        os.makedirs(CHATBOT_DIR, exist_ok=True)
        
        # Copy the integrated server
        shutil.copy2(INTEGRATED_DIR / 'server.py', CHATBOT_DIR / 'server.py')
        
        # If BLE is disabled, create a flag file
        if args.disable_ble:
            with open(CHATBOT_DIR / 'disable_ble.flag', 'w') as f:
                f.write('BLE connectivity disabled by command line argument')
            logger.info("Created BLE disable flag file")
        
        # Copy necessary model files if they exist
        for file in ['model.pth', 'scaler.joblib']:
            source = ROOT_DIR / file
            if source.exists():
                shutil.copy2(source, CHATBOT_DIR / file)
                logger.info(f"Copied {file} to chatbot directory")
        
        # Copy src directory if it exists in the integrated backend
        src_dir = INTEGRATED_DIR / 'src'
        target_src_dir = CHATBOT_DIR / 'src'
        if src_dir.exists():
            # Remove old src directory if it exists
            if target_src_dir.exists():
                shutil.rmtree(target_src_dir)
            
            # Copy the entire src directory
            shutil.copytree(src_dir, target_src_dir)
            logger.info("Copied src directory to chatbot directory")
        else:
            # If no src directory in integrated_neurosri/backend, check the root directory
            root_src_dir = ROOT_DIR / 'src'
            if root_src_dir.exists():
                # Remove old src directory if it exists
                if target_src_dir.exists():
                    shutil.rmtree(target_src_dir)
                
                # Copy the entire src directory
                shutil.copytree(root_src_dir, target_src_dir)
                logger.info("Copied src directory from root to chatbot directory")
            else:
                logger.error("No src directory found!")
                return False
                
        logger.info("Server files copied successfully")
        return True
    except Exception as e:
        logger.error(f"Error copying server files: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def start_integrated_server():
    """Start the integrated server"""
    logger.info("Starting integrated server...")
    
    # Change to the chatbot directory
    os.chdir(CHATBOT_DIR)
    
    server_process = subprocess.Popen(
        [sys.executable, 'server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    logger.info("Integrated server started with PID: %d", server_process.pid)
    
    # Change back to the root directory
    os.chdir(ROOT_DIR)
    return server_process

def start_frontend():
    """Start the frontend web application"""
    logger.info("Starting frontend web application...")
    
    # Change to the frontend directory
    os.chdir(CHATBOT_DIR)
    
    frontend_process = subprocess.Popen(
        ['npm', 'run', 'dev'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True  # Required for npm commands
    )
    
    logger.info("Frontend started with PID: %d", frontend_process.pid)
    
    # Change back to the root directory
    os.chdir(ROOT_DIR)
    return frontend_process

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

def monitor_processes(processes):
    """Monitor running processes and their output"""
    try:
        logger.info("Monitoring processes. Press Ctrl+C to stop...")
        
        # Create non-blocking stdout/stderr readers
        from queue import Queue
        from threading import Thread
        import sys

        def enqueue_output(out, queue, process_name, is_error=False):
            for line in iter(out.readline, ''):
                if line:
                    queue.put((process_name, line.strip(), is_error))
            out.close()

        # Create queues and start reader threads
        output_queue = Queue()
        for name, process in processes.items():
            if process.stdout:
                Thread(target=enqueue_output, args=(process.stdout, output_queue, name, False), daemon=True).start()
            if process.stderr:
                Thread(target=enqueue_output, args=(process.stderr, output_queue, name, True), daemon=True).start()

        # Monitor until keyboard interrupt
        while True:
            # Check if any process has terminated
            for name, process in list(processes.items()):
                if process.poll() is not None:
                    logger.error(f"Process {name} (PID: {process.pid}) terminated unexpectedly with code {process.poll()}")
                    # Remove the terminated process from the dict
                    del processes[name]
                    
                    # If all processes have terminated, exit the loop
                    if not processes:
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
        if process.poll() is None:  # If process is still running
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
                try:
                    process.kill()
                except Exception:
                    pass

def main():
    """Main function to start all system components"""
    logger.info("="*50)
    logger.info("Starting NeuroSri Integrated EEG Counseling System")
    if args.disable_ble:
        logger.info("BLE connectivity is DISABLED")
    logger.info("Features:")
    logger.info("- Real-time EEG BLE acquisition")
    logger.info("- Mental state classification")
    logger.info("- Potential mental disorder detection")
    logger.info("- Personalized counseling responses")
    logger.info("- Mental health technique recommendations")
    logger.info("- Interactive disorder questionnaires")
    logger.info("="*50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        return 1
    
    # Copy integrated server to chatbot directory
    if not copy_integrated_server():
        logger.error("Failed to copy integrated server. Exiting.")
        return 1
    
    # Dictionary to keep track of all processes
    processes = {}
    
    # Start integrated server
    server_process = start_integrated_server()
    processes["integrated_server"] = server_process
    
    # Give the server time to initialize
    time.sleep(2)
    
    # Create questionnaire results directory
    os.makedirs("questionnaire_results", exist_ok=True)
    
    # Start questionnaire API server
    questionnaire_thread = start_questionnaire_server()
    
    # Start frontend
    frontend_process = start_frontend()
    processes["frontend"] = frontend_process
    
    # Monitor processes
    monitor_processes(processes)
    
    return 0

if __name__ == "__main__":
    main() 