from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
import numpy as np
import g4f  # This suggests using GPT4Free for responses
from datetime import datetime
import traceback
import json
import socket
from src.chatbot.chatbot_service import ChatbotService
import json

# Import necessary modules for model training and prediction
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import welch, coherence
import time
import os
import asyncio
import csv
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient
import threading

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Check if BLE is disabled
BLE_DISABLED = False
if Path(__file__).parent.joinpath('disable_ble.flag').exists():
    print("BLE connectivity is DISABLED by flag file")
    logger.info("BLE connectivity is DISABLED by flag file")
    BLE_DISABLED = True

# Now use absolute imports
try:
    from src.chatbot.therapeutic_bot import TherapeuticBot
    from src.chatbot.chatbot_service import ChatbotService
    from src.eeg.signal_processing import SignalProcessor
except ImportError:
    # Add current directory to path as fallback
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
    
    try:
        from src.chatbot.therapeutic_bot import TherapeuticBot
        from src.chatbot.chatbot_service import ChatbotService
        from src.eeg.signal_processing import SignalProcessor
    except ImportError:
        print("ERROR: Could not import required modules. Check src directory structure.")
        # Create stub implementations for required classes
        class TherapeuticBot:
            def __init__(self): pass
            def get_response(self, message): return "Service unavailable - module import error"
            
        class ChatbotService:
            def __init__(self): pass
            def get_response(self, message): return "Service unavailable - module import error"
            def update_system_prompt(self, context): pass
            
        class SignalProcessor:
            def __init__(self): pass
            def process(self, data): return data

# Function to find an available port
def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                # Port is available
                return port
    # If we get here, no ports were available
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")

# Find available port for the Flask app
try:
    PORT = find_available_port(5000)
    logger.info(f"Using port {PORT} for the server")
except Exception as e:
    logger.error(f"Error finding available port: {e}")
    PORT = 5000  # Default fallback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store user information
user_info = None

# BLE and data parameters for EEG device
DEVICE_NAME_PREFIX = "NPG"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
SINGLE_SAMPLE_LEN = 7
BLOCK_COUNT = 10
NEW_PACKET_LEN = SINGLE_SAMPLE_LEN * BLOCK_COUNT
SAMPLE_RATE = 250  # Hz
WINDOW_SEC = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC

# Define CNN Model from train_model.py
class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Reduced filters, added padding
        self.bn1 = nn.BatchNorm1d(16)  # Added batch normalization
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)  # Reduced dropout
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)  # Reduced filters, added padding
        self.bn2 = nn.BatchNorm1d(32)  # Added batch normalization
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)  # Reduced dropout
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._get_conv_output_size(input_size)
        
        self.fc1 = nn.Linear(conv_output_size, 64)  # Reduced size
        self.bn3 = nn.BatchNorm1d(64)  # Added batch normalization
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)
        
    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout3(torch.relu(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x

# Class for EEG data acquisition with BLE support
class EEGStream:
    def __init__(self):
        self.eeg_buffers = [[], [], []]  # Buffers for 3 channels
        self.running = False
        self.connected = False
        self.client = None
        self.ble_thread = None
        self.last_timestamp = time.time()
        self.mock_data_thread = None
        
    async def scan_and_connect(self):
        """Scan for and connect to the EEG device"""
        # If BLE is disabled, skip the real connection
        if BLE_DISABLED:
            logger.info("BLE is disabled, using mock data instead")
            return False
            
        try:
            logger.info("Scanning for NPG BLE devices...")
            devices = await BleakScanner.discover(timeout=10.0)
            npg_devices = [d for d in devices if d.name and d.name.startswith(DEVICE_NAME_PREFIX)]
            
            if not npg_devices:
                logger.error("No NPG BLE devices found!")
                return False
                
            device = npg_devices[0]
            logger.info(f"Connecting to {device.name} ({device.address})...")
            
            # Use exception handling for connection
            try:
                # Create a client with timeout to prevent hanging
                self.client = BleakClient(device, timeout=20.0)
                
                # Connect with timeout
                connect_task = asyncio.create_task(self.client.connect())
                try:
                    await asyncio.wait_for(connect_task, timeout=15.0)
                except asyncio.TimeoutError:
                    logger.error("Connection attempt timed out")
                    if not connect_task.done():
                        connect_task.cancel()
                    return False
                
                if self.client.is_connected:
                    # Send START command
                    await self.client.write_gatt_char(CONTROL_CHAR_UUID, b"START", response=True)
                    logger.info("Connected and sent START command.")
                    
                    # Setup data notification with proper error handling
                    try:
                        await self.client.start_notify(DATA_CHAR_UUID, self.safe_notification_callback)
                        self.connected = True
                        self.running = True
                    except Exception as e:
                        logger.error(f"Error setting up notifications: {e}")
                        logger.error(traceback.format_exc())
                        # Try to disconnect properly if notification setup fails
                        if self.client.is_connected:
                            await self.client.disconnect()
                        return False
                    
                    # Start saving data to CSV
                    with open("eeg_data.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["timestamp", "ch1", "ch2", "ch3"])
                    
                    return True
                else:
                    logger.error("Failed to connect to device")
                    return False
            except Exception as e:
                logger.error(f"Error connecting to device: {e}")
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"Error in BLE connection: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_mock_data(self):
        """Generate sine wave mock data for testing without a real device"""
        logger.info("Starting mock data generation")
        self.connected = True
        self.running = True
        
        # Create empty CSV
        with open("eeg_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ch1", "ch2", "ch3"])
        
        sample_count = 0
        try:
            while self.running:
                now = time.time()
                self.last_timestamp = now
                
                # Generate three sine waves at different frequencies
                t = sample_count / SAMPLE_RATE
                ch1 = int(100 * np.sin(2 * np.pi * 10 * t))  # 10 Hz
                ch2 = int(80 * np.sin(2 * np.pi * 5 * t))    # 5 Hz
                ch3 = int(60 * np.sin(2 * np.pi * 2 * t))    # 2 Hz
                
                # Add some noise
                ch1 += np.random.randint(-10, 10)
                ch2 += np.random.randint(-10, 10)
                ch3 += np.random.randint(-10, 10)
                
                # Store in buffers
                for idx, val in enumerate([ch1, ch2, ch3]):
                    self.eeg_buffers[idx].append(val)
                    if len(self.eeg_buffers[idx]) > WINDOW_SIZE:
                        self.eeg_buffers[idx] = self.eeg_buffers[idx][-WINDOW_SIZE:]
                
                # Save to CSV
                with open("eeg_data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([now, ch1, ch2, ch3])
                
                sample_count += 1
                time.sleep(1 / SAMPLE_RATE)  # Simulate real-time data
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
        finally:
            self.running = False
            self.connected = False
    
    def safe_notification_callback(self, sender, data):
        """Wrapper for handle_notification that catches all exceptions"""
        try:
            if self.running:
                self.handle_notification(sender, data)
        except Exception as e:
            logger.error(f"Error in notification callback: {e}")
            # Don't reraise - this prevents the infinite error loop
            
    def handle_notification(self, sender, data):
        """Handle incoming data from BLE device"""
        if not self.running:
            return
            
        now = time.time()
        self.last_timestamp = now
        
        try:
            # Process data packet
            if len(data) == NEW_PACKET_LEN:
                for i in range(0, NEW_PACKET_LEN, SINGLE_SAMPLE_LEN):
                    sample = data[i:i+SINGLE_SAMPLE_LEN]
                    if len(sample) == SINGLE_SAMPLE_LEN:
                        self.process_sample(sample, now)
            elif len(data) == SINGLE_SAMPLE_LEN:
                self.process_sample(data, now)
                
            # Optionally, save to CSV
            try:
                with open("eeg_data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    ch1 = self.eeg_buffers[0][-1] if self.eeg_buffers[0] else 0
                    ch2 = self.eeg_buffers[1][-1] if self.eeg_buffers[1] else 0
                    ch3 = self.eeg_buffers[2][-1] if self.eeg_buffers[2] else 0
                    writer.writerow([now, ch1, ch2, ch3])
            except Exception as csv_err:
                logger.error(f"Error writing to CSV: {csv_err}")
                
        except Exception as e:
            logger.error(f"Error processing BLE data: {e}")
            # Don't reraise exceptions to prevent cascading errors
    
    def process_sample(self, sample, timestamp):
        """Process a single EEG sample"""
        ch1 = int.from_bytes(sample[1:3], byteorder='big', signed=True)
        ch2 = int.from_bytes(sample[3:5], byteorder='big', signed=True)
        ch3 = int.from_bytes(sample[5:7], byteorder='big', signed=True)
        
        # Store in buffers
        for idx, val in enumerate([ch1, ch2, ch3]):
            self.eeg_buffers[idx].append(val)
            if len(self.eeg_buffers[idx]) > WINDOW_SIZE:
                self.eeg_buffers[idx] = self.eeg_buffers[idx][-WINDOW_SIZE:]
    
    def connect(self):
        """Start BLE connection in a separate thread"""
        if self.connected:
            logger.info("Already connected to EEG device")
            return True
        
        # If BLE is disabled, use mock data instead
        if BLE_DISABLED:
            logger.info("BLE is disabled, using mock data instead")
            self.mock_data_thread = threading.Thread(target=self.generate_mock_data)
            self.mock_data_thread.daemon = True
            self.mock_data_thread.start()
            time.sleep(1)  # Give a moment for mock data generation to start
            return self.connected
            
        def run_ble_loop():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run connection in a try-except block
                try:
                    loop.run_until_complete(self.scan_and_connect())
                except Exception as e:
                    logger.error(f"Error in BLE connection loop: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    # Always close the loop properly
                    try:
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error closing event loop: {e}")
            except Exception as e:
                logger.error(f"Fatal error in BLE thread: {e}")
                logger.error(traceback.format_exc())
        
        # Create a new thread for BLE operations
        self.ble_thread = threading.Thread(target=run_ble_loop)
        self.ble_thread.daemon = True
        self.ble_thread.start()
        
        # Wait a bit to see if connection succeeds
        time.sleep(5)
        
        # If BLE connection failed, fall back to mock data
        if not self.connected:
            logger.info("BLE connection failed, falling back to mock data")
            self.mock_data_thread = threading.Thread(target=self.generate_mock_data)
            self.mock_data_thread.daemon = True
            self.mock_data_thread.start()
            time.sleep(1)  # Give a moment for mock data generation to start
            
        return self.connected
    
    def disconnect(self):
        """Disconnect from the EEG device"""
        if self.connected:
            self.running = False
            
            # If using mock data, just wait for the thread to end
            if BLE_DISABLED or (self.mock_data_thread and self.mock_data_thread.is_alive()):
                logger.info("Stopping mock data generation")
                time.sleep(0.5)  # Give thread time to clean up
                self.connected = False
                return
            
            # For BLE, handle disconnection in a structured way
            try:
                # Create a new thread for disconnection to avoid blocking
                disconnect_thread = threading.Thread(target=self._disconnect_thread)
                disconnect_thread.daemon = True
                disconnect_thread.start()
                
                # Wait for disconnect to complete with timeout
                disconnect_thread.join(timeout=5.0)
                if disconnect_thread.is_alive():
                    logger.warning("Disconnect thread did not complete in time")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self.connected = False
                # Allow time for all callbacks to complete
                time.sleep(0.5)
    
    def _disconnect_thread(self):
        """Thread function to handle BLE disconnection"""
        try:
            # Create a new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Define the async disconnect function
            async def do_disconnect():
                try:
                    if self.client and self.client.is_connected:
                        logger.info("Disconnecting from BLE device...")
                        # Use timeout to prevent hanging
                        disconnect_task = asyncio.create_task(self.client.disconnect())
                        try:
                            await asyncio.wait_for(disconnect_task, timeout=3.0)
                            logger.info("Successfully disconnected from BLE device")
                        except asyncio.TimeoutError:
                            logger.warning("Disconnect timed out, forcing disconnect")
                            if not disconnect_task.done():
                                disconnect_task.cancel()
                except Exception as e:
                    logger.error(f"Error in disconnect: {e}")
                    logger.error(traceback.format_exc())
            
            # Run the disconnect function with a timeout
            try:
                loop.run_until_complete(asyncio.wait_for(do_disconnect(), timeout=4.0))
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Disconnect operation timed out or was cancelled")
            except Exception as e:
                logger.error(f"Unexpected error in disconnect loop: {e}")
            finally:
                # Always close the loop
                try:
                    loop.close()
                except Exception as loop_err:
                    logger.error(f"Error closing event loop: {loop_err}")
        except Exception as thread_err:
            logger.error(f"Error in disconnect thread: {thread_err}")
            logger.error(traceback.format_exc())
    
    def get_data(self):
        """Get the latest EEG data"""
        if not self.connected or not self.running:
            return None
            
        # Return the latest data as a numpy array
        if all(len(buffer) > 0 for buffer in self.eeg_buffers):
            # Create a numpy array with shape (samples, channels)
            data = np.column_stack([np.array(buffer) for buffer in self.eeg_buffers])
            return data, self.last_timestamp
        
        return None

# Define class for EEG prediction
class EEGPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define frequency bands for feature extraction
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Load model if available or create placeholder
        self.load_model()
    
    def load_model(self):
        try:
            # Check if model exists
            model_path = Path('model.pth')
            scaler_path = Path('scaler.joblib')
            
            if model_path.exists() and scaler_path.exists():
                # Create model with fixed architecture
                self.model = CNNModel(input_size=570, num_classes=12)
                # Load state dict
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                # Load scaler
                self.scaler = joblib.load(scaler_path)
                logger.info("Model and scaler loaded successfully")
            else:
                logger.warning("Model files not found. Using placeholder predictions.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
    
    def compute_band_powers(self, data, sf):
        band_powers = []
        for ch in range(data.shape[1]):
            f, Pxx = welch(data[:, ch], sf, nperseg=sf)
            for band, (low, high) in self.bands.items():
                idx = np.logical_and(f >= low, f <= high)
                # Use trapezoid instead of trapz
                band_powers.append(np.trapezoid(Pxx[idx], f[idx]))
        return band_powers

    def compute_coherence_features(self, data, sf):
        coh_feats = []
        # Adapt based on available channels
        num_channels = data.shape[1]
        if num_channels >= 3:
            pairs = [(0,1), (0,2), (1,2)]
        elif num_channels == 2:
            pairs = [(0,1)]
        else:
            return []  # No coherence features if only one channel
            
        for ch1, ch2 in pairs:
            f, Cxy = coherence(data[:, ch1], data[:, ch2], sf, nperseg=sf)
            for band, (low, high) in self.bands.items():
                idx = np.logical_and(f >= low, f <= high)
                coh_feats.append(np.mean(Cxy[idx]))
        return coh_feats

    def extract_features(self, window, sample_rate=250):
        """Extract features from EEG window"""
        try:
            band_powers = self.compute_band_powers(window, sample_rate)
            coh_feats = self.compute_coherence_features(window, sample_rate)
            feats = np.array(band_powers + coh_feats).reshape(1, -1)
            
            # Create a fixed-size feature vector for the scaler
            feature_size = 570  # Expected feature size
            
            # Pad or truncate features to match expected size
            if feats.shape[1] < feature_size:
                pad_width = ((0, 0), (0, feature_size - feats.shape[1]))
                return np.pad(feats, pad_width, mode='constant')
            elif feats.shape[1] > feature_size:
                return feats[:, :feature_size]
            return feats
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            logger.error(traceback.format_exc())
            return np.zeros((1, 570))  # Return zero features as fallback
    
    def predict(self, eeg_data, sample_rate=250):
        """Predict mental state from EEG data"""
        try:
            if self.model is None:
                # Return default values if model not loaded
                return "neutral", 0.5
            
            # Extract features
            features = self.extract_features(eeg_data, sample_rate)
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=np.nanmedian(features))
            
            # Scale the features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Convert to tensor
            x = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                confidence = float(probs[pred_idx])
            
            # Map index to state
            states = {
                0: "Normal/baseline state",
                1: "Mild stress",
                2: "Anxiety",
                3: "Focus/concentration",
                4: "Relaxation",
                5: "Drowsiness",
                6: "Mental fatigue",
                7: "High cognitive load",
                8: "Emotional response",
                9: "Depression indicators",
                10: "ADHD-like patterns",
                11: "Potential cognitive impairment"
            }
            
            state = states.get(pred_idx, "Unknown")
            
            # Get mental state description and possible disorders
            mental_state, confidence, possible_disorders = self.get_mental_state_description(state, probs)
            
            # Get counseling response
            counseling_response = self.get_counseling_response(mental_state, confidence, possible_disorders)
            
            # Get healing techniques
            healing_techniques = self.get_healing_techniques(mental_state, possible_disorders)
            
            # Save prediction to file for external access
            self.save_prediction_to_file(mental_state, confidence, eeg_data, counseling_response, 
                                   possible_disorders, healing_techniques)
            
            return mental_state, confidence
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            return "neutral", 0.5  # Return default on error

    def get_mental_state_description(self, pred_label, probs):
        """Maps the prediction label to a user-friendly description of the mental state"""
        confidence = float(max(probs)) * 100
        
        # Additional mental disorder pattern detection
        potential_disorders = {
            "Anxiety": ["Generalized Anxiety Disorder", "Panic Disorder"],
            "Depression indicators": ["Major Depressive Disorder", "Persistent Depressive Disorder"],
            "ADHD-like patterns": ["ADHD", "Executive Function Disorder"],
            "Potential cognitive impairment": ["OCD", "Cognitive Impairment"]
        }
        
        # Check if this state might indicate a potential disorder
        possible_disorders = potential_disorders.get(str(pred_label), [])
        
        return pred_label, confidence, possible_disorders

    def get_counseling_response(self, mental_state, confidence, possible_disorders=None):
        """Provides appropriate counseling responses based on the detected mental state"""
        if possible_disorders is None:
            possible_disorders = []
        
        # Base responses for mental states
        responses = {
            "Normal/baseline state": "Your brain activity appears normal. Is there anything specific you'd like to discuss?",
            "Mild stress": "I'm detecting some signs of mild stress. Would you like to try a quick breathing exercise?",
            "Anxiety": "Your brain patterns suggest you might be experiencing anxiety. Let's focus on grounding techniques to help center you.",
            "Focus/concentration": "You seem to be in a focused state. This is a great time for productive work or learning.",
            "Relaxation": "Your brain waves indicate you're in a relaxed state. This is ideal for creativity and well-being.",
            "Drowsiness": "I notice signs of drowsiness. Would you like a quick energizing activity?",
            "Mental fatigue": "Your patterns suggest mental fatigue. It might be a good time for a break.",
            "High cognitive load": "You appear to be processing a lot of information. Remember to take breaks to consolidate learning.",
            "Emotional response": "I'm detecting an emotional response. Would you like to talk about what you're feeling?",
            "Depression indicators": "I'm noticing patterns that might indicate low mood. How are you feeling today?",
            "ADHD-like patterns": "Your brain activity shows patterns sometimes associated with attention variations. Would you like some focus techniques?",
            "Potential cognitive impairment": "I'm noticing some unusual patterns in your brain activity. How is your thinking clarity today?"
        }
        
        # Get the base response
        response = responses.get(mental_state, "I'm analyzing your brain patterns. How can I assist you?")
        
        # Add disorder-specific insights and questions if confidence is high enough
        if confidence > 75 and possible_disorders:
            disorder_questions = {
                "Generalized Anxiety Disorder": "I've noticed consistent anxiety patterns. Do you often feel worried about many different things?",
                "Panic Disorder": "Your patterns resemble those seen during panic episodes. Have you experienced sudden intense fear or physical symptoms like racing heart?",
                "Major Depressive Disorder": "I'm seeing patterns consistent with depression. Have you been experiencing persistent sadness or loss of interest in activities?",
                "Persistent Depressive Disorder": "These patterns suggest ongoing low mood. Have you been feeling down for most days over a long period?",
                "ADHD": "Your activity patterns are consistent with attention variations. Do you often struggle with staying focused or completing tasks?",
                "Executive Function Disorder": "I'm noticing patterns related to executive function. Do you have difficulty with planning, organizing, or initiating tasks?",
                "OCD": "These patterns show similarities to obsessive-compulsive patterns. Do you experience unwanted thoughts or feel compelled to perform certain actions?",
                "Cognitive Impairment": "I'm detecting unusual cognitive patterns. Have you noticed changes in your memory, thinking, or problem-solving abilities?"
            }
            
            # Add a question about the most likely disorder
            primary_disorder = possible_disorders[0]
            if primary_disorder in disorder_questions:
                response += f"\n\n{disorder_questions[primary_disorder]}"
        
        return response

    def get_healing_techniques(self, mental_state, possible_disorders=None):
        """Provides appropriate healing techniques based on the detected mental state and possible disorders"""
        if possible_disorders is None:
            possible_disorders = []
        
        # Base techniques for different mental states
        techniques = {
            "Mild stress": [
                "Progressive Muscle Relaxation: Tense and then relax each muscle group in your body, one at a time.",
                "4-7-8 Breathing: Inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds."
            ],
            "Anxiety": [
                "5-4-3-2-1 Grounding: Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
                "Box Breathing: Inhale for 4 counts, hold for 4, exhale for 4, hold for 4."
            ],
            "Mental fatigue": [
                "Pomodoro Technique: Work for 25 minutes, then take a 5-minute break.",
                "Nature Break: Spend a few minutes looking at nature or natural images."
            ],
            "Depression indicators": [
                "Behavioral Activation: Schedule and engage in activities that bring you joy or satisfaction.",
                "Gratitude Practice: List three things you're grateful for each day."
            ],
            "ADHD-like patterns": [
                "Task Chunking: Break large tasks into smaller, manageable chunks.",
                "Body Doubling: Work alongside someone else (even virtually) to help maintain focus."
            ]
        }
        
        # Disorder-specific techniques
        disorder_techniques = {
            "Generalized Anxiety Disorder": [
                "Worry Time: Schedule a specific time to address worries, postponing worry thoughts until then.",
                "Cognitive Restructuring: Identify and challenge anxious thoughts."
            ],
            "Panic Disorder": [
                "Diaphragmatic Breathing: Practice deep belly breathing to reduce physical symptoms.",
                "AWARE technique: Accept anxiety, Watch it, Act normally, Repeat, Expect the best."
            ],
            "Major Depressive Disorder": [
                "Pleasant Activity Scheduling: Plan enjoyable activities throughout your week.",
                "Thought Records: Track negative thoughts and practice reframing them."
            ],
            "OCD": [
                "ERP (Exposure and Response Prevention): Gradually expose yourself to triggers without performing compulsions.",
                "Mindful Observation: Notice obsessive thoughts without judging or responding to them."
            ],
            "ADHD": [
                "Implementation Intentions: Create specific if-then plans for tasks.",
                "Environmental Modifications: Organize your workspace to minimize distractions."
            ]
        }
        
        # Get base techniques for the mental state
        result = techniques.get(mental_state, [])
        
        # Add disorder-specific techniques if applicable
        for disorder in possible_disorders:
            if disorder in disorder_techniques:
                result.extend(disorder_techniques[disorder])
        
        # Limit to at most 3 techniques
        return result[:3]

    def save_prediction_to_file(self, mental_state, confidence, eeg_data, counseling_response, possible_disorders=None, healing_techniques=None):
        """Save the prediction results to a JSON file for the connector to read"""
        if possible_disorders is None:
            possible_disorders = []
        if healing_techniques is None:
            healing_techniques = []
        
        data = {
            "mental_state": mental_state,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "eeg_data": eeg_data.tolist() if isinstance(eeg_data, np.ndarray) else eeg_data,
            "counseling_response": counseling_response,
            "possible_disorders": possible_disorders,
            "healing_techniques": healing_techniques,
            "alert_level": "high" if possible_disorders and confidence > 75 else "normal"
        }
        
        try:
            with open('prediction_output.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving prediction to file: {e}")

# Initialize components with error handling
try:
    logger.info("Initializing EEG stream connection...")
    eeg_stream = EEGStream()
    signal_processor = SignalProcessor()
    
    # Initialize emotion classifier
    logger.info("Loading EEG predictor...")
    emotion_classifier = EEGPredictor()
    
    chatbot = ChatbotService()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    # Continue execution but with limited functionality
    logger.warning("Continuing with limited functionality")

# Add a new endpoint to get raw EEG data
@app.route('/api/raw-eeg', methods=['GET'])
def get_raw_eeg():
    try:
        result = eeg_stream.get_data()
        if result is not None:
            eeg_data, timestamp = result
            return jsonify({
                'eeg_data': eeg_data.tolist() if isinstance(eeg_data, np.ndarray) else eeg_data,
                'timestamp': timestamp,
                'channels': 3,
                'sample_rate': SAMPLE_RATE,
                'status': 'connected'
            })
    except Exception as e:
        logger.error(f"Error in raw EEG endpoint: {e}")
    
    # Return default response in case of error
    return jsonify({
        'eeg_data': [],
        'timestamp': datetime.now().isoformat(),
        'channels': 3,
        'sample_rate': SAMPLE_RATE,
        'status': 'disconnected'
    })

@app.route('/api/emotion', methods=['GET'])
def get_emotion():
    try:
        # Get latest EEG data
        result = eeg_stream.get_data()
        if result is not None:
            eeg_data, timestamp = result
            
            # Ensure data is numpy array with correct dtype and shape
            eeg_data = np.asarray(eeg_data, dtype=np.float32)
            
            # Reshape based on input shape
            if len(eeg_data.shape) == 1:
                eeg_data = eeg_data.reshape(-1, 2)  # Reshape to (samples, channels)
            elif len(eeg_data.shape) == 2 and eeg_data.shape[1] != 2:
                eeg_data = eeg_data.T  # Transpose if channels are in rows
            
            # Get current time
            current_time = datetime.now()
            
            # Get the latest prediction from the emotion classifier
            emotion, confidence = emotion_classifier.predict(eeg_data)
            logger.info(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
            
            # Calculate time until next 5-second mark
            seconds_until_next = 5 - (current_time.timestamp() % 5)
            
            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'eeg_data': eeg_data[:, 0].tolist(),  # First channel data
                'timestamp': current_time.isoformat(),
                'next_prediction_in': seconds_until_next,  # Dynamic time until next prediction
                'counseling_response': emotion_classifier.get_counseling_response(emotion, confidence),
                'healing_techniques': emotion_classifier.get_healing_techniques(emotion)
            })
    except Exception as e:
        logger.error(f"Error in emotion endpoint: {e}")
    
    # Return default response in case of error or no data
    return jsonify({
        'emotion': 'neutral',
        'confidence': 0.5,
        'eeg_data': [],
        'timestamp': datetime.now().isoformat(),
        'next_prediction_in': 5,
        'is_setup_phase': False,
        'setup_complete': True,
        'counseling_response': "I'm initializing the system. How can I help you today?",
        'healing_techniques': []
    })

@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    """Return a more detailed analysis with mental state, counseling, and techniques"""
    try:
        # Try to read from the prediction output file first
        try:
            with open('prediction_output.json', 'r') as f:
                data = json.load(f)
                return jsonify(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
        # Fall back to live analysis
        result = eeg_stream.get_data()
        if result is not None:
            eeg_data, timestamp = result
            
            # Ensure data is numpy array with correct dtype and shape
            eeg_data = np.asarray(eeg_data, dtype=np.float32)
            
            if len(eeg_data.shape) == 1:
                eeg_data = eeg_data.reshape(-1, 2)
                
            # Get prediction
            mental_state, confidence = emotion_classifier.predict(eeg_data)
            
            # Get mental state description
            mental_state, confidence, possible_disorders = emotion_classifier.get_mental_state_description(mental_state, [confidence])
            
            # Get counseling response
            counseling_response = emotion_classifier.get_counseling_response(mental_state, confidence, possible_disorders)
            
            # Get healing techniques
            healing_techniques = emotion_classifier.get_healing_techniques(mental_state, possible_disorders)
            
            return jsonify({
                'mental_state': mental_state,
                'confidence': confidence,
                'eeg_data': eeg_data[:, 0].tolist(),
                'timestamp': datetime.now().isoformat(),
                'counseling_response': counseling_response,
                'possible_disorders': possible_disorders,
                'healing_techniques': healing_techniques,
                'alert_level': "high" if possible_disorders and confidence > 75 else "normal"
            })
    except Exception as e:
        logger.error(f"Error in analysis endpoint: {e}")
        
    # Return default response
    return jsonify({
        'mental_state': 'neutral',
        'confidence': 0.5,
        'eeg_data': [],
        'timestamp': datetime.now().isoformat(),
        'counseling_response': "I'm initializing the system. How can I help you today?",
        'possible_disorders': [],
        'healing_techniques': [],
        'alert_level': "normal"
    })

@app.route('/api/user-info', methods=['POST'])
def user_info():
    data = request.json
    if not data:
        return jsonify({'error': 'No user info provided'}), 400
    
    try:
        # Update chatbot with user info
        user_context = json.dumps(data)
        chatbot.update_system_prompt(user_context)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    user_info = data.get('user_info', {})
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get response from chatbot
        response = chatbot.get_response(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a new endpoint to visualize EEG data
@app.route('/api/visualize-eeg', methods=['GET'])
def visualize_eeg():
    """Return a basic HTML visualization of the EEG data"""
    try:
        result = eeg_stream.get_data()
        if result is not None:
            eeg_data, timestamp = result
            
            # Create simple HTML visualization
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>EEG Visualization</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    canvas { margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>Real-time EEG Data</h1>
                <div style="width:100%; height:600px;">
                    <canvas id="eegChart"></canvas>
                </div>
                <script>
                    const ctx = document.getElementById('eegChart').getContext('2d');
                    const chart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: Array.from({length: %d}, (_, i) => -i),
                            datasets: [
                                {
                                    label: 'Channel 1',
                                    data: %s,
                                    borderColor: 'rgb(31, 119, 180)',
                                    tension: 0.1
                                },
                                {
                                    label: 'Channel 2',
                                    data: %s,
                                    borderColor: 'rgb(255, 127, 14)',
                                    tension: 0.1
                                },
                                {
                                    label: 'Channel 3',
                                    data: %s,
                                    borderColor: 'rgb(44, 160, 44)',
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    reverse: true,
                                    title: {
                                        display: true,
                                        text: 'Samples ago'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Amplitude'
                                    }
                                }
                            }
                        }
                    });
                    
                    // Auto refresh
                    setInterval(() => {
                        fetch('/api/raw-eeg')
                            .then(response => response.json())
                            .then(data => {
                                if (data.eeg_data.length > 0) {
                                    // Update chart data
                                    chart.data.datasets[0].data = data.eeg_data.map(row => row[0]);
                                    chart.data.datasets[1].data = data.eeg_data.map(row => row[1]);
                                    chart.data.datasets[2].data = data.eeg_data.map(row => row[2]);
                                    chart.update();
                                }
                            });
                    }, 1000);
                </script>
            </body>
            </html>
            """ % (
                len(eeg_data),
                json.dumps(eeg_data[:, 0].tolist()),
                json.dumps(eeg_data[:, 1].tolist()),
                json.dumps(eeg_data[:, 2].tolist())
            )
            
            return html
    except Exception as e:
        logger.error(f"Error in visualize EEG endpoint: {e}")
        return """
        <html><body>
            <h1>EEG Visualization</h1>
            <p>Error: Could not retrieve EEG data. Please make sure the EEG device is connected.</p>
        </body></html>
        """

# Replace before_first_request with a better initialization approach
def initialize_streaming():
    global eeg_stream
    logger.info("Attempting to connect to EEG device...")
    if eeg_stream.connect():
        logger.info("✓ Flask server successfully connected to EEG device")
    else:
        logger.error("✗ Failed to connect to EEG device - Please ensure device is powered on and in range")

# Initialize streaming when the app starts
with app.app_context():
    initialize_streaming()

if __name__ == '__main__':
    try:
        print(f"Starting Flask server on port {PORT}...")
        # Allow connections from any IP
        app.run(host='0.0.0.0', port=PORT, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        
    # Ensure clean disconnection
    if 'eeg_stream' in globals():
        eeg_stream.disconnect() 