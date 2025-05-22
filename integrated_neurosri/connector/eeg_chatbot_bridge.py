import time
import logging
import requests
import json
import numpy as np
import threading
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('connector.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants for EEG system
EEG_DATA_FILE = 'eeg_data.csv'
EEG_PREDICTION_FILE = 'prediction_output.json'
EEG_UPDATE_INTERVAL = 1  # seconds

# Constants for NeuroSri-2.0 API
NEUROSRI_API_URL = "http://localhost:5000"  # Base URL
NEUROSRI_EMOTION_ENDPOINT = f"{NEUROSRI_API_URL}/api/emotion"
NEUROSRI_CHAT_ENDPOINT = f"{NEUROSRI_API_URL}/api/chat"
QUESTIONNAIRE_API_URL = "http://localhost:5100"
QUESTIONNAIRE_START_ENDPOINT = f"{QUESTIONNAIRE_API_URL}/api/questionnaire/start"

class EEGChatbotBridge:
    def __init__(self):
        self.last_prediction = None
        self.last_emotion_sent = None
        self.last_sent_time = 0
        self.running = False
        self.eeg_data_last_modified = 0
        self.thread = None
    
    def read_latest_eeg_data(self):
        """Read the latest EEG data from the CSV file"""
        try:
            # Check if file exists and has been modified
            if not os.path.exists(EEG_DATA_FILE):
                logger.warning(f"EEG data file {EEG_DATA_FILE} not found")
                return None
            
            current_modified = os.path.getmtime(EEG_DATA_FILE)
            if current_modified == self.eeg_data_last_modified:
                return None  # File hasn't changed
            
            self.eeg_data_last_modified = current_modified
            
            # Read the last line of the CSV file
            with open(EEG_DATA_FILE, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:  # Only header or empty file
                    return None
                
                last_line = lines[-1].strip()
                values = last_line.split(',')
                
                # Extract channels and timestamps
                # Assuming format: timestamp, ch1, ch2, ch3, ...
                if len(values) < 4:  # Need at least timestamp + 3 channels
                    return None
                
                timestamp = values[0]
                channels = [float(val) for val in values[1:4]]  # First 3 EEG channels
                
                return {
                    "timestamp": timestamp,
                    "eeg_data": channels
                }
                
        except Exception as e:
            logger.error(f"Error reading EEG data: {str(e)}")
            return None
    
    def get_realtime_prediction(self):
        """
        Get the latest prediction from our EEG system
        """
        try:
            # Check if file exists
            if not os.path.exists(EEG_PREDICTION_FILE):
                return None
            
            # Read prediction from file
            with open(EEG_PREDICTION_FILE, 'r') as f:
                prediction_data = json.load(f)
                
            # Get timestamp to check if prediction is fresh
            timestamp = prediction_data.get("timestamp")
            if timestamp:
                pred_time = datetime.fromisoformat(timestamp)
                now = datetime.now()
                # If prediction is more than 5 seconds old, consider it stale
                if (now - pred_time).total_seconds() > 5:
                    return None
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error getting real-time prediction: {str(e)}")
            return None
    
    def send_to_neurosri(self, prediction_data):
        """Send the prediction data to the NeuroSri-2.0 server"""
        try:
            # Extract the mental state and confidence
            mental_state = prediction_data.get("mental_state")
            confidence = prediction_data.get("confidence", 0.5)
            
            # Extract additional information
            possible_disorders = prediction_data.get("possible_disorders", [])
            healing_techniques = prediction_data.get("healing_techniques", [])
            alert_level = prediction_data.get("alert_level", "normal")
            counseling_response = prediction_data.get("counseling_response", "")
            
            # Map our mental state to NeuroSri-2.0 emotion format
            emotion = self.map_mental_state_to_emotion(mental_state)
            
            # Only send if emotion changed or significant time passed
            current_time = time.time()
            if (emotion != self.last_emotion_sent or 
                current_time - self.last_sent_time > 10):  # Send at least every 10 seconds
                
                # Prepare payload for NeuroSri-2.0 emotion endpoint
                emotion_payload = {
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "eeg_data": prediction_data.get("eeg_data", []),
                    "metadata": {
                        "possible_disorders": possible_disorders,
                        "healing_techniques": healing_techniques,
                        "alert_level": alert_level
                    }
                }
                
                # Send to NeuroSri-2.0 emotion endpoint
                response = requests.post(
                    NEUROSRI_EMOTION_ENDPOINT, 
                    json=emotion_payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"Sent emotion update to NeuroSri-2.0: {emotion} (confidence: {confidence:.2f})")
                    if possible_disorders:
                        logger.info(f"Possible disorders: {', '.join(possible_disorders)}")
                    self.last_emotion_sent = emotion
                    self.last_sent_time = current_time
                else:
                    logger.error(f"Failed to send emotion: {response.status_code} - {response.text}")
                
                # If we have a counseling response, send it to the chat endpoint
                if counseling_response:
                    # Send message to chat endpoint
                    chat_payload = {
                        "message": counseling_response,
                        "source": "system",
                        "metadata": {
                            "possible_disorders": possible_disorders,
                            "healing_techniques": healing_techniques,
                            "alert_level": alert_level
                        }
                    }
                    
                    chat_response = requests.post(
                        NEUROSRI_CHAT_ENDPOINT,
                        json=chat_payload,
                        timeout=5
                    )
                    
                    if chat_response.status_code == 200:
                        logger.info(f"Sent counseling message to NeuroSri-2.0: {counseling_response[:50]}...")
                    else:
                        logger.error(f"Failed to send counseling message: {chat_response.status_code}")
                
                # If we have possible disorders with high confidence, start a questionnaire
                if alert_level == "high" and possible_disorders:
                    try:
                        # Start a questionnaire for the first disorder
                        self.start_questionnaire(possible_disorders[0])
                    except Exception as e:
                        logger.error(f"Error starting questionnaire: {str(e)}")
                
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
        except Exception as e:
            logger.error(f"Error sending to NeuroSri-2.0: {str(e)}")
    
    def map_mental_state_to_emotion(self, mental_state):
        """
        Maps mental state classifications to emotional categories 
        that NeuroSri-2.0 can understand.
        """
        # Define mapping from mental states to emotions
        state_to_emotion = {
            "Normal/baseline state": "neutral",
            "Mild stress": "stressed",
            "Anxiety": "anxious",
            "Focus/concentration": "focused",
            "Relaxation": "relaxed",
            "Drowsiness": "drowsy",
            "Mental fatigue": "tired",
            "High cognitive load": "concentrated",
            "Emotional response": "emotional",
            "Depression indicators": "sad",
            "ADHD-like patterns": "distracted",
            "Potential cognitive impairment": "confused"
        }
        
        # Default to neutral if state not recognized
        return state_to_emotion.get(mental_state, "neutral")
    
    def start_questionnaire(self, disorder):
        """Start a questionnaire for a specific disorder"""
        try:
            # Prepare payload for the questionnaire API
            payload = {
                "disorder": disorder
            }
            
            # Send request to start the questionnaire
            response = requests.post(
                QUESTIONNAIRE_START_ENDPOINT,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                session_id = response.json().get("session_id")
                logger.info(f"Started questionnaire for {disorder} (session ID: {session_id})")
            else:
                logger.error(f"Failed to start questionnaire: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error starting questionnaire: {str(e)}")
    
    def run(self):
        """Main loop to continuously read predictions and send to NeuroSri-2.0"""
        self.running = True
        
        while self.running:
            try:
                # Get latest prediction from our EEG system
                prediction_data = self.get_realtime_prediction()
                
                if prediction_data:
                    # Send to NeuroSri-2.0
                    self.send_to_neurosri(prediction_data)
                
                # Sleep to avoid busy waiting
                time.sleep(EEG_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in connector loop: {str(e)}")
                time.sleep(5)  # Wait a bit longer on error
    
    def start(self):
        """Start the connector in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Connector already running")
            return
        
        logger.info("Starting EEG to NeuroSri-2.0 connector...")
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Connector started")
    
    def stop(self):
        """Stop the connector"""
        logger.info("Stopping connector...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5)
        logger.info("Connector stopped")

def main():
    """Main function"""
    logger.info("Starting EEG to NeuroSri-2.0 bridge...")
    
    bridge = EEGChatbotBridge()
    
    try:
        bridge.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    finally:
        bridge.stop()
        logger.info("Bridge shut down.")

if __name__ == "__main__":
    main() 