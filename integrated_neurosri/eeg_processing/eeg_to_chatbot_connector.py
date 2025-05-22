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

# Constants
CHATBOT_API_URL = "http://localhost:5000"  # Base URL without /api
EMOTION_ENDPOINT = f"{CHATBOT_API_URL}/api/emotion"  # Add /api/ prefix
CHAT_ENDPOINT = f"{CHATBOT_API_URL}/api/chat"  # Add /api/ prefix
EEG_DATA_FILE = 'eeg_data.csv'
UPDATE_INTERVAL = 1  # seconds

# Function to map mental state to emotion category
def map_mental_state_to_emotion(mental_state):
    """
    Maps mental state classifications to emotional categories 
    that the chatbot can understand.
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

class EEGChatbotConnector:
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
        Get the latest prediction from realtime_prediction.py
        This uses the prediction stored in a temporary file or shared memory
        """
        # For now, we'll use a simplified approach and just read from the console output
        # In a real implementation, this would use a proper IPC mechanism
        try:
            # Check if file exists (this would be created by realtime_prediction.py)
            prediction_file = "prediction_output.json"
            if not os.path.exists(prediction_file):
                return None
            
            # Read prediction from file
            with open(prediction_file, 'r') as f:
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
    
    def send_to_chatbot(self, prediction_data):
        """Send the prediction data to the chatbot server"""
        try:
            # Extract the mental state and confidence
            mental_state = prediction_data.get("mental_state")
            confidence = prediction_data.get("confidence", 0.5)
            
            # Extract additional information
            possible_disorders = prediction_data.get("possible_disorders", [])
            healing_techniques = prediction_data.get("healing_techniques", [])
            alert_level = prediction_data.get("alert_level", "normal")
            
            # Map mental state to emotion
            emotion = map_mental_state_to_emotion(mental_state)
            
            # Only send if emotion changed or significant time passed
            current_time = time.time()
            if (emotion != self.last_emotion_sent or 
                current_time - self.last_sent_time > 10):  # Send at least every 10 seconds
                
                # Prepare payload for emotion endpoint
                emotion_payload = {
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "eeg_data": prediction_data.get("eeg_data", []),
                    "possible_disorders": possible_disorders,
                    "healing_techniques": healing_techniques,
                    "alert_level": alert_level
                }
                
                # Send to emotion endpoint
                response = requests.post(
                    EMOTION_ENDPOINT, 
                    json=emotion_payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"Sent emotion update: {emotion} (confidence: {confidence:.2f})")
                    if possible_disorders:
                        logger.info(f"Possible disorders: {', '.join(possible_disorders)}")
                    self.last_emotion_sent = emotion
                    self.last_sent_time = current_time
                else:
                    logger.error(f"Failed to send emotion: {response.status_code} - {response.text}")
                
                # Get counseling response if available
                counseling_response = prediction_data.get("counseling_response")
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
                        CHAT_ENDPOINT,
                        json=chat_payload,
                        timeout=5
                    )
                    
                    if chat_response.status_code == 200:
                        logger.info(f"Sent counseling message: {counseling_response[:50]}...")
                    else:
                        logger.error(f"Failed to send counseling message: {chat_response.status_code}")
                
                # If we have an alert and healing techniques, send those separately
                if alert_level == "high" and healing_techniques:
                    techniques_message = "Based on your brain activity, here are some techniques that might help:\n\n" + \
                                         "\n".join([f"• {technique}" for technique in healing_techniques])
                    
                    techniques_payload = {
                        "message": techniques_message,
                        "source": "system",
                        "metadata": {
                            "message_type": "healing_techniques",
                            "alert_level": alert_level
                        }
                    }
                    
                    techniques_response = requests.post(
                        CHAT_ENDPOINT,
                        json=techniques_payload,
                        timeout=5
                    )
                    
                    if techniques_response.status_code == 200:
                        logger.info(f"Sent healing techniques message")
                    else:
                        logger.error(f"Failed to send healing techniques: {techniques_response.status_code}")
                
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
        except Exception as e:
            logger.error(f"Error sending to chatbot: {str(e)}")
    
    def run(self):
        """Main loop to continuously read predictions and send to chatbot"""
        self.running = True
        
        while self.running:
            try:
                # Get latest prediction
                prediction_data = self.get_realtime_prediction()
                
                if prediction_data:
                    # Send to chatbot
                    self.send_to_chatbot(prediction_data)
                
                # Sleep to avoid busy waiting
                time.sleep(UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in connector loop: {str(e)}")
                time.sleep(5)  # Wait a bit longer on error
    
    def start(self):
        """Start the connector in a separate thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Connector already running")
            return
        
        logger.info("Starting EEG to Chatbot connector...")
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
    logger.info("Starting EEG to Chatbot connector...")
    
    connector = EEGChatbotConnector()
    
    try:
        connector.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    finally:
        connector.stop()
        logger.info("Connector shut down.")

if __name__ == "__main__":
    main() 