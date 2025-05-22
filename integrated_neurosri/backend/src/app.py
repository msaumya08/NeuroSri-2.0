from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
import numpy as np
import g4f  # This suggests using GPT4Free for responses
from datetime import datetime
import traceback
from chatbot.chatbot_service import ChatbotService
import json

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now use absolute imports
from src.chatbot.therapeutic_bot import TherapeuticBot
from src.eeg.data_acquisition import EEGStream
from src.eeg.signal_processing import SignalProcessor
from src.ml.deep_emotion_model import EEGEmotionClassifier


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store user information
user_info = None

# Initialize components with error handling
try:
    logger.info("Initializing EEG stream connection...")
    eeg_stream = EEGStream()
    signal_processor = SignalProcessor()
    
    # Initialize emotion classifier with model download if needed
    logger.info("Loading emotion classifier model...")
    emotion_classifier = EEGEmotionClassifier()
    
    chatbot = ChatbotService()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    sys.exit(1)

@app.route('/api/emotion', methods=['GET'])
def get_emotion():
    # For now, return mock data
    return jsonify({
        'emotion': 'neutral',
        'confidence': 0.5,
        'eeg_data': [],
        'is_setup_phase': False,
        'setup_complete': True
    })

def get_current_eeg_analysis():
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
            emotion, confidence = emotion_classifier.predict_realtime(eeg_data)
            logger.info(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
            
            # Calculate time until next 5-second mark
            seconds_until_next = 5 - (current_time.timestamp() % 5)
            
            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'eeg_data': eeg_data[:, 0].tolist(),  # First channel data
                'timestamp': current_time.isoformat(),
                'next_prediction_in': seconds_until_next  # Dynamic time until next prediction
            })
        
        return jsonify({
            'emotion': 'neutral',
            'confidence': 0,
            'eeg_data': [],
            'timestamp': datetime.now().isoformat(),
            'next_prediction_in': 5
        })
    except Exception as e:
        logger.error(f"Error in emotion endpoint: {e}")
        return jsonify({'error': str(e)}), 500

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

# Replace before_first_request with a better initialization approach
def initialize_streaming():
    global eeg_stream
    logger.info("Attempting to connect to LSL stream...")
    if eeg_stream.connect():
        logger.info("✓ Flask server successfully connected to LSL stream")
    else:
        logger.error("✗ Failed to connect to LSL stream - Please ensure LSL stream is running")

# Initialize streaming when the app starts
with app.app_context():
    initialize_streaming()

if __name__ == '__main__':
    try:
        print("Starting Flask server...")
        # Allow connections from any IP
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}") 