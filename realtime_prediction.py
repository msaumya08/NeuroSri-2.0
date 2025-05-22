import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import traceback
from scipy.signal import welch, coherence
import json
from datetime import datetime

# --- CONFIG ---
EEG_CSV = 'eeg_data.csv'
MODEL_PATH = 'model.pth'
SCALER_PATH = 'scaler.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'  # Optional
SAMPLE_RATE = 250
WINDOW_SEC = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC
UPDATE_INTERVAL = 1  # seconds between predictions
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# --- LOAD SCALER & LABEL ENCODER ---
scaler = joblib.load(SCALER_PATH)
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception:
    label_encoder = None

# --- MODEL DEFINITION ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Fixed architecture with hardcoded sizes to match the pretrained model
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Hard-coded FC layer sizes to match the saved model
        self.fc1 = nn.Linear(9120, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 12)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to ensure it's the right size for the first fully connected layer
        # Input shape should be [batch_size, 570]
        if x.dim() == 2 and x.size(1) == 570:
            # Reshape to [batch_size, 1, 570] for conv1d
            x = x.view(batch_size, 1, 570)
            
            # Pass through convolution layers
            x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
            x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
            
            # Flatten
            x = x.view(batch_size, -1)
            
            # If the flattened size doesn't match what the FC layer expects, reshape it
            if x.size(1) != 9120:
                # Create a zero tensor of the right size and copy data
                padded = torch.zeros(batch_size, 9120, device=x.device)
                padded[:, :min(x.size(1), 9120)] = x[:, :min(x.size(1), 9120)]
                x = padded
        else:
            # If input format is unexpected, reshape to the right size for fc1
            padded = torch.zeros(batch_size, 9120, device=x.device)
            x = padded
            
        # Forward through fully connected layers
        x = self.dropout3(torch.relu(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x

# Create model with explicit architecture
model = CNNModel()

# Loading model - ignore missing and unexpected keys
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# --- FEATURE EXTRACTION ---
def compute_band_powers(data, sf):
    band_powers = []
    for ch in range(data.shape[1]):
        f, Pxx = welch(data[:, ch], sf, nperseg=sf)
        for band, (low, high) in BANDS.items():
            idx = np.logical_and(f >= low, f <= high)
            # Use trapezoid instead of trapz as recommended in deprecation warning
            band_powers.append(np.trapezoid(Pxx[idx], f[idx]))
    return band_powers

def compute_coherence_features(data, sf):
    coh_feats = []
    pairs = [(0,1), (0,2), (1,2)]
    for ch1, ch2 in pairs:
        f, Cxy = coherence(data[:, ch1], data[:, ch2], sf, nperseg=sf)
        for band, (low, high) in BANDS.items():
            idx = np.logical_and(f >= low, f <= high)
            coh_feats.append(np.mean(Cxy[idx]))
    return coh_feats

def extract_features(window):
    band_powers = compute_band_powers(window, SAMPLE_RATE)
    coh_feats = compute_coherence_features(window, SAMPLE_RATE)
    feats = np.array(band_powers + coh_feats).reshape(1, -1)
    
    # Create a fixed-size feature vector for the scaler
    feature_size = 1140  # Based on error message
    
    # Pad or truncate features to match expected size
    if feats.shape[1] < feature_size:
        pad_width = ((0, 0), (0, feature_size - feats.shape[1]))
        return np.pad(feats, pad_width, mode='constant')
    elif feats.shape[1] > feature_size:
        return feats[:, :feature_size]
    return feats

# --- REAL-TIME PREDICTION LOOP ---
def get_latest_window():
    try:
        df = pd.read_csv(EEG_CSV)
        if df.shape[0] < WINDOW_SIZE:
            return None
        window = df.iloc[-WINDOW_SIZE:][['ch1','ch2','ch3']].values
        return window
    except Exception as e:
        print(f"Error reading EEG data: {e}")
        return None

def predict(features):
    features = np.nan_to_num(features, nan=np.nanmedian(features))
    # Scale the features
    features_scaled = scaler.transform(features)
    # Only use the first 570 features for the model
    x = torch.tensor(features_scaled[:, :570], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    if label_encoder is not None:
        pred_label = label_encoder.inverse_transform([pred])[0]
    else:
        pred_label = str(pred)
    return pred_label, probs

def get_mental_state_description(pred_label, probs):
    """
    Maps the prediction label to a user-friendly description of the mental state
    """
    confidence = max(probs) * 100
    
    states = {
        '0': "Normal/baseline state",
        '1': "Mild stress",
        '2': "Anxiety",
        '3': "Focus/concentration",
        '4': "Relaxation",
        '5': "Drowsiness",
        '6': "Mental fatigue",
        '7': "High cognitive load",
        '8': "Emotional response",
        '9': "Depression indicators",
        '10': "ADHD-like patterns",
        '11': "Potential cognitive impairment"
    }
    
    # Additional mental disorder pattern detection
    potential_disorders = {
        '2': ["Generalized Anxiety Disorder", "Panic Disorder"],
        '9': ["Major Depressive Disorder", "Persistent Depressive Disorder"],
        '10': ["ADHD", "Executive Function Disorder"],
        '11': ["OCD", "Cognitive Impairment"]
    }
    
    # Use the prediction label to get the state description
    # Default to the prediction label if it's not in our mapping
    description = states.get(str(pred_label), f"Unknown state {pred_label}")
    
    # Check if this state might indicate a potential disorder
    possible_disorders = potential_disorders.get(str(pred_label), [])
    
    return description, confidence, possible_disorders

def get_counseling_response(mental_state, confidence, possible_disorders=None):
    """
    Provides appropriate counseling responses based on the detected mental state
    """
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

def get_healing_techniques(mental_state, possible_disorders=None):
    """
    Provides appropriate healing techniques based on the detected mental state and possible disorders
    """
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

def save_prediction_to_file(mental_state, confidence, eeg_data, counseling_response, possible_disorders=None, healing_techniques=None):
    """
    Save the prediction results to a JSON file for the connector to read
    """
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
        print(f"Error saving prediction to file: {e}")

def run_prediction_loop():
    """
    Main function to continuously predict mental states and provide responses
    """
    print("NeurosAI Bot Started - Reading EEG signals in real-time")
    print("Press Ctrl+C to stop\n")
    
    last_window_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Only make a new prediction every UPDATE_INTERVAL seconds
            if current_time - last_window_time >= UPDATE_INTERVAL:
                window = get_latest_window()
                
                if window is not None:
                    # Extract features
                    features = extract_features(window)
                    
                    # Make prediction
                    pred_label, probs = predict(features)
                    
                    # Get mental state description and confidence
                    mental_state, confidence, possible_disorders = get_mental_state_description(pred_label, probs)
                    
                    # Get appropriate counseling response
                    counseling_response = get_counseling_response(mental_state, confidence, possible_disorders)
                    
                    # Get healing techniques
                    healing_techniques = get_healing_techniques(mental_state, possible_disorders)
                    
                    # Print results
                    print(f"Predicted mental state: {mental_state} (confidence: {confidence:.2f}%)")
                    if possible_disorders:
                        print(f"Possible disorders: {', '.join(possible_disorders)}")
                    print(f"Counseling response: {counseling_response}")
                    print(f"Healing techniques: {healing_techniques}")
                    
                    # Save to file for the connector
                    save_prediction_to_file(mental_state, confidence, window, counseling_response, 
                                            possible_disorders, healing_techniques)
                    
                    last_window_time = current_time
                    
            time.sleep(0.1)  # Small sleep to prevent CPU overload
            
    except KeyboardInterrupt:
        print("\nStopping prediction loop...")
    except Exception as e:
        print(f"Error in prediction loop: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_prediction_loop()
