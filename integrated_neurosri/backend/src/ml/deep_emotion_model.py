import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging
import os
from datetime import datetime
import sys
import time
import traceback
from scipy.signal import butter, filtfilt, iirnotch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'eeg_model.log'))
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = Path(project_root) / 'models' / 'cnn_lstm_emotion_model.pt'
SCALER_PATH = Path(project_root) / 'models' / 'eeg_scaler.joblib'
DATASET_PATH = Path(project_root) / 'dataset' / 'merged_dataset_swapped.csv'
SAMPLING_RATE = 255  # Hz
WINDOW_SIZE = 240    # Window size for feature extraction

# Create models directory if it doesn't exist
os.makedirs(MODEL_PATH.parent, exist_ok=True)

def apply_filters(eeg_data, sampling_rate=255):
    """Apply preprocessing filters to EEG data"""
    try:
        # Notch filter for power line interference (50 Hz)
        notch_freq = 50.0  # For 50 Hz power line
        quality_factor = 30.0  # Quality factor
        b_notch, a_notch = iirnotch(notch_freq, quality_factor, sampling_rate)
        
        # Bandpass filter (0.5 - 45 Hz for EEG)
        nyquist_freq = sampling_rate / 2
        low_cut = 0.5 / nyquist_freq
        high_cut = 45.0 / nyquist_freq
        b_bandpass, a_bandpass = butter(4, [low_cut, high_cut], btype='band')
        
        filtered_data = np.zeros_like(eeg_data)
        
        # Apply filters to each channel
        for channel in range(eeg_data.shape[1]):
            # Apply notch filter
            notched = filtfilt(b_notch, a_notch, eeg_data[:, channel])
            
            # Apply bandpass filter
            filtered_data[:, channel] = filtfilt(b_bandpass, a_bandpass, notched)
            
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error in filter application: {str(e)}")
        logger.error(traceback.format_exc())
        return eeg_data  # Return original data if filtering fails

# Define frequency bands for feature extraction
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)  # Updated to match our filter
}

# Define the CNN-LSTM hybrid model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels=2, seq_length=WINDOW_SIZE):
        super(CNNLSTMModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2) 
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # x shape: [batch_size, channels, seq_length]
        
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape for LSTM: [batch_size, seq_length//8, 64]
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class EEGEmotionClassifier:
    """CNN-LSTM hybrid model for EEG emotion classification"""
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.last_prediction = ("neutral", 0.5)  # Store last prediction
        self.window_size = WINDOW_SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        try:
            if model_path is None:
                model_path = MODEL_PATH
            else:
                model_path = Path(model_path)
            
            if model_path.exists() and SCALER_PATH.exists():
                self.load_model(model_path)
            else:
                logger.info("No existing model found. Training new model...")
                self.train_model()
                
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            raise
    
    def extract_features(self, eeg_data):
        """Extract features from raw EEG data"""
        try:
            # Ensure contiguous array with proper memory layout and shape
            eeg_data = np.ascontiguousarray(eeg_data, dtype=np.float32)
            
            # Apply preprocessing filters
            filtered_data = apply_filters(eeg_data, SAMPLING_RATE)
            
            # Return the filtered data directly for the CNN-LSTM model
            # The CNN part of the model will extract features automatically
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return eeg_data  # Return original data if extraction fails
    
    def load_model(self, model_path):
        """Load trained model and scaler"""
        try:
            # Load model
            self.model = CNNLSTMModel()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler if needed
            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
                
            logger.info("Model and scaler loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def train_model(self):
        """Train the CNN-LSTM model using cleaned dataset"""
        try:
            # Load and prepare training data
            logger.info("Loading dataset...")
            data = pd.read_csv(DATASET_PATH)
            logger.info(f"Successfully loaded dataset with {len(data)} samples")
            
            # Process training data into windows
            X = []
            y = []
            temp_buffer = []
            labels = []
            current_label = None
            
            for idx, row in data.iterrows():
                if idx % 1000 == 0:
                    logger.info(f"Processing row {idx}/{len(data)}")
                
                if current_label is None:
                    current_label = row['Emotion']
                
                if row['Emotion'] == current_label:
                    channel_data = [float(row['Channel 1']), float(row['Channel 2'])]
                    temp_buffer.append(channel_data)
                    labels.append(row['Emotion'])
                    
                    if len(temp_buffer) == self.window_size:
                        # Process window
                        window_array = np.array(temp_buffer, dtype=np.float32)
                        processed_window = self.extract_features(window_array)
                        X.append(processed_window)
                        y.append(current_label)
                        # Use sliding window with 50% overlap
                        temp_buffer = temp_buffer[self.window_size//2:]
                        labels = labels[self.window_size//2:]
                else:
                    temp_buffer = []
                    labels = []
                    current_label = row['Emotion']
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            
            # Convert labels to numerical format
            label_map = {'Relaxed': 1, 'Stressed': 0}
            y = np.array([label_map[str(label).strip()] for label in y])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
            )
            
            # Create PyTorch datasets and dataloaders
            # Transpose X to match the expected input shape [batch_size, channels, seq_length]
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize the model
            self.model = CNNLSTMModel().to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            # Training loop
            num_epochs = 30
            best_accuracy = 0.0
            
            logger.info("Training CNN-LSTM model...")
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                train_accuracy = correct / total
                train_loss = train_loss / len(train_loader)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        # Statistics
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                val_accuracy = correct / total
                val_loss = val_loss / len(test_loader)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Log progress
                logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                # Save best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), MODEL_PATH)
                    logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            # Load best model
            self.model.load_state_dict(torch.load(MODEL_PATH))
            logger.info(f"Best model loaded with accuracy: {best_accuracy:.4f}")
            
            # Final evaluation
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            final_accuracy = correct / total
            logger.info(f"Final test accuracy: {final_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    def predict_realtime(self, eeg_data):
        """Real-time emotion prediction from EEG data"""
        try:
            # Convert and reshape input data
            eeg_data = np.ascontiguousarray(eeg_data, dtype=np.float32)
            
            # Handle different input shapes
            if len(eeg_data.shape) == 1:  # Single sample (2,)
                eeg_data = eeg_data.reshape(1, 2)
            elif len(eeg_data.shape) == 2:
                if eeg_data.shape[1] != 2:  # If channels are not in second dimension
                    eeg_data = eeg_data.reshape(-1, 2)
            
            # Process window - already filtered in extract_features
            processed_data = self.extract_features(eeg_data)
            
            # Reshape for CNN input [batch, channels, sequence]
            processed_data = np.transpose(processed_data, (1, 0)).reshape(1, 2, -1)
            
            # Convert to torch tensor
            tensor_data = torch.FloatTensor(processed_data).to(self.device)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(tensor_data)
                probabilities = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()
            
            # Convert to emotion label
            emotion = "stressed" if pred_class == 0 else "relaxed"
            
            # Add small random variation to prevent getting stuck
            confidence = min(0.95, max(0.05, confidence * (1 + np.random.normal(0, 0.05))))
            
            # Store prediction
            self.last_prediction = (emotion, confidence)
            logger.info(f"Prediction: {emotion}, Confidence: {confidence:.4f}")
            
            return emotion, confidence
                
        except Exception as e:
            logger.error(f"Error in real-time prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return self.last_prediction if hasattr(self, 'last_prediction') else ("neutral", 0.5)

# For direct testing
if __name__ == "__main__":
    try:
        # Initialize classifier
        classifier = EEGEmotionClassifier()
        logger.info("Classifier initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing classifier: {str(e)}") 