import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.append(str(project_root))

from src.ml.emotion_classifier import EmotionClassifier

def load_eeg_data(data_path):
    """Load and preprocess EEG data from the specified directory."""
    all_data = []
    all_labels = []
    
    logging.info(f"Loading data from {data_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(data_path)
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
        
        # Extract features (all columns except the last one)
        features = df.iloc[:, :-1].values
        
        # Extract labels (last column)
        labels = df.iloc[:, -1].values
        
        logging.info(f"Loaded dataset with {len(features)} samples and {features.shape[1]} features")
        logging.info(f"Unique emotions in dataset: {np.unique(labels)}")
        
        return features, labels
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_emotion_classifier():
    try:
        # Define data path relative to project root
        data_path = project_root / 'src' / 'dataset' / 'train_75Perc.csv'
        
        # Load the dataset
        X, y = load_eeg_data(data_path)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize the classifier
        classifier = EmotionClassifier('models/emotion_classifier.keras')
        
        # Initialize scaler with training data
        classifier.scaler.fit(X_train)
        X_train_scaled = classifier.scaler.transform(X_train)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        # Train the model
        classifier.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_pred = classifier.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        logging.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Evaluate on test set
        test_pred = classifier.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        logging.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Print detailed classification report
        logging.info("\nClassification Report:")
        logging.info("\n" + classification_report(y_test, test_pred))
        
        # Save the trained model
        classifier.save_model('models/emotion_classifier.keras')
        logging.info("Model saved successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_emotion_classifier() 