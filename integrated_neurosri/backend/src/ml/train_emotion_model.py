import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from deep_emotion_model import EEGEmotionNet
from eeg.signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ModelTrainer:
    def __init__(self, 
                 data_path: Path,
                 model_save_dir: Path,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100):
        
        self.data_path = data_path
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model save directory
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.signal_processor = SignalProcessor()
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Load and preprocess the dataset"""
        logger.info("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Extract features from raw EEG data
        X = df[['Channel 1', 'Channel 2']].values
        y = (df['Emotion'] == 'Stressed').astype(int).values
        
        # Process data in windows
        window_size = 2500  # 10 seconds at 250 Hz
        features_list = []
        labels_list = []
        
        for i in range(0, len(X) - window_size + 1, window_size//2):  # 50% overlap
            window = X[i:i + window_size]
            if len(window) == window_size:
                # Extract features
                features = self.signal_processor.preprocess_data(window)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(y[i + window_size//2])  # Use middle window label
        
        X_processed = np.array(features_list)
        y_processed = np.array(labels_list)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_processed, 
            test_size=0.2, 
            random_state=42,
            stratify=y_processed
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Create data loaders
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size
        )
        
        logger.info(f"Prepared {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
    def train(self):
        """Train the model"""
        model = EEGEmotionNet().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5
        )
        
        best_accuracy = 0
        train_losses = []
        val_accuracies = []
        
        logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            
            for batch_X, batch_y in progress_bar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            
            logger.info(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    model.state_dict(), 
                    self.model_save_dir / 'emotion_model.pth'
                )
                logger.info(f'New best model saved with accuracy: {accuracy:.2f}%')
            
            scheduler.step(accuracy)
        
        # Plot training progress
        self.plot_training_progress(train_losses, val_accuracies)
        
    def plot_training_progress(self, train_losses, val_accuracies):
        """Plot training progress"""
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_progress.png')
        plt.close()

def train_emotion_model():
    """Main training function"""
    try:
        # Setup paths
        project_root = Path(__file__).resolve().parent.parent.parent
        data_path = project_root / 'dataset' / 'merged_dataset_swapped.csv'
        model_save_dir = project_root / 'models'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            data_path=data_path,
            model_save_dir=model_save_dir,
            batch_size=32,
            learning_rate=0.001,
            num_epochs=100
        )
        
        # Prepare data and train model
        trainer.prepare_data()
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_emotion_model() 