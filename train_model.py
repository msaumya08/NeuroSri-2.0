import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Custom Dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# CNN Model
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

# Data Loading and Preprocessing
def load_and_preprocess_data(filepath):
    # Load data with bad lines skipped
    data = pd.read_csv(filepath, on_bad_lines='skip')
    
    # Drop unnecessary columns
    columns_to_drop = ["education", "IQ", "no.", "eeg.date", "main.disorder"]
    new_data = data.drop(columns=columns_to_drop)
    
    # Select EEG features starting with "AB" or "COH"
    PSD = [col for col in new_data.columns if col.startswith("AB") or col.startswith("COH")]
    
    # Handle missing values by imputing with median
    X = new_data[PSD].values
    X = np.nan_to_num(X, nan=np.nanmedian(X))  # Replace NaN with median
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    y = new_data['specific.disorder']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Calculate class distribution
    class_counts = np.bincount(y_encoded)
    print("\nClass distribution before balancing:")
    for i, count in enumerate(class_counts):
        print(f"Class {le.classes_[i]}: {count} samples")
    
    # Remove classes with too few samples (less than 5)
    valid_classes = [i for i, count in enumerate(class_counts) if count >= 5]
    if len(valid_classes) < 2:
        raise ValueError("Not enough valid classes for classification. Need at least 2 classes with 5 or more samples each.")
    
    # Filter data to keep only valid classes
    mask = np.isin(y_encoded, valid_classes)
    X_filtered = X_scaled[mask]
    y_filtered = y_encoded[mask]
    
    # Recalculate class distribution after filtering
    class_counts = np.bincount(y_filtered)
    majority_class_size = np.max(class_counts)
    
    # Balance dataset using SMOTE
    # Only oversample minority classes to match majority class
    sampling_strategy = {i: majority_class_size for i in range(len(class_counts)) if class_counts[i] < majority_class_size}
    
    # Configure SMOTE with fewer neighbors for small classes
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(5, min(class_counts) - 1),  # Use fewer neighbors for small classes
        random_state=42
    )
    
    X_balanced, y_balanced = smote.fit_resample(X_filtered, y_filtered)
    
    print("\nClass distribution after balancing:")
    balanced_counts = np.bincount(y_balanced)
    for i, count in enumerate(balanced_counts):
        print(f"Class {le.classes_[valid_classes[i]]}: {count} samples")
    
    # Shuffle data
    balanced_data = pd.DataFrame(X_balanced, columns=PSD)
    balanced_data['specific.disorder'] = y_balanced
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    joblib.dump(scaler, 'scaler.joblib')
    
    return balanced_data, le.classes_[valid_classes]

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
           #if patience_counter >= patience:
               #print(f'\nEarly stopping triggered after {epoch + 1} epochs')
               #model.load_state_dict(best_model_state)
               #break
    
    return train_losses, val_losses, train_accs, val_accs

# Main Function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    filepath = r'Dataset/EEG.machinelearing_data_BRMH.csv'  # Using raw string to handle backslashes
    balanced_data, class_names = load_and_preprocess_data(filepath)
    
    # Prepare features and labels
    X = balanced_data.drop(columns=['specific.disorder']).values
    y = balanced_data['specific.disorder'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create datasets and dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(class_names)
    model = CNNModel(input_size, num_classes).to(device)
    
    # Define loss function and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # Reduced learning rate, added weight decay
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs=50, patience=10)
    
    # Save the trained model weights
    torch.save(model.state_dict(), 'model.pth')
    print("Model weights saved to model.pth")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()