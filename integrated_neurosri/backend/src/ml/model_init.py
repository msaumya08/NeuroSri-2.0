import os
import requests
from pathlib import Path
import torch
import logging
import numpy as np

# Remove circular import
class EEGEmotionNet(torch.nn.Module):
    def __init__(self, input_size=110):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/your-model/emotion_model/resolve/main/emotion_model.pth"  # Replace with actual URL
FALLBACK_MODEL_URL = "https://drive.google.com/uc?export=download&id=your-model-id"  # Replace with backup URL

def download_model(model_path: Path) -> bool:
    """Download the emotion model if it doesn't exist"""
    try:
        if not model_path.exists():
            logger.info(f"Downloading emotion model to {model_path}")
            os.makedirs(model_path.parent, exist_ok=True)
            
            # Try primary URL
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
            except:
                # Try fallback URL
                logger.warning("Primary download failed, trying fallback URL")
                response = requests.get(FALLBACK_MODEL_URL, stream=True)
                response.raise_for_status()
            
            # Save the model
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify the model
            try:
                torch.load(model_path, map_location='cpu')
                logger.info("Model downloaded and verified successfully")
                return True
            except:
                logger.error("Downloaded model is corrupt")
                if model_path.exists():
                    model_path.unlink()
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def create_default_model(model_path: Path) -> bool:
    """Create a default model if download fails"""
    try:
        logger.info("Creating default model...")
        model = EEGEmotionNet()
        
        # Initialize with some default weights
        for param in model.parameters():
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        
        # Save the model
        os.makedirs(model_path.parent, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info("Default model created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating default model: {e}")
        return False

def initialize_model(model_dir: Path = Path("models")) -> Path:
    """Initialize the emotion model"""
    model_path = model_dir / "emotion_model.pth"
    
    # If model exists, return it
    if model_path.exists():
        logger.info("Using existing model")
        return model_path
        
    # Try to create default model
    if create_default_model(model_path):
        return model_path
        
    raise RuntimeError("Failed to initialize emotion model") 