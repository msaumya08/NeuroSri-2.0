# Integrated NeuroSri: EEG-Based Mental Health Assistant

A comprehensive mental health support system that combines real-time EEG analysis with advanced AI to detect mental states, identify potential disorders, and provide personalized counseling.

## Overview

Integrated NeuroSri combines a powerful EEG processing engine with the NeuroSri-2.0 chatbot interface to create a complete mental health support system. The platform analyzes brain waves in real-time to detect emotional states and potential mental disorders, then provides appropriate counseling responses through a modern web interface.

## Key Features

- **Real-time EEG Analysis**: Processes EEG signals to detect mental states and emotional patterns
- **Mental State Classification**: Identifies various mental states including stress, anxiety, focus, relaxation
- **Mental Disorder Detection**: Analyzes brain activity patterns to identify potential mental health disorders
- **Screening Questionnaires**: Provides clinically-based screening tools for various mental disorders
- **Personalized Counseling**: Offers tailored responses based on detected mental states
- **Healing Techniques**: Recommends evidence-based techniques for managing detected conditions
- **Alert System**: Notifies users when concerning patterns are detected
- **Modern Web Interface**: Clean, responsive UI for interacting with the system

## System Architecture

The integrated system combines several components:

1. **EEG Processing Engine**: Analyzes EEG data to detect mental states and potential disorders
2. **NeuroSri-2.0 Chatbot**: Provides the therapeutic chat interface and conversational AI
3. **Questionnaire System**: Offers disorder-specific screening tools and assessments
4. **EEG-Chatbot Bridge**: Connects the EEG processing system with the chatbot interface
5. **Frontend Web Interface**: Provides a user-friendly way to interact with the system

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm (for frontend)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone this repository to your local machine.

2. On Windows, you can simply run the setup batch file which will create a virtual environment and install all dependencies:
   ```
   start_neurosri.bat
   ```

3. For manual installation:
   - Create a virtual environment:
     ```
     python -m venv venv
     ```
   - Activate the virtual environment:
     ```
     # On Windows
     venv\Scripts\activate
     
     # On Unix/MacOS
     source venv/bin/activate
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - Install frontend dependencies:
     ```
     cd frontend
     npm install
     ```

## Running the System

### Using the Batch File (Windows)

Simply double-click `start_neurosri.bat` or run it from the command line:
```
start_neurosri.bat
```

### Manual Start

1. Activate your virtual environment if not already activated.

2. Run the integrated system:
   ```
   python start_integrated_system.py
   ```

3. The system will:
   - Start the EEG processing engine
   - Launch the NeuroSri-2.0 backend
   - Start the questionnaire API
   - Start the EEG-Chatbot bridge
   - Launch the frontend web interface
   - Open your browser to the web interface (http://localhost:5173)

## Usage

1. Connect an EEG device, or if no device is available, the system will simulate EEG data.

2. Interact with NeuroSri through the web interface.

3. The system will:
   - Analyze your EEG data in real-time
   - Display your current mental state and confidence level
   - Provide personalized counseling based on your mental state
   - Alert you if potential mental health issues are detected
   - Offer appropriate healing techniques and coping strategies
   - Provide screening questionnaires when relevant

## Troubleshooting

If you encounter issues:

1. Check the log files:
   - `integrated_system.log`: Main integration logs
   - `connector.log`: EEG-Chatbot bridge logs
   - `questionnaire_api.log`: Questionnaire API logs

2. Ensure all required files exist:
   - `model.pth`: EEG classification model
   - `scaler.joblib`: Feature scaler for EEG data
   - `eeg_data.csv`: EEG data file (created automatically if not present)

3. Common Issues:
   - **Frontend not loading**: Ensure Node.js and npm are installed correctly
   - **EEG device not detected**: Check device connection or use simulated data
   - **Backend errors**: Check dependency installation and Python version

## Important Notes and Disclaimer

This system is designed as an assistive tool and not a replacement for professional mental health services. The system:

- Does not provide medical diagnoses
- Should be used as a supplementary tool alongside professional care
- May have varying accuracy depending on EEG data quality and individual differences

Always consult with qualified healthcare professionals for proper diagnosis and treatment of mental health conditions. 