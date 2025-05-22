# NeuroSri: Integrated EEG Mental Health Analysis System

NeuroSri is an integrated system that combines EEG signal processing, machine learning prediction, and a chatbot interface to provide mental health support based on real-time brain activity analysis.

## Key Features

- **Real-time EEG Analysis**: Processes EEG signals to detect mental states and emotional patterns
- **Mental State Classification**: Identifies various mental states including stress, anxiety, focus, relaxation, and more
- **Mental Disorder Detection**: Analyzes brain activity patterns to identify potential mental health disorders
- **Interactive Questionnaires**: Provides clinically-based screening tools for mental health disorders
- **Personalized Counseling**: Offers tailored responses based on detected mental states
- **Healing Techniques**: Recommends evidence-based techniques for managing detected conditions
- **Alert System**: Notifies users when concerning patterns are detected

## System Architecture

The NeuroSri system consists of the following components:

1. **EEG Data Processing** (`realtime_prediction.py`): Analyzes EEG signals in real-time to predict mental states and detect potential disorders.
2. **Chatbot Server** (`Chatbot/NeuroSri-2.0/server.py`): Provides the backend API for the chatbot interface.
3. **Connector** (`eeg_to_chatbot_connector.py`): Bridges the EEG processing and chatbot components.
4. **Questionnaire API** (`questionnaire_api.py`): Handles interactive mental health screening questionnaires.
5. **Disorder Questionnaire Module** (`disorder_questionnaire.py`): Contains standardized assessment tools for various mental health disorders.
6. **Web Frontend** (`Chatbot/NeuroSri-2.0/frontend`): User interface for interacting with the system.
7. **System Launcher** (`start_system.py`): Coordinates the startup of all components.

## Mental Disorder Detection

NeuroSri can detect patterns associated with various mental health disorders, including:

- **Generalized Anxiety Disorder**: Excessive worry and anxiety about everyday events
- **Panic Disorder**: Recurring panic attacks with physical symptoms
- **Major Depressive Disorder**: Persistent sadness and loss of interest
- **Persistent Depressive Disorder**: Long-term depression symptoms
- **OCD (Obsessive-Compulsive Disorder)**: Unwanted thoughts and repetitive behaviors
- **ADHD**: Difficulty maintaining attention and hyperactivity/impulsivity
- **Executive Function Disorder**: Challenges with planning, organizing, and completing tasks
- **Cognitive Impairment**: Changes in memory, thinking, or problem-solving abilities

When potential disorder patterns are detected in the EEG signal with sufficient confidence, NeuroSri will:

1. Alert the user about the potential pattern
2. Ask targeted questions related to the potential disorder
3. Offer the option to complete a standardized screening questionnaire
4. Provide appropriate healing techniques and coping strategies
5. Suggest professional consultation when appropriate

## Healing Techniques

NeuroSri offers evidence-based techniques for different mental states and potential disorders, including:

- **Anxiety Management**: Breathing exercises, grounding techniques, cognitive restructuring
- **Depression Support**: Behavioral activation, gratitude practices, thought reframing
- **ADHD Strategies**: Task chunking, environmental modifications, implementation intentions
- **OCD Techniques**: Exposure and response prevention (ERP), mindful observation
- **Stress Reduction**: Progressive muscle relaxation, guided meditation, mindfulness practices

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm
- Required Python packages (listed in `requirements.txt`)

### Installation Steps

1. Clone this repository to your local machine.

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```
   cd Chatbot/NeuroSri-2.0
   npm install
   ```

4. Train the model (if not already trained):
   ```
   python train_model.py
   ```
   This will generate `model.pth` and `scaler.joblib` files required for prediction.

## Running the System

The entire system can be started with a single command:

```
python start_system.py
```

Alternatively, on Windows, you can use:

```
start_neurosri.bat
```

This script will:
1. Check for all required dependencies
2. Start the EEG processing component
3. Start the chatbot server
4. Start the connector between EEG processing and chatbot
5. Start the questionnaire API server
6. Launch the web frontend

Once started, you can access the web interface at: [http://localhost:3000](http://localhost:3000)

## Data Flow

1. EEG data is collected and stored in `eeg_data.csv`
2. `realtime_prediction.py` analyzes this data and makes predictions, including potential disorder detection
3. Predictions are saved to `prediction_output.json`
4. The connector reads this file and sends data to the chatbot server
5. When potential disorders are detected, the system can initiate questionnaires via the questionnaire API
6. Questionnaire results are saved in the `questionnaire_results` directory
7. The frontend displays mental state information, chatbot responses, and questionnaires

## Troubleshooting

If you encounter issues:

1. Check the log files:
   - `integration.log`: Main integration logs
   - `connector.log`: Connector logs
   - `questionnaire_api.log`: Questionnaire API logs
   - `server.log`: Chatbot server logs

2. Ensure all required files exist:
   - `model.pth`: Trained model file
   - `scaler.joblib`: Scaler for feature normalization
   - `eeg_data.csv`: EEG data file

## Important Notes and Disclaimer

NeuroSri is designed as an assistive tool and not a replacement for professional mental health services. The system:

- Does not provide medical diagnoses
- Should be used as a supplementary tool alongside professional care
- May have varying accuracy depending on EEG data quality and individual differences

Always consult with qualified healthcare professionals for proper diagnosis and treatment of mental health conditions.

## License

[Your license information]

## Contact

[Your contact information]

## System Integration Updates

### Integrated Server Architecture
The system now uses an integrated server architecture that combines:
- BLE EEG device connectivity
- EEG data processing and analysis
- Chatbot functionality
- Mental state prediction

### Key Features
- **Direct BLE Connection**: Connect directly to NPG BLE EEG devices
- **Real-time Visualization**: View EEG data through a web interface
- **Mental State Analysis**: Analyze brain waves to detect mental states and potential disorders
- **Therapeutic Responses**: Generate personalized therapeutic responses based on EEG and conversation data
- **Smart Port Detection**: Automatically finds available ports to avoid conflicts

### How to Use
1. Connect your NPG BLE EEG device and ensure it's powered on
2. Run `start_neurosri.bat` to initialize the environment and launch the system
3. The integrated server will automatically scan for and connect to your EEG device
4. Access the chat interface at http://localhost:3000 (or another available port)
5. Access the EEG visualization at http://localhost:5000/api/visualize-eeg (or another available port)

### API Endpoints
- **/api/raw-eeg**: Get raw EEG data in JSON format
- **/api/emotion**: Get current emotion/mental state based on EEG analysis
- **/api/analysis**: Get detailed mental state analysis with therapeutic suggestions
- **/api/chat**: Send and receive chat messages with context-aware responses
- **/api/user-info**: Update user profile information
- **/api/visualize-eeg**: Real-time visualization of EEG data in a web browser 