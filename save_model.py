import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or whatever model you used
import joblib

# Load your training data
# Replace 'path_to_your_training_data.csv' with your actual training data file
data = pd.read_csv('Dataset/EEG.machinelearing_data_BRMH.csv')

# Print column names to see what we're working with
print("Available columns in the CSV file:")
print(data.columns.tolist())

# Assuming your data has columns: timestamp, ch1, ch2, ch3, label
# Adjust column names as needed
X = data[['ch1', 'ch2', 'ch3']].values
y = data['label'].values  # Replace 'label' with your actual label column name

# Create and fit the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
# Replace RandomForestClassifier with your actual model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Print model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.3f}")
print(f"Testing accuracy: {test_score:.3f}")

# Save the model and scaler
joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved successfully!") 