from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import torch
import pickle
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScalers
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model

import logging

# Initialize Flask App
app = Flask(__name__)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load pre-trained scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load pre-trained KMeans model
with open('knn_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the autoencoder model
with open('autoencoder.pkl', 'rb') as f:
    autoencoder = pickle.load(f)  # This should be the trained Autoencoder model
autoencoder.eval()  # Set to evaluation mode

@app.route('/')
def home():
    return "Earthquake Anomaly Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        input_data = np.array([[data['magnitude'], data['depth'], data['latitude'], data['longitude']]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_scaled)

        # Get encoded features
        with torch.no_grad():
            encoded_features = autoencoder.encoder(input_tensor).numpy()

        # Get cluster label
        cluster_label = kmeans.predict(encoded_features)[0]

        # Get reconstruction error
        with torch.no_grad():
            reconstructed_data = autoencoder(input_tensor)
        reconstruction_error = torch.mean((input_tensor - reconstructed_data) ** 2).item()

        # Define anomaly threshold
        threshold = 0.01  # Adjust based on dataset analysis
        anomaly = reconstruction_error > threshold

        return jsonify({
            "cluster": int(cluster_label),
            "reconstruction_error": reconstruction_error,
            "anomaly": bool(anomaly)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
# Setup Logging
logging.basicConfig(level=logging.INFO)

# ====== Load Models & Scalers ======
custom_objects = {"mse": MeanSquaredError()}
lstm_model = load_model("cyclone_lstm_model.h5", custom_objects=custom_objects)  # LSTM for path prediction
speed_model = joblib.load("speed_model.pkl")        # XGBoost for speed
dir_model = joblib.load("dir_model.pkl")            # XGBoost for direction
scaler_X = joblib.load("scaler_X.pkl")              # Scaler for LSTM input
scaler_y = joblib.load("scaler_y.pkl")              # Scaler for LSTM output

# Load trained severity classification models
encoder = load_model("severity_encoder.h5")          # Trained encoder model
scaler_severity = joblib.load("severity_scaler.pkl") # Scaler for severity
kmeans = joblib.load("severity_kmeans.pkl")                   # KMeans clustering (Fixed the previous issue)

# Define Severity Labels
severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}

# ====== Preprocessing Function ======
def preprocess_input(data, task='path'):
    """Preprocess input JSON data for LSTM or ML models."""
    df = pd.DataFrame([data])

    # Convert time to datetime
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')

    # Extract features
    df['HOUR'] = df['ISO_TIME'].dt.hour
    df['MONTH'] = df['ISO_TIME'].dt.month
    df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
    df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))

    # Include interaction terms
    if 'lat_lon_interaction' not in df:
        df['lat_lon_interaction'] = df['LAT'] * df['LON']
    if 'speed_lat_interaction' not in df:
        df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
    if 'speed_lon_interaction' not in df:
        df['speed_lon_interaction'] = df['STORM_SPEED'] * df['LON']

    features = [
        'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos',
        'lat_lon_interaction', 'speed_lat_interaction', 'speed_lon_interaction'
    ]

    X = df[features].values

    # Scale input for LSTM
    if task == 'path':
        X = scaler_X.transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))  # LSTM format

    return X

def preprocess_xgboost_input(data):
    """Preprocess input JSON for XGBoost-based models (speed, direction)."""
    df = pd.DataFrame([data])

    # Convert ISO_TIME to datetime
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['HOUR'] = df['ISO_TIME'].dt.hour
    df['MONTH'] = df['ISO_TIME'].dt.month

    # Compute direction encodings
    df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
    df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))

    # Interaction Features
    df['lat_lon_interaction'] = df['LAT'] * df['LON']
    df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']

    # Default values for lag/moving average (can be replaced with real data)
    df['STORM_SPEED_LAG1'] = df['STORM_SPEED']
    df['LAT_LAG'] = df['LAT']
    df['LON_LAG'] = df['LON']
    df['SPEED_MA3'] = df['STORM_SPEED']

    # Features for XGBoost (Fixed: Removed 'speed_lon_interaction')
    features = [
        'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH',
        'dir_sin', 'dir_cos', 'STORM_SPEED_LAG1', 'LAT_LAG', 'LON_LAG', 'SPEED_MA3',
        'lat_lon_interaction', 'speed_lat_interaction'
    ]

    X = df[features].values
    return X


    

# ====== API Routes ======

@app.route('/predict-path', methods=['POST'])
def predict_path():
    """Predict cyclone path using LSTM."""
    try:
        data = request.get_json()
        X = preprocess_input(data, task='path')

        if X is None:
            return jsonify({'error': 'Invalid input data'}), 400

        y_pred = lstm_model.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred)

        return jsonify({'predicted_lat_lon': y_pred.tolist()})
    except Exception as e:
        logging.error(f"Error in predict_path: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict-speed', methods=['POST'])
def predict_speed():
    """Predict storm speed and direction using XGBoost."""
    try:
        data = request.get_json()
        X = preprocess_xgboost_input(data)

        if X.shape[1] != 13:  # Ensure the correct number of features
            return jsonify({'error': f'Feature shape mismatch, expected 13, got {X.shape[1]}'}), 400

        speed_pred = speed_model.predict(X)
        dir_pred = dir_model.predict(X)

        return jsonify({
            'predicted_speed': speed_pred.tolist(),
            'predicted_direction': dir_pred.tolist()
        })
    except Exception as e:
        logging.error(f"Error in predict_speed: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500



@app.route('/classify-severity', methods=['POST'])
def classify_severity():
    """Classify cyclone severity using autoencoder & KMeans."""
    try:
        data = request.get_json()
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])

        # Preprocess input for severity classification
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df['HOUR'] = df['ISO_TIME'].dt.hour.fillna(0)
        df['MONTH'] = df['ISO_TIME'].dt.month.fillna(0)
        df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
        df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))

        features = ['LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos']
        X = df[features].values

        # ✅ Ensure the scaler is fitted
        if not hasattr(scaler_severity, 'mean_'):
            return jsonify({'error': 'Severity scaler is not fitted. Train the model first.'}), 500
        
        X_scaled = scaler_severity.transform(X)

        # ✅ Ensure correct input shape for Autoencoder
        expected_features = encoder.input_shape[1]  # Get expected feature count
        if X_scaled.shape[1] != expected_features:
            return jsonify({'error': f'Feature shape mismatch, expected {expected_features}, got {X_scaled.shape[1]}'}), 400
        
        # Extract latent features from autoencoder
        latent_features = encoder.predict(X_scaled)

        # ✅ Ensure severity labels exist
        if 'severity_labels' not in globals():
            return jsonify({'error': 'Severity labels mapping is missing.'}), 500

        # Predict severity cluster
        cluster_labels = kmeans.predict(latent_features)
        df['Severity'] = [severity_labels.get(c, "Unknown") for c in cluster_labels]

        return jsonify(df[['LAT', 'LON', 'STORM_SPEED', 'Severity']].to_dict(orient='records'))
    
    except Exception as e:
        logging.error(f"Error in classify_severity: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# ====== Run the Flask App ======
if __name__ == '__main__':
    app.run(debug=True)
