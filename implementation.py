#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:22:49 2025

@author: gagan
"""
import os
import librosa
import numpy as np
import pandas as pd
import random
import folium
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle


URBAN_SOUND_PATH = 'UrbanSound8K/audio'
METADATA_FILE = 'UrbanSound8K/metadata/UrbanSound8K.csv'
SAMPLE_LIMIT = None  
RANDOM_SEED = 42

lat_range = (40.70, 40.85)
lon_range = (-74.02, -73.90)

def load_metadata(metadata_file):
    """
    Load the UrbanSound8K metadata CSV into a pandas DataFrame.
    """
    metadata = pd.read_csv(metadata_file)
    return metadata

# --- STEP 2: FEATURE EXTRACTION ---
def extract_features(file_path, sr=22050, n_mfcc=40):
    """
    Load an audio file and extract MFCC features.
    Returns the mean of each MFCC over time.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr, duration=4.0)  # duration limited to 4 seconds
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Take mean across time frames
        mfccs_mean = np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None
    return mfccs_mean

def process_dataset(metadata, base_audio_path, sample_limit=None):
    """
    Iterate through metadata rows to extract features and collect labels.
    Optionally limit the number of samples.
    Returns feature matrix X, labels y, and a DataFrame with simulated geolocations.
    """
    features = []
    labels = []
    geo_data = []  # To store simulated latitude and longitude

    # If sample_limit is set, shuffle metadata and take a subset
    if sample_limit is not None:
        metadata = metadata.sample(n=sample_limit, random_state=RANDOM_SEED)

    for index, row in metadata.iterrows():
        fold = row['fold']
        file_name = row['slice_file_name']
        file_path = os.path.join(base_audio_path, f"fold{fold}", file_name)
        # Extract MFCC features
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append(row['classID'])  # numeric label
            # Simulate geolocation: for demo, assign random lat/lon within a city range
            lat = random.uniform(lat_range[0], lat_range[1])
            lon = random.uniform(lon_range[0], lon_range[1])
            geo_data.append({'file': file_name, 'lat': lat, 'lon': lon, 'classID': row['classID']})
    X = np.array(features)
    y = np.array(labels)
    geo_df = pd.DataFrame(geo_data)
    return X, y, geo_df

# --- STEP 3: MODEL TRAINING ---
def train_model(X, y):
    """
    Split the dataset, train a RandomForest classifier, and return the trained model along with test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    # Evaluate model performance on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf, X_test, y_test

# --- STEP 4: SAVE/LOAD MODEL ---
def save_model(model, filename='rf_urbansound_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename='rf_urbansound_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

# --- STEP 5: GEOSPATIAL VISUALIZATION ---
def create_noise_map(geo_df, predictions, output_html='noise_map.html'):
    """
    Create an interactive Folium map with markers for each audio sample.
    Markers are color-coded based on the predicted noise class.
    """
    # Define a color mapping for classes (you can modify these as needed)
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'darkred', 6: 'lightgray', 7: 'black', 8: 'beige', 9: 'cadetblue'}
    # Initialize map centered at the average location
    avg_lat = geo_df['lat'].mean()
    avg_lon = geo_df['lon'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    # Add markers
    for idx, row in geo_df.iterrows():
        # Use prediction from model; here we assume geo_df order aligns with our processed order.
        pred_class = predictions[idx]
        color = colors.get(pred_class, 'blue')
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            popup=f"File: {row['file']}, Predicted Class: {pred_class}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
    # Save map to HTML file
    m.save(output_html)
    print(f"Noise map saved to {output_html}")

# --- MAIN EXECUTION ---
def main():
    # Step 1: Load metadata
    metadata = load_metadata(METADATA_FILE)
    print("Metadata loaded. Total samples:", len(metadata))
    
    # Step 2: Process dataset to extract features and simulate geolocation
    print("Extracting features from audio files...")
    X, y, geo_df = process_dataset(metadata, URBAN_SOUND_PATH, sample_limit=SAMPLE_LIMIT)
    print("Feature extraction complete. Feature matrix shape:", X.shape)
    
    # Step 3: Train classifier
    print("Training classifier...")
    model, X_test, y_test = train_model(X, y)
    
    # Optionally, save the trained model
    save_model(model)
    
    # For demonstration, predict on the entire dataset to simulate mapping
    predictions = model.predict(X)
    
    # Add predictions to geo_df for mapping
    geo_df = geo_df.reset_index(drop=True)
    geo_df['pred_class'] = predictions[:len(geo_df)]  # align lengths
    
    # Step 4: Create and save an interactive noise map
    create_noise_map(geo_df, predictions, output_html='urban_noise_map.html')

if __name__ == "__main__":
    main()
