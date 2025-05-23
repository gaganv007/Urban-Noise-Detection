#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 18:22:49 2023

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
    metadata = pd.read_csv(metadata_file)
    return metadata

# FEATURE EXTRACTION
def extract_features(file_path, sr=22050, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=4.0)  # duration limited to 4 seconds
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {e}")
        return None
    return mfccs_mean

def process_dataset(metadata, base_audio_path, sample_limit=None):
    features = []
    labels = []
    geo_data = []  

    if sample_limit is not None:
        metadata = metadata.sample(n=sample_limit, random_state=RANDOM_SEED)

    for index, row in metadata.iterrows():
        fold = row['fold']
        file_name = row['slice_file_name']
        file_path = os.path.join(base_audio_path, f"fold{fold}", file_name)
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append(row['classID'])  
            lat = random.uniform(lat_range[0], lat_range[1])
            lon = random.uniform(lon_range[0], lon_range[1])
            geo_data.append({'file': file_name, 'lat': lat, 'lon': lon, 'classID': row['classID']})
    X = np.array(features)
    y = np.array(labels)
    geo_df = pd.DataFrame(geo_data)
    return X, y, geo_df

# MODEL TRAINING
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf, X_test, y_test

# SAVE AND LOAD MODEL 
def save_model(model, filename='rf_urbansound_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename='rf_urbansound_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

# GEOSPATIAL VISUALIZATION
def create_noise_map(geo_df, predictions, output_html='noise_map.html'):
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 5: 'darkred', 6: 'lightgray', 7: 'black', 8: 'beige', 9: 'cadetblue'}
    avg_lat = geo_df['lat'].mean()
    avg_lon = geo_df['lon'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    for idx, row in geo_df.iterrows():
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
    m.save(output_html)
    print(f"Noise map saved to {output_html}")

def main():
    metadata = load_metadata(METADATA_FILE)
    print("Metadata loaded. Total samples:", len(metadata))
    print("Extracting features from audio files...")
    X, y, geo_df = process_dataset(metadata, URBAN_SOUND_PATH, sample_limit=SAMPLE_LIMIT)
    print("Feature extraction complete. Feature matrix shape:", X.shape)
    print("Training classifier...")
    model, X_test, y_test = train_model(X, y)
    save_model(model)

    predictions = model.predict(X)

    geo_df = geo_df.reset_index(drop=True)
    geo_df['pred_class'] = predictions[:len(geo_df)]  
    
    create_noise_map(geo_df, predictions, output_html='urban_noise_map.html')

if __name__ == "__main__":
    main()
