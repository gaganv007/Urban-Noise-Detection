#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:56:12 2023

@author: gagan
"""

import os
import librosa
import numpy as np
import pandas as pd
import random
import folium
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle


URBAN_SOUND_PATH = 'UrbanSound8K/audio'
METADATA_FILE = 'UrbanSound8K/metadata/UrbanSound8K.csv'
SAMPLE_LIMIT = None  
RANDOM_SEED = 42
EPOCHS = 500
BATCH_SIZE = 32

# LOADING DATA
def load_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file)
    return metadata

# FEATURE EXTRACTION
def extract_features(file_path, sr=22050, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=4.0)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error processing file: {file_path}. Error: {e}")
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
            lat = random.uniform(40.70, 40.85)
            lon = random.uniform(-74.02, -73.90)
            geo_data.append({'file': file_name, 'lat': lat, 'lon': lon, 'classID': row['classID']})
    X = np.array(features)
    y = np.array(labels)
    geo_df = pd.DataFrame(geo_data)
    return X, y, geo_df

# TRAINING NEURAL NETWORK
def build_nn_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_nn_model(X, y, epochs, batch_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    num_classes = len(np.unique(y))
    model = build_nn_model(X_train.shape[1], num_classes)
    print(model.summary())
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=1)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return model, X_test, y_test, history

# SAVE AND LOAD MODEL
def save_nn_model(model, filename='nn_urbansound_model.h5'):
    model.save(filename)
    print(f"Neural network model saved to {filename}")

def load_nn_model(filename='nn_urbansound_model.h5'):
    model = tf.keras.models.load_model(filename)
    print(f"Neural network model loaded from {filename}")
    return model

# GEOSPATIAL VISUALIZATION
def create_noise_map(geo_df, predictions, output_html='urban_noise_map.html'):
    colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'purple', 4: 'orange', 
              5: 'darkred', 6: 'lightgray', 7: 'black', 8: 'beige', 9: 'cadetblue'}
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
    print("Extracting features from audio files (this may take some time)...")
    X, y, geo_df = process_dataset(metadata, URBAN_SOUND_PATH, sample_limit=SAMPLE_LIMIT)
    print("Feature extraction complete. Feature matrix shape:", X.shape)
    print("Training neural network model over multiple epochs...")
    model, X_test, y_test, history = train_nn_model(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    save_nn_model(model)
    
    predictions = np.argmax(model.predict(X), axis=1)
    
    geo_df = geo_df.reset_index(drop=True)
    geo_df['pred_class'] = predictions[:len(geo_df)]
    create_noise_map(geo_df, predictions, output_html='urban_noise_map.html')

if __name__ == "__main__":
    main()
