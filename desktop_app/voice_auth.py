import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import tensorflow as tf
from tensorflow.keras import layers, Model
import requests
import os

def extract_features(audio, fs=16000, zcr_threshold=None, energy_threshold=None):
    mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13, n_fft=int(0.03 * fs), hop_length=int(0.02 * fs))
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=int(0.03 * fs), hop_length=int(0.02 * fs))[0]
    pitch, mag = librosa.piptrack(y=audio, sr=fs, n_fft=int(0.03 * fs), hop_length=int(0.02 * fs))
    energy = librosa.feature.rms(y=audio, frame_length=int(0.03 * fs), hop_length=int(0.02 * fs))[0]

    pitch_values = []
    for t in range(mag.shape[1]):
        max_idx = mag[:, t].argmax()
        pitch_values.append(pitch[max_idx, t])
    pitch_values = np.array(pitch_values)

    if zcr_threshold is None:
        zcr_threshold = np.mean(zcr)
    if energy_threshold is None:
        energy_threshold = np.mean(energy)

    voiced_frames = [i for i in range(len(zcr)) if zcr[i] < zcr_threshold and energy[i] > energy_threshold]

    voiced_mfccs = mfccs[:, voiced_frames]
    voiced_pitch = pitch_values[voiced_frames]
    voiced_features = np.concatenate((voiced_mfccs, np.expand_dims(voiced_pitch, axis=0)), axis=0)

    return voiced_features

def feature_extraction(audio_file_name, fs=16000, max_frames=80000, zcr_threshold=None, energy_threshold=None):
    audio, _ = sf.read(audio_file_name)
    audio = audio.astype(np.float32)
    
    audio=audio[fs*2:]
    voiced_features = extract_features(audio, fs, zcr_threshold, energy_threshold)
    
    voiced_features = voiced_features.T  # Shape: (n_frames, n_features)

    n_frames, n_features = voiced_features.shape
    if n_frames < max_frames:
        padding = np.zeros((max_frames - n_frames, n_features))
        voiced_features = np.vstack((voiced_features, padding))
    else:
        voiced_features = voiced_features[:max_frames, :]

    return voiced_features

def create_base_network(input_shape):
    model = tf.keras.Sequential([
        layers.Conv1D(64, kernel_size=5, activation=None, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=5, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=5, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.GlobalMaxPooling1D(),

        layers.Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.5),

        layers.Dense(64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.5)
    ])
    return model

def create_siamese_network(input_shape):
    base_network = create_base_network(input_shape)
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    embedding_a = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding_a)
    embedding_b = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding_b)

    distance = layers.Lambda(lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True) + 1e-8))([embedding_a, embedding_b])

    return Model(inputs=[input_a, input_b], outputs=distance)

def contrastive_loss(y_true, y_pred, margin=1):
    """Contrastive loss as a custom loss function."""
    y_true = tf.cast(y_true, tf.float32)
    squared_distance = tf.square(y_pred)
    margin_distance = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean((1 - y_true) * squared_distance + y_true * margin_distance)

def load_model(model_path):
    model = create_siamese_network((80000, 14))
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get('https://drive.google.com/uc?export=download&id=1Y2uRY-Ddb49482xn9H28aqzqdcqNt9kN')
        with open(model_path, 'wb') as file:
            file.write(response.content)
        model.load_weights(model_path)
    return model

def verify_speaker(new_features, original_feature_path, model_path):
    features = np.load(original_feature_path)
    features = np.expand_dims(features, axis=0)
    new_features = np.expand_dims(new_features, axis=0)
    
    model = load_model(model_path)
    prediction = model.predict([new_features, features], verbose=0)
    print(prediction)
    prediction = 0 if round(prediction[0][0],3) <= 0.005 else 1
    return prediction