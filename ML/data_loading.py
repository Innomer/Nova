import os
import numpy as np
import soundfile as sf
import librosa
import pickle

def convert_flac_to_numpy(base_dir, max_frames=250):
    """
    Convert FLAC files to numpy arrays with features of uniform shape.
    
    Args:
        base_dir (str): Base directory containing LibriSpeech dataset.
        max_frames (int): Maximum number of frames to pad/truncate to.
    """
    data = []
    labels = []
    len_features = []
    
    part_number = 0
    for folder in os.listdir(base_dir):
        if len(folder.split('.')) > 1:  # Skip files
            continue
        print("Processing folder:", folder)
        for subfolder in os.listdir(os.path.join(base_dir, folder)):
            speaker_id = subfolder
            print("Processing speaker:", speaker_id)
            for subsubfolder in os.listdir(os.path.join(base_dir, folder, subfolder)):
                for file in os.listdir(os.path.join(base_dir, folder, subfolder, subsubfolder)):
                    if file.endswith('.flac'):
                        audio_path = os.path.join(base_dir, folder, subfolder, subsubfolder, file)
                        audio, fs = sf.read(audio_path)
                        
                        # Extract features
                        voiced_features = extract_features_from_audio(audio, fs, max_frames=max_frames)
                        data.append(voiced_features)
                        labels.append(speaker_id)
            
            # Save data in chunks to avoid memory issues
            if len(data) > 1000:
                pickle.dump(data, open(f'dataset/features_part_{part_number}.pkl', 'wb'))
                pickle.dump(labels, open(f'dataset/labels_part_{part_number}.pkl', 'wb'))
                print(f'Saved part {part_number}')
                
                len_features.append(len(data))
                data.clear()
                labels.clear()
                part_number += 1
    
    # Save metadata
    pickle.dump(len_features, open('dataset/feature_length.pkl', 'wb'))
    print('Data and labels saved to disk')

def extract_features(audio, fs=16000, zcr_threshold=None, energy_threshold=None):
    """
    Extract MFCCs, ZCR, Pitch, and Energy features from audio.
    """
    # Extract features
    mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13, n_fft=int(0.03 * fs), hop_length=int(0.02 * fs))
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=int(0.03 * fs), hop_length=int(0.02 * fs))[0]
    pitch, mag = librosa.piptrack(y=audio, sr=fs, n_fft=int(0.03 * fs), hop_length=int(0.02 * fs))
    energy = librosa.feature.rms(y=audio, frame_length=int(0.03 * fs), hop_length=int(0.02 * fs))[0]

    # Extract pitch from highest magnitude bins
    pitch_values = []
    for t in range(mag.shape[1]):  # Iterate over time frames
        max_idx = mag[:, t].argmax()  # Index of max magnitude in frequency bins
        pitch_values.append(pitch[max_idx, t])
    pitch_values = np.array(pitch_values)

    # Compute thresholds if not provided
    if zcr_threshold is None:
        zcr_threshold = np.mean(zcr)  # Adaptive threshold for ZCR
    if energy_threshold is None:
        energy_threshold = np.mean(energy)  # Adaptive threshold for energy

    # Detect voiced frames
    voiced_frames = [i for i in range(len(zcr)) if zcr[i] < zcr_threshold and energy[i] > energy_threshold]

    # Get voiced features
    voiced_mfccs = mfccs[:, voiced_frames]
    voiced_pitch = pitch_values[voiced_frames]

    # Combine MFCCs and pitch
    voiced_features = np.concatenate((voiced_mfccs, np.expand_dims(voiced_pitch, axis=0)), axis=0)

    return voiced_features

def extract_features_from_audio(audio, fs=16000, max_frames=250, zcr_threshold=0.1, energy_threshold=0.1):
    """
    Extract features from an audio signal and ensure uniform shape.
    """
    # Extract voiced features
    voiced_features = extract_features(audio, fs, zcr_threshold, energy_threshold)

    # Transpose to (n_frames, n_features)
    voiced_features = voiced_features.T  # Shape: (n_frames, n_features)

    # Pad or truncate to ensure uniform frame length
    n_frames, n_features = voiced_features.shape
    if n_frames < max_frames:
        # Pad with zeros
        padding = np.zeros((max_frames - n_frames, n_features))
        voiced_features = np.vstack((voiced_features, padding))
    else:
        # Truncate to max_frames
        voiced_features = voiced_features[:max_frames, :]

    return voiced_features

if __name__ == '__main__':
    base_dir = 'LibriSpeech'
    convert_flac_to_numpy(base_dir)