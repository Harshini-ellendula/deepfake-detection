import os
import librosa
import numpy as np
from tqdm import tqdm

DATASET_PATH = "audio_dataset"
FEATURES = []
LABELS = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


print("Processing REAL audio...")

real_path = os.path.join(DATASET_PATH, "real")

for file in tqdm(os.listdir(real_path)):
    file_path = os.path.join(real_path, file)

    try:
        features = extract_features(file_path)
        FEATURES.append(features)
        LABELS.append(0)   # real
    except:
        pass


print("Processing FAKE audio...")

fake_path = os.path.join(DATASET_PATH, "fake")

for file in tqdm(os.listdir(fake_path)):
    file_path = os.path.join(fake_path, file)

    try:
        features = extract_features(file_path)
        FEATURES.append(features)
        LABELS.append(1)   # fake
    except:
        pass


FEATURES = np.array(FEATURES)
LABELS = np.array(LABELS)

np.save("audio_features.npy", FEATURES)
np.save("audio_labels.npy", LABELS)

print("Feature extraction completed")
print("Features shape:", FEATURES.shape)