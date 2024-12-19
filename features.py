import librosa
import os
import numpy as np

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
labels = np.arange(len(emotions))

def to_categorical(labels, num_classes):
    return np.eye(num_classes)[labels]
def load_dataset():
    X = []
    y = []
    for root, dirs, files in os.walk('speech_emotion'):
        for file in files:
            if file.endswith('.wav'):
                audio, sr = librosa.load(os.path.join(root, file))
                mfccs = librosa.feature.mfcc(sr=sr,y=audio)
                X.append(mfccs.mean(axis=1))
                filename = file.split('-')
                emotion = int(filename[2])
                y.append(emotion - 1)
    X = np.array(X)
    y = np.array(y)
    return X, y
