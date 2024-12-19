import librosa
from model import train_model
from features import load_dataset
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from joblib import load

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

x,y = load_dataset()
model,accuracy = train_model(x,y)
print("Accuracy  :  ",accuracy)

model = load('ser_rf.joblib')

def make_prediction(audio_file, model):
    audio, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(sr=sr, y=audio)
    mfccs = mfccs.mean(axis=1)
    mfccs = mfccs.reshape(1, -1)  # Ensure it's a 2D array for prediction

    # Get prediction probabilities
    prediction_probabilities = model.predict_proba(mfccs)

    # Debugging: Check the shape of prediction probabilities
    print(f"Prediction Probabilities Shape: {prediction_probabilities.shape}")  # Should be (1, num_classes)

    if prediction_probabilities.ndim == 2:
        predicted_index = np.argmax(prediction_probabilities, axis=1)[0]  # Get the index of the highest probability
        predicted_emotion = emotions[predicted_index]  # Emotion corresponding to the index
        intensity = prediction_probabilities[0][predicted_index]  # Confidence score for the predicted emotion
        return predicted_emotion, intensity
    else:
        raise ValueError("Prediction probabilities should be a 2D array.")


# Example usage
predicted_emotion, intensity = make_prediction(
    'Actor_03/03-01-01-01-01-01-03.wav',
    model)
print(f"Predicted Emotion: {predicted_emotion}, Intensity: {intensity:.2f}")