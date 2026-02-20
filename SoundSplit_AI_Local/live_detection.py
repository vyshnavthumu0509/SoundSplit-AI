import librosa
import numpy as np
import joblib

def load_and_predict(audio_file_path):
    """
    Loads the saved SoundSplit AI model and predicts a new, unseen sound.
    """
    # 1. Load the trained model
    try:
        model = joblib.load("soundsplit_ai_model.pkl")
    except FileNotFoundError:
        print("Error: Model not found. Please run the training script first.")
        return
    
    # 2. Extract features from the new incoming sound
    # (Just like we did during training)
    audio, sample_rate = librosa.load(audio_file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Reshape for a single prediction
    features = mfccs_scaled.reshape(1, -1)
    
    # 3. Predict!
    prediction = model.predict(features)[0]
    
    # Optional: Get the confidence probabilities
    probabilities = model.predict_proba(features)[0]
    confidence_score = max(probabilities) * 100
    
    print(f"Sound Detected: {prediction.upper()}")
    print(f"AI Confidence:  {confidence_score:.2f}%\n")
    
    # The Cloud would now send this 'prediction' string back to the car's local hub
    return prediction

# ==========================================
# Test it with a fake transmission from the car
# ==========================================
if __name__ == "__main__":
    print("Incoming data from SoundSplit Local Hub...")
    # Replace this with the path of an audio clip you want to test
    test_file = "test_siren_from_highway.wav" 
    
    # If the file exists, predict it
    import os
    if os.path.exists(test_file):
        load_and_predict(test_file)
    else:
        print(f"Please provide a valid audio file named '{test_file}' to test.")