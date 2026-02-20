import librosa
import numpy as np
import os
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ignore warnings about MP3 decoding
warnings.filterwarnings("ignore")

def extract_features(file_path):
    try:
        # librosa handles .mp3 automatically if audioread is installed
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Skipping {file_path} - Error: {e}")
        return None

def train():
    print("Extracting features from MP3 files... (This might take a few minutes)")
    DATA_DIR = "./dataset/"
    CATEGORIES = ["ambulance", "police", "horn", "background"]
    
    X, y = [], []
    for category in CATEGORIES:
        folder_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(folder_path): 
            print(f"Waiting for folder: {folder_path}")
            continue
            
        # TWEAK: Changed .wav to .mp3 here!
        for file in os.listdir(folder_path):
            if file.endswith(".mp3"):
                file_path = os.path.join(folder_path, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(category)

    if not X:
        print("No MP3 data found! Make sure your files are inside dataset/ambulance, etc.")
        return

    print(f"Successfully processed {len(X)} audio files. Training model now...")

    # Train the Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Edge Model Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
    
    # Save the model
    joblib.dump(model, "soundsplit_ai_model.pkl")
    print("SUCCESS: Model saved locally as 'soundsplit_ai_model.pkl'")

if __name__ == "__main__":
    train()