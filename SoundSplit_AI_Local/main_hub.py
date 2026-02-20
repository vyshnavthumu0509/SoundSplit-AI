import numpy as np
import librosa
import joblib
import time
from scipy.fft import rfft, irfft

# ---------------------------------------------------------
# 1. MATHEMATICS: DIRECTION DETECTION (TDOA)
# ---------------------------------------------------------
def gcc_phat(sig1, sig2, sample_rate=44100):
    n = len(sig1) + len(sig2)
    SIG1 = rfft(sig1, n=n)
    SIG2 = rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.fftshift(irfft(R / (np.abs(R) + 1e-15)))
    shift = np.argmax(np.abs(cc)) - (n // 2)
    return shift / sample_rate

def detect_direction(fl, fr, rl, rr, sr=44100):
    delay_lr = (gcc_phat(fl, fr, sr) + gcc_phat(rl, rr, sr)) / 2
    delay_fb = (gcc_phat(fl, rl, sr) + gcc_phat(fr, rr, sr)) / 2
    threshold = 0.0001 
    
    horizontal = "Right" if delay_lr > threshold else "Left" if delay_lr < -threshold else "Center"
    vertical = "Rear" if delay_fb > threshold else "Front" if delay_fb < -threshold else "Middle"
    return f"{vertical}-{horizontal}"

# ---------------------------------------------------------
# 2. MACHINE LEARNING: SOUND CLASSIFICATION
# ---------------------------------------------------------
# Load the AI model into the hardware's memory on startup
try:
    local_ai_model = joblib.load("soundsplit_ai_model.pkl")
except FileNotFoundError:
    print("Error: soundsplit_ai_model.pkl not found! Run train_local_model.py first.")
    exit()

def classify_sound(audio_array, sample_rate):
    """Extracts features from the raw array and classifies it."""
    # We use librosa to resample the raw array to 22050Hz for the model
    audio_resampled = librosa.resample(y=audio_array, orig_sr=sample_rate, target_sr=22050)
    mfccs = librosa.feature.mfcc(y=audio_resampled, sr=22050, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)
    
    prediction = local_ai_model.predict(features)[0]
    return prediction.upper()

# ---------------------------------------------------------
# 3. THE UNIFIED PIPELINE (Runs every 1 second)
# ---------------------------------------------------------
def process_environment(fl_mic, fr_mic, rl_mic, rr_mic, sr=44100):
    start_time = time.perf_counter()
    
    # 1. Where is it coming from?
    direction = detect_direction(fl_mic, fr_mic, rl_mic, rr_mic, sr)
    
    # 2. What is it? (We just need to analyze one mic's audio for this)
    classification = classify_sound(fl_mic, sr)
    
    # 3. The Logic Filter (Don't alert the driver for normal noise)
    process_time = (time.perf_counter() - start_time) * 1000
    
    print("\n--- SoundSplit Hub: Event Detected ---")
    if classification == "BACKGROUND":
        print(f"Status: Ignoring normal street noise. ({process_time:.2f}ms)")
    else:
        # This is where the hardware triggers the Whisper Speaker!
        print(f"ALERT TRIGGERED -> {classification} approaching from {direction}")
        print(f"Local Processing Time: {process_time:.2f} ms")
        print("--------------------------------------")

# ==========================================
# SIMULATION TEST
# ==========================================
if __name__ == "__main__":
    sr = 44100
    
    # Simulating a 1-second blast of noise hitting the 4 mics
    print("System Booted. Microphones active. Processing completely offline...")
    base_sound = np.random.randn(sr) 
    delay_samples = 220 
    
    # Simulate sound hitting the Rear-Right first
    rr_mic = base_sound 
    rl_mic = np.pad(base_sound, (delay_samples, 0))[:-delay_samples]
    fr_mic = np.pad(base_sound, (delay_samples, 0))[:-delay_samples]
    fl_mic = np.pad(base_sound, (delay_samples * 2, 0))[:-(delay_samples * 2)]
    
    process_environment(fl_mic, fr_mic, rl_mic, rr_mic, sr)