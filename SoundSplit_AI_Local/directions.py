import numpy as np
from scipy.fft import rfft, irfft
import time

def gcc_phat(sig1, sig2, sample_rate=44100):
    """Calculates microsecond time delay between two audio signals."""
    n = len(sig1) + len(sig2)
    SIG1 = rfft(sig1, n=n)
    SIG2 = rfft(sig2, n=n)
    
    # Cross-Power Spectrum Phase Transform
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.fftshift(irfft(R / (np.abs(R) + 1e-15)))
    
    shift = np.argmax(np.abs(cc)) - (n // 2)
    return shift / sample_rate

def detect_direction(fl, fr, rl, rr, sr=44100):
    """Determines the quadrant of the sound source."""
    # Compare Left vs Right sides
    delay_lr = (gcc_phat(fl, fr, sr) + gcc_phat(rl, rr, sr)) / 2
    # Compare Front vs Back sides
    delay_fb = (gcc_phat(fl, rl, sr) + gcc_phat(fr, rr, sr)) / 2

    threshold = 0.0001 # 0.1 milliseconds
    
    horizontal = "Right" if delay_lr > threshold else "Left" if delay_lr < -threshold else "Center"
    vertical = "Rear" if delay_fb > threshold else "Front" if delay_fb < -threshold else "Middle"

    return f"{vertical}-{horizontal}"

# --- 1-DAY EXPERIMENT SIMULATION ---
if __name__ == "__main__":
    sr = 44100
    base_sound = np.random.randn(sr) # Simulate a 1-second horn blast
    
    # Simulate a sound coming from the REAR-LEFT
    # Speed of sound is 343 m/s. Car dimensions cause slight delays to each mic.
    delay_samples = 220 
    
    print("Testing Sound Location: Rear-Left")
    start_time = time.perf_counter()
    
    # Rear Left gets it first (no delay)
    rl_mic = base_sound 
    # Rear Right gets it later (delayed by car width)
    rr_mic = np.pad(base_sound, (delay_samples, 0))[:-delay_samples]
    # Front Left gets it later (delayed by car length)
    fl_mic = np.pad(base_sound, (delay_samples, 0))[:-delay_samples]
    # Front Right gets it last (delayed by diagonal distance)
    fr_mic = np.pad(base_sound, (delay_samples * 2, 0))[:-(delay_samples * 2)]
    
    result = detect_direction(fl_mic, fr_mic, rl_mic, rr_mic, sr)
    calc_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Algorithm Detected:   {result}")
    print(f"Processing Time:      {calc_time:.2f} ms")
