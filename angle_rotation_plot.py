import numpy as np
from scipy.signal import find_peaks

def find_average_period(timer, signal, prominence=None):
    """
    Compute the average period of a periodic signal by finding peaks.
    
    Parameters:
    - timer: Array of time values.
    - signal: Array of signal values corresponding to timer.
    - prominence: Optional parameter for find_peaks to filter out small peaks.
    
    Returns:
    - avg_period: Average time difference between consecutive peaks.
    """
    # Find peaks
    peaks, _ = find_peaks(signal, prominence=prominence)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks to determine period")
    # Get the times of peaks
    peak_times = timer[peaks]
    # Compute differences between consecutive peaks
    diffs = np.diff(peak_times)
    # Average the differences
    avg_period = np.mean(diffs)
    return avg_period

def find_period_fft_uniform(timer, signal):
    """
    Compute the period of a periodic signal using FFT, assuming uniform sampling.
    
    Parameters:
    - timer: Array of time values (assumed uniformly spaced).
    - signal: Array of signal values corresponding to timer.
    
    Returns:
    - period: Period of the fundamental frequency.
    """
    # Compute average time step (dt)
    dt = np.mean(np.diff(timer))
    N = len(signal)
    # Compute FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)
    # Find the frequency with the highest magnitude (excluding DC component)
    positive_freqs = freqs > 0
    if np.any(positive_freqs):
        idx = np.argmax(np.abs(fft_result[positive_freqs]))
        fundamental_freq = freqs[positive_freqs][idx]
        if fundamental_freq > 0:
            period = 1 / fundamental_freq
            return period
    raise ValueError("Could not find fundamental frequency")

def estimate_alpha(input_timer, input_force, output_timer, output_position, prominence=0.1):
    """
    Estimate alpha using two methods: peak-valley and FFT.
    
    Parameters:
    - input_timer: Array of input timer values.
    - input_force: Array of input force values.
    - output_timer: Array of output timer values.
    - output_position: Array of output position values.
    - prominence: Prominence threshold for peak detection.
    
    Returns:
    - alpha_pv: Alpha estimated using peak-valley method.
    - alpha_fft: Alpha estimated using FFT method.
    """
    # Check for uniform sampling (optional warning)
    def is_uniform(timer, tol=1e-6):
        dt = np.diff(timer)
        return np.all(np.abs(dt - dt.mean()) < tol)
    
    if not is_uniform(input_timer):
        print("Warning: input_timer is not uniformly spaced. FFT method may be inaccurate.")
    if not is_uniform(output_timer):
        print("Warning: output_timer is not uniformly spaced. FFT method may be inaccurate.")
    
    # Method 1: Peak-valley
    try:
        P_in_pv = find_average_period(input_timer, input_force, prominence)
        P_out_pv = find_average_period(output_timer, output_position, prominence)
        alpha_pv = P_in_pv / P_out_pv
    except ValueError as e:
        print(f"Error in peak-valley method: {e}")
        alpha_pv = None
    
    # Method 2: FFT (assuming uniform sampling)
    try:
        P_in_fft = find_period_fft_uniform(input_timer, input_force)
        P_out_fft = find_period_fft_uniform(output_timer, output_position)
        alpha_fft = P_in_fft / P_out_fft
    except ValueError as e:
        print(f"Error in FFT method: {e}")
        alpha_fft = None
    
    return alpha_pv, alpha_fft

# Example usage (replace with actual data)
# input_timer = np.array([...])
# input_force = np.array([...])
# output_timer = np.array([...])
# output_position = np.array([...])


# Set parameters
alpha = 2
beta = 5
t = np.arange(0, 10.1, 0.01)  # Physical time from 0 to 8 with step 0.1
output_timer = t
input_timer = alpha * t + beta
P_in = 8
P_out = 4
noise_level = 10

# Generate input_force (triangular wave)
saw_in = (input_timer % P_in) / P_in
input_force = 1 - 2 * np.abs(saw_in - 0.5)
input_force_noisy = input_force + np.random.normal(0, noise_level, size=input_force.shape)

# Generate output_position (sinusoidal wave)
output_position = np.sin(2 * np.pi * output_timer / P_out)
output_position_noisy = output_position + np.random.normal(0, noise_level, size=output_position.shape)

# The generated arrays are:
# - input_timer: shape (81,)
# - input_force_noisy: shape (81,)
# - output_timer: shape (81,)
# - output_position_noisy: shape (81,)

alpha_pv, alpha_fft = estimate_alpha(input_timer, input_force, output_timer, output_position)
print(f"Alpha from peak-valley: {alpha_pv}")
print(f"Alpha from FFT: {alpha_fft}")