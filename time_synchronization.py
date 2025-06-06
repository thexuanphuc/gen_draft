import numpy as np
import matplotlib.pyplot as plt

def plot_delta_t(timer, timer_name):
    """
    Plot the time differences (delta t) between consecutive samples for a timer array.
    
    Parameters:
    - timer: Array of time values.
    - timer_name: String name of the timer (e.g., 'input_timer' or 'output_timer').
    """
    # Compute delta t
    delta_t = np.diff(timer)
    # Generate sample indices
    sample_indices = np.arange(1, len(timer))
    
    # Plot delta t
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, delta_t, marker='o', linestyle='-', color='#1f77b4')
    plt.xlabel('Sample Index')
    plt.ylabel('Delta t (Time Difference)')
    plt.title(f'Time Differences for {timer_name}')
    plt.grid(True)
    plt.show()

# Example usage within your estimate_alpha function
def estimate_alpha(input_timer, input_force, output_timer, output_position, prominence=0.1):
    """
    Estimate alpha using peak-valley and FFT methods, and plot delta t if timers are non-uniform.
    """
    def is_uniform(timer, tol=1e-6):
        dt = np.diff(timer)
        return np.all(np.abs(dt - dt.mean()) < tol)
    
    # Check uniformity and plot delta t if non-uniform
    if not is_uniform(input_timer):
        print("Warning: input_timer is not uniformly spaced. FFT method may be inaccurate.")
        plot_delta_t(input_timer, 'input_timer')
    if not is_uniform(output_timer):
        print("Warning: output_timer is not uniformly spaced. FFT method may be inaccurate.")
        plot_delta_t(output_timer, 'output_timer')
    
    # Existing alpha estimation code (peak-valley and FFT)
    try:
        P_in_pv = find_average_period(input_timer, input_force, prominence)
        P_out_pv = find_average_period(output_timer, output_position, prominence)
        alpha_pv = P_in_pv / P_out_pv
    except ValueError as e:
        print(f"Error in peak-valley method: {e}")
        alpha_pv = None
    
    try:
        P_in_fft = find_period_fft_uniform(input_timer, input_force)
        P_out_fft = find_period_fft_uniform(output_timer, output_position)
        if P_out_fft == None or P_in_fft is None:
            return None, None
        alpha_fft = P_in_fft / P_out_fft
    except ValueError as e:
        print(f"Error in FFT method: {e}")
        alpha_fft = None
    
    return alpha_pv, alpha_fft

# Placeholder for find_average_period and find_period_fft_uniform
def find_average_period(timer, signal, prominence=None):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, prominence=prominence)
    if len(peaks) < 2:
        raise ValueError("Not enough peaks to determine period")
    peak_times = timer[peaks]
    diffs = np.diff(peak_times)
    return np.mean(diffs)

import numpy as np

def find_period_fft_uniform(timer, signal):
    dt = np.mean(np.diff(timer))
    N = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)
    positive_freqs = freqs > 0
    
    if np.any(positive_freqs):
        # Get amplitudes for positive frequencies
        amplitudes = np.abs(fft_result[positive_freqs])
        # Find indices of top two amplitudes
        sorted_indices = np.argsort(amplitudes)[::-1]  # Descending order
        if len(sorted_indices) > 1:  # Ensure there's at least two amplitudes
            max_amp = amplitudes[sorted_indices[0]]
            second_max_amp = amplitudes[sorted_indices[1]]
            # Check if max amplitude is at least 10 times larger
            if max_amp >= 10 * second_max_amp:
                fundamental_freq = freqs[positive_freqs][sorted_indices[0]]
                if fundamental_freq > 0:
                    return 1 / fundamental_freq
        else:
            # If only one positive frequency, check if it's valid
            fundamental_freq = freqs[positive_freqs][sorted_indices[0]]
            if fundamental_freq > 0:
                return 1 / fundamental_freq
    return None

# Example test data (non-uniform for demonstration)
np.random.seed(42)
t = np.sort(np.concatenate([np.arange(0, 4, 0.1), np.arange(4, 8, 0.15)]))  # Non-uniform sampling
input_timer = 2 * t + 5
output_timer = t
input_force = 1 - 2 * np.abs((input_timer % 8) / 8) + np.random.normal(0, 0.05, len(t))
output_position = np.sin(2 * np.pi * output_timer / 4) + np.random.normal(0, 0.05, len(t))

# Run estimation and plot delta t if non-uniform
alpha_pv, alpha_fft = estimate_alpha(input_timer, input_force, output_timer, output_position, prominence=0.2)
print(f"Alpha from peak-valley: {alpha_pv}")
print(f"Alpha from FFT: {alpha_fft}")