import control
import numpy as np
import matplotlib.pyplot as plt

# Define the time constants and sampling time
T1 = 0.00001
T2 = 0.00005
Ts = 0.0002  # Sampling time

# Create the continuous-time transfer function H(s)
s = control.tf('s')
H_s = s / ((T1 * s + 1) * (T2 * s + 1))

print("Continuous-time transfer function H(s):")
print(H_s)

# Discretize using bilinear (Tustin) transform
H_z = control.sample_system(H_s, Ts, method='tustin')

print("\nDiscrete-time transfer function H(z) with sampling time Ts =", Ts)
print(H_z)

# Plot Bode plot of the discrete system
plt.figure()
control.bode_plot(H_z, dB=True)
plt.suptitle('Bode Plot of Discrete Transfer Function')



# Define the time axis
t = np.linspace(0, 10, int(10/0.0002))  # time from 0 to 10 seconds

# Define the signal, e.g., a sine wave
signal = np.sin(t)
derivative = control.forced_response(H_z, U=signal)[1]
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Signal (sin(t))', color='blue')
plt.plot(t, derivative, label="Derivative of Signal (numerical)", color='red', linestyle='--')
plt.title('Signal and its Numerical Derivative')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
