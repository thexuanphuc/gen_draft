import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sigma0 = 1.8 # N/m, bristle stiffness -> magnitude of the friction force, amplifies the contribution of the internal state z
sigma1 = 0.9 # Ns/m, bristle damping
sigma2 = 0.001 # Ns/m, viscous friction coefficient
Fs = 0.12 * 0.250 # N, static friction
Fc = 0.07 * 0.250 # N, Coulomb friction
vs = 0.00000000000001 # m/s, Stribeck velocity

# Define time and velocity profile
t = np.arange(0, 10, 0.01)  # Time from 0 to 10 seconds
v = 0.02 * np.sin(6 * np.pi * 0.1 * t)  # Sinusoidal velocity, max 0.03 m/s

# Initialize arrays for z and F
z = np.zeros_like(t)
F = np.zeros_like(t)
z[0] = 0  # Initial condition for z

# Simulate the LuGre model
for i in range(1, len(t)):
    # Compute g(v)
    g_v = Fc + (Fs - Fc) * np.exp(-(v[i-1] / vs)**2) + sigma2 * np.abs(v[i-1])
    
    # Compute dz/dt
    dz_dt = v[i-1] - (np.abs(v[i-1]) / g_v) * z[i-1]
    
    # Update z using Euler's method
    z[i] = z[i-1] + dz_dt * (t[i] - t[i-1])
    
    # Compute friction force F
    F[i] = sigma0 * z[i] + sigma1 * dz_dt + sigma2 * v[i-1]

# Plot F vs. v and z vs. v
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(v, F, color="#2F91D7")
plt.xlabel('Velocity (m/s)')
plt.ylabel('Friction Force (N)')
plt.title('Friction Force vs. Velocity')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(v, z, color="#e0720a")
plt.xlabel('Velocity (m/s)')
plt.ylabel('Internal State z (m)')
plt.title('Internal State z vs. Velocity')
plt.grid(True)

plt.tight_layout()
plt.show()