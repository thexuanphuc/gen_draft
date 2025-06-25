import numpy as np

import numpy as np
import matplotlib.pyplot as plt

class DahlFriction:
    def __init__(self, k1, k2, sigma0, sigma1, sigma2, vs, z0=0.0):
        """
        Initialize the Dahl friction model.
        
        Parameters:
        - k1: Static friction coefficient
        - k2: Coulomb friction coefficient
        - sigma0: Stiffness (N/m)
        - sigma1: Damping coefficient (Ns/m)
        - sigma2: Viscous friction coefficient (Ns/m)
        - vs: Stribeck velocity (m/s)
        - z0: Initial internal state (default 0.0)
        """
        self.k1 = k1
        self.k2 = k2
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.vs = vs
        self.z = z0

    def update(self, v, N, dt):
        """
        Compute friction force and update internal state.
        
        Parameters:
        - v: Current velocity (m/s)
        - N: Normal force (N)
        - dt: Time step (s)
        
        Returns:
        - Friction force (N)
        """
        # Compute static and Coulomb friction forces
        F_s = self.k1 * N
        F_c = self.k2 * N
        
        # Compute dz/dt
        if abs(v) < 1e-10:  # Handle zero velocity
            dzdt = 0.0
        else:
            F_sigma = F_c + (F_s - F_c) * np.exp(-(v / self.vs)**2)
            dzdt = v - (self.sigma0 * np.abs(v) / F_sigma) * self.z
        
        # Compute friction force
        F_f = self.sigma0 * self.z + self.sigma1 * dzdt + self.sigma2 * v
        
        # Update internal state
        self.z += dzdt * dt
        
        return F_f

    def reset(self):
        """
        Reset the internal state to zero.
        """
        self.z = 0.0

# Parameters
k1 = 0.12      # Static friction coefficient
k2 = 0.07      # Coulomb friction coefficient
sigma0 = 1100  # Stiffness (N/m)
sigma1 = 6    # Damping (Ns/m)
sigma2 = 0.0001  # Viscous friction coefficient (Ns/m)
vs = 0.00005   # Stribeck velocity (m/s) -> 50 micrometers per second
v_max = 0.03 # Maximum velocity for simulation (m/s)
f = 1         # Frequency (Hz)
T = 10        # Total time (s)
N = 0.28       # Normal force (N)
dt = 0.00001    # Time step (s)

# Time and velocity arrays
t = np.arange(0, T, dt)
v_t = v_max * np.sin(2 * np.pi * f * t)

# Initialize model
model = DahlFriction(k1, k2, sigma0, sigma1, sigma2, vs)

# Compute friction forces and internal states
F_f_list = []
z_list = [model.z]
for v in v_t:
    F_f = model.update(v, N, dt)
    F_f_list.append(F_f)
    z_list.append(model.z)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot friction force vs velocity
axs[0].plot(v_t, F_f_list, label='Friction Force', color='#1f77b4')
axs[0].set_xlabel('Velocity (m/s)')
axs[0].set_ylabel('Friction Force (N)')
axs[0].set_title('Dahl Friction Model: Force vs Velocity')
axs[0].grid(True)
axs[0].legend()

# Plot internal state z vs velocity
axs[1].plot(v_t, z_list[:-1], label='Internal State z', color='#ff7f0e')
axs[1].set_xlabel('Velocity (m/s)')
axs[1].set_ylabel('Internal State z')
axs[1].set_title('Dahl Friction Model: Internal State z vs Velocity')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()