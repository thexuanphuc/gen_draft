import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters based on user input
sigma0 = 1100  # Stiffness (N/m)
sigma1 = 6    # Damping (Ns/m)
sigma2 = 0.0001  # Viscous friction coefficient (Ns/m)
Fs = 0.03     # Static friction (N)
Fc = 0.02     # Coulomb friction (N)
vs = 0.00005   # Stribeck velocity (m/s) -> 50 micrometers per second
v_max = 0.03 # Maximum velocity for simulation (m/s)
f = 5         # Frequency of velocity oscillation (Hz)
T = 10        # Total simulation time (s)   

# Time points
t_span = (0, T)
t_eval = np.linspace(0, T, 100000)

# Velocity function: sinusoidal
def v(t):
    return v_max * np.sin(2 * np.pi * f * t)

# ODE for the internal state z
def dahl_ode(t, z, t_eval, v_func, sigma0, Fc, Fs, vs):
    v_t = v_func(t)
    if abs(v_t) < 1e-10:  # Handle zero velocity
        return 0
    F_sigma = Fc + (Fs - Fc) * np.exp(-(v_t / vs)**2)
    return v_t - (sigma0 * np.abs(v_t) / F_sigma) * z

# Solve the ODE
sol = solve_ivp(dahl_ode, t_span, [0], args=(t_eval, v, sigma0, Fc, Fs, vs), t_eval=t_eval, method='RK45')

# Extract results
z = sol.y[0]
t = sol.t
v_t = v(t)
dzdt = np.array([dahl_ode(ti, zi, t_eval, v, sigma0, Fc, Fs, vs) for ti, zi in zip(t, z)])
F_f = sigma0 * z + sigma1 * dzdt + sigma2 * v_t

# Plot force vs velocity
plt.figure(figsize=(10, 6))
plt.plot(v_t, F_f, label='Friction Force', color='#1f77b4')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Friction Force (N)')
plt.title('Dahl Friction Model: Force vs Velocity')
plt.grid(True)
plt.legend()
plt.show()