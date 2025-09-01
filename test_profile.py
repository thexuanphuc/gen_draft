import numpy as np
import matplotlib.pyplot as plt

A_MAX = 2048
V_MAX = 1500
flag_decrese = False

# System parameters
h = 20 / 5000   # Sampling period
T = 1           # Total simulation time
N = int(T / h)  # Number of steps
v = 200     # Desired setpoint

# Initialize arrays
t = np.arange(0, T + h, h)
v1 = np.zeros(N + 1)
v2 = np.zeros(N + 1)
u_arr = np.zeros(N)

# Initial conditions
v1[0] = 0
v2[0] = 0


def simple_profile(e1, v2):
    global flag_decrese
    if flag_decrese:
        accel = -A_MAX * np.sign(e1)
    elif np.abs(e1) <= np.abs( v2 * h * 0.5) + (v2 ** 2) / (2 * A_MAX) + 1:
        accel = -A_MAX * np.sign(e1)
        flag_decrese = True
    elif np.abs(v2) < V_MAX:
        accel = A_MAX * np.sign(e1)
    else:
        accel = 0
    return accel

for k in range(N):
    e1 = v - v1[k]
    if np.abs(e1) < 4:
        v2[k+1] = 0
        v1[k+1] = v
        u_arr[k] = 0
        break
    u = simple_profile(e1, v2[k])
    u_arr[k] = u
    v2[k+1] = v2[k] + h * u
    v1[k+1] = v1[k] + h * v2[k+1]

# Plotting
plt.figure(figsize=(12, 8)) 
# Plot position
plt.subplot(3, 1, 1)
plt.step(t[:k+2], v1[:k+2], label='Position (v1)')
plt.step(t[:k+2], [v]*(k+2), '--', label='Setpoint (v)')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.title('Position Response')
plt.grid(True)

# Plot velocity
plt.subplot(3, 1, 2)
plt.step(t[:k+2], v2[:k+2], label='Velocity (v2)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity Response')
plt.grid(True)

# Plot control input
plt.subplot(3, 1, 3)
plt.step(t[:k+1], u_arr[:k+1], label='Control Input (u)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Input Response')
plt.grid(True)
plt.tight_layout()
plt.show()

