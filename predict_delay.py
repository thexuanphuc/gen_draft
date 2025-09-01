import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 1.0  # Plant gain
T = 1.0  # Plant time constant
tau = 1.0  # Time delay (seconds)
dt = 0.01  # Sample time (seconds)
t_sim = 10.0  # Simulation time (seconds)
Kp = 2.0  # Proportional controller gain
r = 1.0  # Reference setpoint

# Tracking differentiator parameters
r_td = 10.0  # Tracking differentiator speed
h = dt  # Step size for tracking differentiator

# Time array
t = np.arange(0, t_sim, dt)
n = len(t)

# Initialize arrays
u = np.zeros(n)  # Control input
y = np.zeros(n)  # Actual output (with delay)
y_model = np.zeros(n)  # Model output (no delay)
y_model_delayed = np.zeros(n)  # Model output (with delay)
y0_pred = np.zeros(n)  # Predicted delay-free output
y_td = np.zeros(n)  # Tracking differentiator output
dy_td = np.zeros(n)  # Tracking differentiator derivative

# Delay buffer
delay_steps = int(tau / dt)
u_delayed = np.zeros(n)

# Tracking differentiator function (simplified)
def tracking_differentiator(y, y_prev, dy_prev, r_td, h):
    """
    Simple tracking differentiator to estimate y0 and its derivative.
    Based on ADRC tracking differentiator: tracks input signal and predicts future value.
    """
    # Fastest tracking function (Han, 2009)
    a0 = h * dy_prev
    y0 = y_prev + a0
    a1 = np.sqrt(r_td * (r_td + 8 * abs(y - y0)))
    a = a0 + np.sign(y - y0) * (a1 - r_td) / 2
    dy = dy_prev + h * (-r_td * a)
    y_new = y_prev + h * dy
    return y_new, dy

# Simulation loop
for i in range(1, n):
    # Plant model (first-order system: dy/dt = -y/T + K*u/T)
    y_model[i] = y_model[i-1] + dt * (-y_model[i-1]/T + K*u[i-1]/T)
    
    # Delayed model output (shift u by tau)
    if i >= delay_steps:
        u_delayed[i] = u[i - delay_steps]
    y_model_delayed[i] = y_model_delayed[i-1] + dt * (-y_model_delayed[i-1]/T + K*u_delayed[i-1]/T)
    
    # Actual plant output (with delay)
    y[i] = y_model_delayed[i]  # Simulate actual output as delayed model output
    
    # Tracking differentiator to predict y0
    y_td[i], dy_td[i] = tracking_differentiator(y[i], y_td[i-1], dy_td[i-1], r_td, h)
    
    # Smith Predictor: y0_pred = y_model + (y - y_model_delayed)
    y0_pred[i] = y_model[i] + (y[i] - y_model_delayed[i])
    
    # Controller (simple P control based on predicted y0)
    error = r - y0_pred[i]
    u[i] = Kp * error

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Actual Output (y, with delay)')
plt.plot(t, y0_pred, label='Predicted Delay-Free Output (y0)')
plt.plot(t, y_model, label='Model Output (no delay)')
# plt.plot(t, y_td, label='Tracking Differentiator Output')
plt.plot(t, np.ones(n) * r, 'k--', label='Reference')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Smith Predictor with Tracking Differentiator')
plt.legend()
plt.grid(True)
plt.show()