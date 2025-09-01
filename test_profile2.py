def euler_position_velocity(s0, v0, a, delta_T):
    s = s0
    v = v0
    k = 0
    # List to store trajectory
    s_list = [s]
    v_list = [v]

    # Iterate until velocity reaches or goes below zero
    while v > 0:
        s = s + v * delta_T
        v = v - a * delta_T
        s_list.append(s)
        v_list.append(v)
        k += 1

    # Analytical step count
    N_analytical = int(v0 / (a * delta_T))

    # Analytical s_N formula from above
    s_N_analytical = s0 + (v0**2) / (2 * a) + (v0 * delta_T) / 2

    return s_list[-1], k, s_N_analytical, N_analytical

# Example parameters
s0 = 0.0
v0 = 600.0  # initial velocity
a = 2000.0    # acceleration (assumed positive for deceleration here)
delta_T = 25/5000  # time step, can be large

s_N_numerical, steps_taken, s_N_analytical, N_analytical = euler_position_velocity(s0, v0, a, delta_T)

print(f"Numerical s_N after {steps_taken} steps: {s_N_numerical}")
print(f"Analytical s_N: {s_N_analytical}")
print(f"Numerical steps needed: {steps_taken}")
print(f"Analytical steps estimate: {N_analytical}")
