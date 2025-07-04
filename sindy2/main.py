from params import Params
from sindy import SINDy
import torch as T
import numpy as np

# Define parameters (example values)
params = Params(
    m1=1.0, m2=1.0, k1=1.0, k2=1.0, c1=0.1, c2=0.1,
    fr1={'friction_force_ratio': 0.1, 'DR_flag': False, 'a': 0.1, 'b': 0.1, 'c': 1.0, 'eps': 1e-6, 'V_star': 1.0},
    fr2={'friction_force_ratio': 0.1, 'DR_flag': False, 'a': 0.1, 'b': 0.1, 'c': 1.0, 'eps': 1e-6, 'V_star': 1.0},
    F1=1.0, freq1=1.0, F2=0.5, freq2=0.5, phi=0.0,
    x0=[0.0, 0.0, 0.0, 0.0], timefinal=10.0, timestep=0.01,
    poly_order=2, cos_phases=[1.0], sin_phases=[],
    y1_sgn_flag=True, y2_sgn_flag=True, x_sgn_flag=False, y_sgn_flag=False,
    log_1_fr1=False, log_2_fr1=False, log_1_fr2=False, log_2_frPOSE=False,
    lr=0.001, weightdecay=0.0, num_iter=10, num_epochs=100
)

# Initialize SINDy
sindy = SINDy(params)

# Generate data
ts, x_denoised = sindy.generate_data()

# Simulate noisy data
noise = np.random.normal(0, 0.01, x_denoised.shape)
x = x_denoised + noise

# Convert to torch tensors
train_set = T.tensor(x, dtype=T.float32)
times = T.tensor(ts, dtype=T.float32).unsqueeze(1)

# Learn the model
loss_track = sindy.learn(train_set, times)

# Print the learned equation
equation = sindy.print_learned_equation()
print("Learned Equation:", equation)