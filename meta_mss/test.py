import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results
from sysidentpy.utils.generate_data import get_mimo_data # Utility to generate MISO/MIMO-like data

# --- 1. Data Generation (Simulating a MISO System) ---
# Define a system with two inputs (x1, x2) and one output (y)
# Target Equation (Nonlinear NARX): y(k) = 0.5*y(k-1) + 0.8*x1(k-1) + 0.3*x2(k-1)*x1(k-1) + e(k)

N = 1000  # Number of data points
n_train = int(N * 0.9)

# Generate Inputs and Noise
x1 = np.random.uniform(-1, 1, N).reshape(-1, 1)
x2 = np.random.uniform(-1, 1, N).reshape(-1, 1)
noise = np.random.normal(0, 0.05, N).reshape(-1, 1)

# Combined Input Matrix X (MISO requires input matrix X to have multiple columns, e.g., N x 2) [9, 10]
X = np.concatenate([x1, x2], axis=1)

# Initialize output y
y = np.zeros_like(noise)
y = 0.1 # Initial condition

# Simulation loop for the target MISO system
for k in range(1, N):
    y[k] = (0.5 * y[k-1] + 0.8 * x1[k-1] + 0.3 * x2[k-1] * x1[k-1] + noise[k])

# Split data
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# --- 2. Model Structure Selection (MSS) using FROLS ---

# Define the basis function: Polynomial degree 2 to capture the nonlinear interaction (x2*x1) [6]
basis_function = Polynomial(degree=2)

# Define the parameter estimator (Least Squares for models linear in parameters) [7, 11]
estimator = LeastSquares(unbiased=False)

# Define lags for the MISO system:
# ylag=1 for y(k-1); xlag=[[12]] for x1(k-1) and x2(k-1)
# Note: xlag must be a list of lists/arrays, one for each input dimension (2 inputs here) [10]
MAX_LAG = 2 

model = FROLS(
    order_selection=True,  # Enable MSS termination based on Information Criteria [13]
    n_info_values=15,      # Evaluate up to 15 terms (or terms needed for BIC minimum) [13]
    ylag=MAX_LAG,
    xlag=[list(range(1, MAX_LAG + 1))] * X_train.shape[12], # Set lags for both input columns
    info_criteria="bic",   # Use Bayesian Information Criterion for complexity selection [13]
    estimator=estimator,
    basis_function=basis_function,
)

# Fit the model: Performs MSS (FROLS) and Parameter Estimation (LS) [14]
model.fit(X=X_train, y=y_train)

# --- 3. Prediction and Validation (Free Run Simulation) ---

# Prepare data for prediction, concatenating the necessary initial conditions [15]
y_test_init = np.concatenate([y_train[-model.max_lag:], y_test])
X_test_init = np.concatenate([X_train[-model.max_lag:], X_test])

# Perform infinity-step-ahead prediction (free run simulation) [15, 16]
yhat = model.predict(X=X_test_init, y=y_test_init)

# Calculate performance metric (RRSE) [17]
rrse = root_relative_squared_error(
    y_test[model.max_lag :], 
    yhat[model.max_lag :]
)

# --- 4. Display Results ---

# Display the final model terms, parameters, and ERR values [18]
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

print("\n--- Selected MISO NARX Model Structure ---")
print(r)
print(f"\nRoot Relative Squared Error (RRSE) on Validation Data: {rrse:.4f}")

# Plotting the free run simulation results
plot_results(
    y=y_test_init[model.max_lag:], 
    yhat=yhat[model.max_lag:], 
    n=100, 
    title=f"MISO NARX Free Run Simulation (RRSE: {rrse:.4f})"
)