import numpy as np

# Example matrices (replace with your actual A and B)
A1= np.array([[1, 1],
               [0, 1]])
B1 = np.array([[0],[1]])
B2 = np.array([[0],[1]])
C1 = np.array([[1, 1]])


A = np.array([[1, 1, 0]])
B = np.random.rand(m, 1)  # Example: m x 1 vector

# Extract the relevant columns of A (for p1, p2, p3)
A_sub = A[:, 2:5]  # Columns 3, 4, 5 (0-based indexing)

# Solve the linear system A_sub @ [p1, p2, p3] = -B
if A_sub.shape[0] == A_sub.shape[1]:  # Check if A_sub is square
    p = np.linalg.solve(A_sub, -B)
else:
    print("System is not square; use least-squares or cvxpy.")
    p = np.linalg.lstsq(A_sub, -B, rcond=None)[0]

# Construct the full P vector
P = np.zeros((5, 1))
P[2:5, 0] = p.flatten()  # Assign p1, p2, p3

# Verify the solution
residual = -A @ P - B
print("Solution P:", P.flatten())
print("Residual norm:", np.linalg.norm(residual))
print("Is solution valid (small residual)?", np.linalg.norm(residual) < 1e-6)