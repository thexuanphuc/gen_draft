import numpy as np
import matplotlib.pyplot as plt

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (T·m/A)
I = 1.0  # Current (A)
a = 0.1  # Half-width of rectangle along x (m)
b = 0.05  # Half-height of rectangle along y (m)

# Define vertices of the rectangular loop centered at origin
vertices = [
    np.array([a, b, 0]),      # Vertex A
    np.array([a, -b, 0]),     # Vertex B
    np.array([-a, -b, 0]),    # Vertex C
    np.array([-a, b, 0])      # Vertex D
]

# Convert vertices to NumPy array for plotting
vertices_array = np.array(vertices)

# Function to compute magnetic field at point r due to a line segment
def B_segment(r, p1, p2, I, N=100):
    """Compute B field at r due to segment from p1 to p2 with current I."""
    dl = (p2 - p1) / N
    t = np.linspace(0, 1, N)
    B = np.zeros(3)
    for i in range(N):
        r_prime = p1 + t[i] * (p2 - p1)
        r_minus_r_prime = r - r_prime
        r_mag = np.linalg.norm(r_minus_r_prime)
        if r_mag < 1e-10:  # Avoid singularity
            continue
        cross = np.cross(dl, r_minus_r_prime)
        B += (mu_0 * I / (4 * np.pi * r_mag**3)) * cross
    return B

# Compute magnetic field at point r due to rectangular loop
def B_rectangle(r):
    B_total = np.zeros(3)
    # Sum contributions from each side (A->B, B->C, C->D, D->A)
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        B_total += B_segment(r, p1, p2, I)
    return B_total

# Compute Bz in the xz-plane for heatmap
X, Z = np.meshgrid(np.linspace(-0.2, 0.2, 50), np.linspace(-0.2, 0.2, 50))
Bz_grid = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B = B_rectangle(np.array([X[i, j], 0, Z[i, j]]))
        Bz_grid[i, j] = B[2]  # z-component

# Create plots
fig = plt.figure(figsize=(12, 6))

# Plot Bz heatmap in xz-plane with coil edges
plt.subplot(1, 2, 1)
contour = plt.contourf(X, Z, Bz_grid, levels=50, cmap='RdBu', vmin=-np.max(np.abs(Bz_grid)), vmax=np.max(np.abs(Bz_grid)))
plt.colorbar(contour, label='B_z (Tesla)')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Magnetic Flux Density B_z in xz-Plane')
# Draw coil edges at x = ±a
plt.axvline(a, color='black', linestyle='--', linewidth=1, label='Coil Edges')
plt.axvline(-a, color='black', linestyle='--', linewidth=1)
plt.axhline(0, color='black', linestyle='-', linewidth=1)  # Coil plane
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axis('equal')

# # Plot the rectangular coil
# plt.subplot(1, 2, 2)
# vertices_closed = np.vstack([vertices_array, vertices_array[0]])  # Close the loop
# plt.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'k-', label='Rectangular Coil')
# plt.scatter(vertices_array[:, 0], vertices_array[:, 1], color='black', s=50, label='Vertices')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title(f'Rectangular Coil (Width={2*a}m, Height={2*b}m)')
# plt.grid(True)
# plt.legend()
# plt.axis('equal')

plt.tight_layout()
plt.show()