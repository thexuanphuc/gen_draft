import numpy as np

def compute_trajectory(x_r, y_r, theta, x_d, y_d):
    """
    Compute the trajectory of a point on a rigid body given the reference point's position and angle over time.

    Parameters:
    - x_r (np.array): Array of x-coordinates of the reference point over time (shape: (n,))
    - y_r (np.array): Array of y-coordinates of the reference point over time (shape: (n,))
    - theta (np.array): Array of angles (in radians) of the rigid body over time (shape: (n,))
    - x_d (float or np.array): x-coordinate(s) of the desired point(s) relative to the reference point
    - y_d (float or np.array): y-coordinate(s) of the desired point(s) relative to the reference point

    Returns:
    - x_a (np.array): x-coordinates of the desired point(s) over time (shape: (n,) or (n, m))
    - y_a (np.array): y-coordinates of the desired point(s) over time (shape: (n,) or (n, m))
    """
    # Ensure inputs are NumPy arrays
    x_r = np.asarray(x_r)
    y_r = np.asarray(y_r)
    theta = np.asarray(theta)
    x_d = np.asarray(x_d)
    y_d = np.asarray(y_d)

    # Compute cosine and sine of theta for all time steps
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Compute the absolute positions of the desired point(s)
    # If x_d and y_d are scalars (single point), x_a and y_a will be (n,)
    # If x_d and y_d are arrays (multiple points), x_a and y_a will be (n, m)
    x_a = x_r[:, None] + x_d[None, :] * cos_theta[:, None] - y_d[None, :] * sin_theta[:, None]
    y_a = y_r[:, None] + x_d[None, :] * sin_theta[:, None] + y_d[None, :] * cos_theta[:, None]

    # Remove extra dimensions if x_d and y_d are scalars
    if x_d.ndim == 0 and y_d.ndim == 0:
        x_a = x_a.flatten()
        y_a = y_a.flatten()

    return x_a, y_a

# Example usage
if __name__ == "__main__":
    # Example data
    t = np.linspace(0, 10, 100)  # Time steps
    x_r = np.sin(t)  # Reference point x-position
    y_r = np.cos(t)  # Reference point y-position
    theta = t  # Angle (radians)
    x_d = 1.0  # Relative x-position of desired point
    y_d = 0.5  # Relative y-position of desired point

    # Compute trajectory
    x_a, y_a = compute_trajectory(x_r, y_r, theta, x_d, y_d)

    # Optional: Visualize the trajectory
    import matplotlib.pyplot as plt
    plt.plot(x_a, y_a, label="Trajectory of desired point")
    plt.plot(x_r, y_r, label="Reference point trajectory", linestyle="--")
    plt.title("Trajectory of Point on Rigid Body")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()