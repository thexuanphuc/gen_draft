import numpy as np
from scipy.integrate import solve_ivp
import torch as T
from .params import Params

class PhysicalSystem:
    """Class to model the physical system and generate data."""
    def __init__(self, params: Params):
        self.params = params

    def build_true_model(self, x, t):
        """Compute the true dynamics of the system."""
        Ffric1 = self.params.fr1['friction_force_ratio']
        Ffric2 = self.params.fr2['friction_force_ratio']

        if self.params.fr1.get('DR_flag', False):
            Ffric1 += (self.params.fr1["a"] * np.log((np.abs(x[2]) + self.params.fr1["eps"]) / self.params.fr1["V_star"]) +
                       self.params.fr1["b"] * np.log(self.params.fr1["c"] + self.params.fr1["V_star"] / (np.abs(x[2]) + self.params.fr1["eps"])))
        if self.params.fr2.get('DR_flag', False):
            Ffric2 += (self.params.fr2["a"] * np.log((np.abs(x[3]) + self.params.fr2["eps"]) / self.params.fr2["V_star"]) +
                       self.params.fr2["b"] * np.log(self.params.fr2["c"] + self.params.fr2["V_star"] / (np.abs(x[3]) + self.params.fr2["eps"])))

        derivs = np.array([
            x[2],
            x[3],
            (-(self.params.k1 + self.params.k2) / self.params.m1 * x[0] -
             (self.params.c1 + self.params.c2) / self.params.m1 * x[2] +
             self.params.k2 / self.params.m1 * x[1] +
             self.params.c2 / self.params.m1 * x[3] -
             Ffric1 / self.params.m1 * np.sign(x[2]) +
             self.params.F1 / self.params.m1 * np.cos(self.params.freq1 * t)),
            (-self.params.k2 / self.params.m2 * x[1] -
             self.params.c2 / self.params.m2 * x[3] +
             self.params.k2 / self.params.m2 * x[0] +
             self.params.c2 / self.params.m2 * x[2] +
             self.params.F2 / self.params.m2 * np.cos(self.params.freq2 * (t + self.params.phi)) -
             Ffric2 / self.params.m2 * np.sign(x[3]))
        ])

        if (np.abs(x[2]) <= 1e-5 and 
            np.abs(self.params.F1 * np.cos(self.params.freq1 * t) + self.params.c2 * x[3] + self.params.k2 * x[1] - 
                   (self.params.k1 + self.params.k2) * x[0]) <= np.abs(Ffric1)):
            derivs[[0, 2]] = 0.
        if (np.abs(x[3]) <= 1e-5 and 
            np.abs(self.params.c2 * x[2] + self.params.k2 * x[0] - self.params.k2 * x[1]) <= np.abs(Ffric2)):
            derivs[[1, 3]] = 0.

        return derivs

    def generate_data(self):
        """Generate ground truth data using the true model."""
        ts = np.arange(0, self.params.timefinal, self.params.timestep)
        sol = solve_ivp(
            lambda t, x: self.build_true_model(x, t),
            t_span=[ts[0], ts[-1]], y0=self.params.x0, t_eval=ts
        )
        return ts, np.transpose(sol.y)

    def apply_known_physics(self, x, times):
        """Compute known physical terms using torch tensors."""
        known_terms_1 = (-(self.params.c1 + self.params.c2) / self.params.m1 * x[:, 2] -
                         (self.params.k1 + self.params.k2) / self.params.m1 * x[:, 0] +
                         self.params.k2 / self.params.m1 * x[:, 1] +
                         self.params.c2 / self.params.m1 * x[:, 3] +
                         self.params.F1 / self.params.m1 * T.cos(self.params.freq1 * times.squeeze(1)))
        known_terms_2 = (-self.params.c2 / self.params.m2 * x[:, 3] -
                         self.params.k2 / self.params.m2 * x[:, 1] +
                         self.params.k2 / self.params.m2 * x[:, 0] +
                         self.params.c2 / self.params.m2 * x[:, 2])
        return T.column_stack((known_terms_1, known_terms_2))