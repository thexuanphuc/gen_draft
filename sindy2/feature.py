import numpy as np
import torch as T
from itertools import combinations_with_replacement, chain
from .params import Params

class FeatureLibrary:
    """Class to generate and manage feature candidates."""
    def __init__(self, params: Params):
        self.params = params

    def apply_features(self, x, t, torch_flag=True):
        """Apply feature candidates to the data."""
        pol_indeces = list(chain.from_iterable(combinations_with_replacement(range(x.shape[1]), i) 
                                               for i in range(self.params.poly_order + 1)))
        if torch_flag:
            return T.column_stack((
                *[x[:, inds].prod(1) for inds in pol_indeces],
                *[T.cos(ph * t) for ph in self.params.cos_phases],
                *[T.sign(x[:, 2]),] * self.params.y1_sgn_flag,
                *[T.sign(x[:, 3]),] * self.params.y2_sgn_flag,
                *[T.log((T.abs(x[:, 2]) + self.params.fr1["eps"]) / self.params.fr1["V_star"]),] * self.params.log_1_fr1,
                *[T.log(self.params.fr1["c"] + self.params.fr1["V_star"] / (T.abs(x[:, 2]) + self.params.fr1["eps"])),] * self.params.log_2_fr1,
                *[T.log((T.abs(x[:, 3]) + self.params.fr2["eps"]) / self.params.fr2["V_star"]),] * self.params.log_1_fr2,
                *[T.log(self.params.fr2["c"] + self.params.fr2["V_star"] / (T.abs(x[:, 3]) + self.params.fr2["eps"])),] * self.params.log_2_fr2,
            ))
        else:
            return np.column_stack((
                *[x[:, inds].prod(1) for inds in pol_indeces],
                *[np.cos(ph * t) for ph in self.params.cos_phases],
                *[np.sign(x[:, 2]),] * self.params.y1_sgn_flag,
                *[np.sign(x[:, 3]),] * self.params.y2_sgn_flag,
                *[np.log((np.abs(x[:, 2]) + self.params.fr1["eps"]) / self.params.fr1["V_star"]),] * self.params.log_1_fr1,
                *[np.log(self.params.fr1["c"] + self.params.fr1["V_star"] / (np.abs(x[:, 2]) + self.params.fr1["eps"])),] * self.params.log_2_fr1,
                *[np.log((np.abs(x[:, 3]) + self.params.fr2["eps"]) / self.params.fr2["V_star"]),] * self.params.log_1_fr2,
                *[np.log(self.params.fr2["c"] + self.params.fr2["V_star"] / (np.abs(x[:, 3]) + self.params.fr2["eps"])),] * self.params.log_2_fr2,
            ))

    def get_feature_names(self):
        """Return feature names as strings."""
        return [
            *[f"cos({ph:.1f} t)" for ph in self.params.cos_phases],
            *[f"sin({ph:.1f} t)" for ph in self.params.sin_phases],
            *["sgn(x)",] * self.params.x_sgn_flag,
            *["sgn(y)",] * self.params.y_sgn_flag,
            *["1",] * (self.params.poly_order >= 0),
            *["x", "y"] * (self.params.poly_order >= 1),
            *["x^2", "xy", "y^2"] * (self.params.poly_order >= 2),
            *["x^3", "x^2y", "xy^2", "y^3"] * (self.params.poly_order >= 3),
            *["x^4", "x^3y", "x^2y^2", "xy^3", "y^4"] * (self.params.poly_order >= 4),
        ]

    def get_n_features(self):
        """Compute the number of features."""
        x_dummy = np.zeros((1, 4))
        t_dummy = np.zeros((1,))
        features = self.apply_features(x_dummy, t_dummy, torch_flag=False)
        return features.shape[1]