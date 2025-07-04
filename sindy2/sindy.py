import torch as T
import numpy as np
import torch_optimizer as optim_all
from .params import Params
from .physical_system import PhysicalSystem
from .features import FeatureLibrary
from .learning import CoeffsDictionary

class SINDy:
    """Main class for Sparse Identification of Nonlinear Dynamics."""
    def __init__(self, params: Params):
        self.params = params
        self.physical_system = PhysicalSystem(params)
        self.feature_library = FeatureLibrary(params)
        n_combinations = self.feature_library.get_n_features()
        n_eqs = 2  # Number of equations (e.g., for 2-DOF system)
        self.coeffs = CoeffsDictionary(n_combinations, n_eqs)

    def generate_data(self):
        """Generate data using the physical system."""
        return self.physical_system.generate_data()

    def apply_rk4_SparseId(self, x, times, timesteps):
        """Apply 4th order Runge-Kutta with sparse identification."""
        d1 = self.feature_library.apply_features(x, times, torch_flag=True)
        k1 = T.column_stack((x[:, 2:],
                             self.physical_system.apply_known_physics(x, times) + self.coeffs(d1) * T.sign(x[:, 2:])))
        k1[:, [0, 2]] = T.where((T.abs(x[:, 2]).unsqueeze(1) <= 1e-3), 0., k1[:, [0, 2]])
        k1[:, [1, 3]] = T.where((T.abs(x[:, 3]).unsqueeze(1) <= 1e-3), 0., k1[:, [1, 3]])

        xtemp = x + 0.5 * timesteps * k1
        d2 = self.feature_library.apply_features(xtemp, times + 0.5 * timesteps, torch_flag=True)
        k2 = T.column_stack((xtemp[:, 2:],
                             self.physical_system.apply_known_physics(xtemp, times) + self.coeffs(d2) * T.sign(xtemp[:, 2:])))
        k2[:, [0, 2]] = T.where((T.abs(xtemp[:, 2]).unsqueeze(1) <= 1e-3), 0., k2[:, [0, 2]])
        k2[:, [1, 3]] = T.where((T.abs(xtemp[:, 3]).unsqueeze(1) <= 1e-3), 0., k2[:, [1, 3]])

        xtemp = x + 0.5 * timesteps * k2
        d3 = self.feature_library.apply_features(xtemp, times + 0.5 * timesteps, torch_flag=True)
        k3 = T.column_stack((xtemp[:, 2:],
                             self.physical_system.apply_known_physics(xtemp, times) + self.coeffs(d3) * T.sign(xtemp[:, 2:])))
        k3[:, [0, 2]] = T.where((T.abs(xtemp[:, 2]).unsqueeze(1) <= 1e-3), 0., k3[:, [0, 2]])
        k3[:, [1, 3]] = T.where((T.abs(xtemp[:, 3]).unsqueeze(1) <= 1e-3), 0., k3[:, [1, 3]])

        xtemp = x + timesteps * k3
        d4 = self.feature_library.apply_features(xtemp, times + timesteps, torch_flag=True)
        k4 = T.column_stack((xtemp[:, 2:],
                             self.physical_system.apply_known_physics(xtemp, times) + self.coeffs(d4) * T.sign(xtemp[:, 2:])))
        k4[:, [0, 2]] = T.where((T.abs(xtemp[:, 2]).unsqueeze(1) <= 1e-3), 0., k4[:, [0, 2]])
        k4[:, [1, 3]] = T.where((T.abs(xtemp[:, 3]).unsqueeze(1) <= 1e-3), 0., k4[:, [1, 3]])

        return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timesteps

    def learn(self, train_set, times, num_iter=None, num_epochs=None, lr_reduction=10):
        """Learn the sparse model coefficients."""
        num_iter = num_iter or self.params.num_iter
        num_epochs = num_epochs or self.params.num_epochs
        opt_func = optim_all.RAdam(self.coeffs.parameters(), lr=self.params.lr, weight_decay=self.params.weightdecay)
        criteria = T.nn.MSELoss()
        loss_track = np.zeros((num_iter, num_epochs))

        timesteps = T.tensor(np.diff(times.numpy(), axis=0)).float()
        for p in range(num_iter):
            for g in range(num_epochs):
                self.coeffs.train()
                opt_func.zero_grad()
                y_pred = self.apply_rk4_SparseId(train_set[:-1], times[:-1], timesteps)
                loss = criteria(y_pred, train_set[1:])
                loss_track[p, g] = loss.item()
                loss.backward()
                opt_func.step()
            # Optionally reduce learning rate
            for param_group in opt_func.param_groups:
                param_group['lr'] /= lr_reduction
        return loss_track

    def print_learned_equation(self):
        """Print the learned governing equation."""
        feature_names = self.feature_library.get_feature_names()
        learned_coeffs = self.coeffs.linear.weight.detach().numpy()
        string_list = [f'{"+" if coeff > 0 else ""}{coeff:.3f} {feat}' 
                       for coeff, feat in zip(np.squeeze(learned_coeffs, axis=1), feature_names) 
                       if np.abs(coeff) > 1e-5]
        equation = " ".join(string_list)
        if equation and equation[0] == "+":
            equation = equation[1:]
        return equation