import torch as T

class CoeffsDictionary(T.nn.Module):
    """Class to manage coefficients for sparse identification."""
    def __init__(self, n_combinations, n_eqs):
        super(CoeffsDictionary, self).__init__()
        self.linear = T.nn.Linear(n_combinations, n_eqs, bias=False)
        self.linear.weight = T.nn.Parameter(0 * self.linear.weight.clone().detach())

    def forward(self, x):
        """Forward pass to compute coefficients."""
        return self.linear(x)