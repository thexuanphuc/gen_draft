import torch as T
from .params import Params

def scale_torch(unscaled_tensor, params: Params):
    """Apply standard scaling to a torch tensor."""
    if params.mus is None or params.stds is None:
        raise ValueError("Params must include mus and stds for scaling.")
    return (unscaled_tensor - T.tensor(params.mus)) / T.tensor(params.stds)