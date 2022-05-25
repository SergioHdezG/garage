"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule, CNNModule
from garage.torch.value_functions.value_function import ValueFunction


class GaussianCNNValueFunction(ValueFunction):
    """Gaussian CNN Value Function with Model."""
    pass
