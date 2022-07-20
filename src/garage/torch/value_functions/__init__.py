"""Value functions which use PyTorch."""
from garage.torch.value_functions.gaussian_mlp_value_function import (
    GaussianMLPValueFunction)
from garage.torch.value_functions.gaussian_cnn_value_function import \
    GaussianCNNValueFunction
from garage.torch.value_functions.resnet_mlp_value_function import \
    ResNetMLPValueFunction
from garage.torch.value_functions.value_function import ValueFunction

__all__ = ['ValueFunction', 'GaussianMLPValueFunction',
           'GaussianCNNValueFunction', 'ResNetMLPValueFunction']
