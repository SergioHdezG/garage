import numpy as np
from garage.torch.policies.stochastic_policy import StochasticPolicy
import torch
import akro
from torch import nn
from garage import InOutSpec
from garage.torch.modules import CNNModule, MultiHeadedMLPModule
class RandomPolicy(StochasticPolicy):
    """Random Policy"""

    def __init__(self,
                 env_spec,
                 image_format,
                 kernel_sizes,
                 *,
                 hidden_channels,
                 strides=1,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 paddings=0,
                 padding_mode='zeros',
                 max_pool=False,
                 pool_shape=None,
                 pool_stride=1,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='CategoricalCNNPolicy'):

        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError('CategoricalMLPPolicy only works '
                             'with akro.Discrete action space.')
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError('CNN policies do not support '
                             'with akro.Dict observation spaces.')

        super().__init__(env_spec, name)

        self._cnn_module = CNNModule(InOutSpec(
            self._env_spec.observation_space, None),
            image_format=image_format,
            kernel_sizes=kernel_sizes,
            strides=strides,
            hidden_channels=hidden_channels,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            hidden_nonlinearity=hidden_nonlinearity,
            paddings=paddings,
            padding_mode=padding_mode,
            max_pool=max_pool,
            pool_shape=pool_shape,
            pool_stride=pool_stride,
            layer_normalization=layer_normalization)
        self._mlp_module = MultiHeadedMLPModule(
            n_heads=1,
            input_dim=self._cnn_module.spec.output_space.flat_dim,
            output_dims=[self._env_spec.action_space.flat_dim],
            hidden_sizes=hidden_sizes,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            hidden_nonlinearity=hidden_nonlinearity,
            output_w_inits=output_w_init,
            output_b_inits=output_b_init)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Observations to act on.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """
        # We're given flattened observations.
        observations = observations.reshape(
            -1, *self._env_spec.observation_space.shape)
        cnn_output = self._cnn_module(observations)
        mlp_output = self._mlp_module(cnn_output)[0]
        logits = torch.softmax(mlp_output, axis=1)
        dist = torch.distributions.Categorical(torch.Tensor([0.33, 0.33, 0.33]))
        return dist, {}
