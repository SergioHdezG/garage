"""ResNetCNNPolicy."""
import akro
import torch
from torch import nn
from torchvision import models, transforms

from garage.torch.modules import MultiHeadedMLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class ResNetCNNPolicy(StochasticPolicy):
    """ResNetCNNPolicy.

    A policy that contains a CNN initialized as ResNet and a MLP to make
    prediction based on a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (garage.EnvSpec): Environment specification.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match env_spec. Gym
            uses NHWC by default, but PyTorch uses NCHW by default.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 *,
                 freeze=True,
                 hidden_sizes=(512, 256),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 name='ResNetCNNPolicy'):

        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError('CategoricalMLPPolicy only works '
                             'with akro.Discrete action space.')
        if isinstance(env_spec.observation_space, akro.Dict):
            raise ValueError('CNN policies do not support '
                             'with akro.Dict observation spaces.')

        super().__init__(env_spec, name)

        self.freeze = freeze
        self._resnet_module = models.resnet18(pretrained=True)
        self._preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

        if self.freeze:
            self._resnet_module.eval()
            for param in self._resnet_module.parameters():
                param.requires_grad = False

        input_dim = self._resnet_module.fc.in_features

        # Delete last fc layer of resnet
        self._resnet_module = torch.nn.Sequential(
            *(list(self._resnet_module.children())[:-1]))

        self._mlp_module = MultiHeadedMLPModule(
            n_heads=1,
            input_dim=input_dim,
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
        # TODO[Sergio]: he modificado los faltten que se hacen en garage/src/
        #  garage/toch/policies/stochastic_policy.py porque hacer un reshape
        #  aquí puede estar dando problemas.
        # We're given flattened observations.
        # observations = observations.reshape(
        #     -1, *self._env_spec.observation_space.shape)
        # Reshape to be compatible with NCHW
        if len(observations.shape) == 4:
            obs = observations.permute((0, 3, 1, 2))
        elif len(observations.shape) == 5:
            obs = observations.permute((0, 1, 4, 2, 3))
            obs = obs.squeeze(0)

        obs = self._preprocess(obs)
        if torch.cuda.is_available():
            obs.to('cuda')
        resnet_output = self._resnet_module(obs)
        # Delete non batch dimensions
        resnet_output = resnet_output.reshape(resnet_output.shape[0], -1)
        mlp_output = self._mlp_module(resnet_output)[0]
        probs = torch.softmax(mlp_output, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist, {}
