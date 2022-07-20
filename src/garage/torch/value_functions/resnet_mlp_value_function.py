"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule
from garage.torch.value_functions.value_function import ValueFunction
from garage.torch.modules import MultiHeadedMLPModule
from torchvision import models, transforms


class ResNetMLPValueFunction(ValueFunction):
    """ResNet MLP Value Function with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
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
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.

    """

    def __init__(self,
                 env_spec,
                 *,
                 freeze=True,
                 hidden_sizes=(128, 128),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 name='ResNetCNNValueFunction',
                 layer_normalization=False,
                 is_image=None):

        super(ResNetMLPValueFunction, self).__init__(env_spec, name)

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

        self.is_image = is_image
        output_dim = 1

        self._mlp_module = MultiHeadedMLPModule(
            n_heads=1,
            input_dim=input_dim,
            output_dims=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            hidden_nonlinearity=hidden_nonlinearity,
            output_w_inits=output_w_init,
            output_b_inits=output_b_init,
            layer_normalization=layer_normalization,
            output_nonlinearities=None  # None => linear
        )
            # output_nonlinearities=output_nonlinearity)

    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        if self.is_image:
            # Flatten the tensor in order to be fed into the value function
            if len(obs.shape) > 4:
                obs = obs.flatten(start_dim=2)
            else:
                obs = obs.flatten(start_dim=1)

        mlp_output = self.module(obs)
        probs = torch.softmax(mlp_output, dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        ll = dist.log_prob(returns.reshape(-1, 1))
        loss = -ll.mean()
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        # We're given flattened observations.
        obs = obs.reshape(
            -1, *self._env_spec.observation_space.shape)
        # Reshape to be compatible with NCHW
        obs = obs.permute((0, 3, 1, 2))
        obs = self._preprocess(obs)
        if torch.cuda.is_available():
            obs.to('cuda')
        resnet_output = self._resnet_module(obs)
        # Delete non batch dimensions
        resnet_output = resnet_output.reshape(resnet_output.shape[0], -1)
        mlp_output = self._mlp_module(resnet_output)[0]

        return mlp_output
