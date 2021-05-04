from typing import Any, Dict, List, Tuple

import gym
import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn


def same_padding_1d(
    in_size: int,
    filter_size: int,
    stride_size: int,
) -> (int, Tuple[int, int]):
    """Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter

    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_width = in_size
    filter_width = filter_size
    stride_width = stride_size

    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right)
    output = out_width
    return padding, output


class CustomFC(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

    def sum_params(self):
        s = 0
        for p in self.parameters():
            s += p.sum()
        return s.item()


class SlimConv1d(nn.Module):
    """Simple mock of tf.slim Conv2d"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel: int,
            stride: int,
            padding: Tuple[int, int],
            # Defaulting these to nn.[..] will break soft torch import.
            initializer: Any = "default",
            activation_fn: Any = "default",
            bias_init: float = 0):
        """Creates a standard Conv2d layer, similar to torch.nn.Conv2d

            Args:
                in_channels(int): Number of input channels
                out_channels (int): Number of output channels
                kernel (Union[int, Tuple[int, int]]): If int, the kernel is
                    a tuple(x,x). Elsewise, the tuple can be specified
                stride (Union[int, Tuple[int, int]]): Controls the stride
                    for the cross-correlation. If int, the stride is a
                    tuple(x,x). Elsewise, the tuple can be specified
                padding (Union[int, Tuple[int, int]]): Controls the amount
                    of implicit zero-paddings during the conv operation
                initializer (Any): Initializer function for kernel weights
                activation_fn (Any): Activation function at the end of layer
                bias_init (float): Initalize bias weights to bias_init const
        """
        super().__init__()
        layers = []
        # Padding layer.
        if padding:
            layers.append(nn.ConstantPad1d(padding, 0))
        # Actual Conv1D layer (including correct initialization logic).
        conv = nn.Conv1d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        # Activation function (if any; default=ReLu).
        if isinstance(activation_fn, str):
            if activation_fn == "default":
                activation_fn = nn.ReLU
            else:
                activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        if x.dim() == 4:
            x = torch.squeeze(x, dim=2)
        return self._model(x)


class CustomCNN(TorchModelV2, nn.Module):
    """Generic vision network."""
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        if not model_config.get("conv_filters"):
            raise ValueError("Config for conv_filters is required")
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0,\
            "Must provide at least 1 entry in `conv_filters`!"

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None
        # Holds the current "base" output (before logits layer).
        self._features = None
        self.num_outputs = num_outputs if num_outputs else action_space.shape[0]
        self.filters = filters
        self.activation = activation
        self.obs_space = obs_space

        self._create_model()

    def _create_model(self):
        filters = self.filters
        activation = self.activation
        branches = {}
        for obs_name, space in self.obs_space.original_space.spaces.items():
            layers = []
            w, in_channels = space.shape
            in_size = w
            for i, (out_channels, kernel, stride) in enumerate(filters):
                padding, out_size = same_padding_1d(in_size, kernel, stride)
                layers.append(
                    SlimConv1d(in_channels,
                               out_channels,
                               kernel,
                               stride,
                               None if i == (len(filters) - 1) else padding,
                               activation_fn=activation))
                in_channels = out_channels
                in_size = out_size

            branches[obs_name] = nn.Sequential(*layers)

        self._convs = nn.ModuleDict(branches)

        out_channels *= len(self._convs)
        # num_outputs defined. Use that to create an exact
        # `num_output`-sized (1,1)-Conv2D.
        if self.num_outputs:
            in_size = np.ceil((in_size - kernel) / stride)

            padding, _ = same_padding_1d(in_size, 1, 1)
            self._logits = SlimConv1d(out_channels,
                                      self.num_outputs,
                                      1,
                                      1,
                                      padding,
                                      activation_fn=None)
        # num_outputs not known -> Flatten, then set self.num_outputs
        # to the resulting number of nodes.
        else:
            self.last_layer_is_flattened = True
            layers.append(nn.Flatten())
            self.num_outputs = out_channels

        # Build the value layers
        self._value_branch = SlimFC(out_channels,
                                    1,
                                    initializer=normc_initializer(0.01),
                                    activation_fn=None)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str,
                                       TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        features = []
        for obs_name, obs in input_dict['obs'].items():
            features.append(self._convs[obs_name](obs.permute(0, 2, 1)))
        self._features = torch.cat(features, dim=1)

        logits = self._logits(self._features).squeeze(2)
        assert logits.shape[0] == input_dict['obs_flat'].shape[0]

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        features = self._features.squeeze(2)
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 2, 1))  # switch to channel-major
        res = res.squeeze(2)
        return res


class SemanticNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        if not model_config.get("fcnet_hiddens"):
            raise ValueError("Config for fcnet_hiddens is required")
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = self.model_config.get("fcnet_activation")
        hidden_sizes = self.model_config["fcnet_hiddens"]
        assert len(hidden_sizes) > 0,\
            "Must provide at least 1 entry in `fcnet_hiddens`!"

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None
        # Holds the current "base" output (before logits layer).
        self._features = None
        self.num_outputs = num_outputs
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.obs_space = obs_space

        self._create_model()

    def _create_model(self):
        activation = self.activation
        layers = []
        in_channels = sum([
            v.shape[0] for k, v in
            self.obs_space.original_space.child_space.spaces.items()
        ])
        for i, out_channels in enumerate(self.hidden_sizes):
            layers.append(
                SlimFC(in_channels,
                       out_channels,
                       initializer=normc_initializer(0.01),
                       activation_fn=activation))
            in_channels = out_channels

        self._layers = nn.Sequential(*layers)
        self.feat_size = out_channels

        # num_outputs defined. Use that to create an exact
        # `num_output`-sized mlp.
        if self.num_outputs:
            self._logits = SlimFC(out_channels,
                                  self.num_outputs,
                                  initializer=normc_initializer(0.01),
                                  activation_fn=None)
        # num_outputs not known -> Flatten, then set self.num_outputs
        # to the resulting number of nodes.
        else:
            self.last_layer_is_flattened = True
            layers.append(nn.Flatten())
            self.num_outputs = out_channels

        # Build the value layers
        self._value_branch = SlimFC(out_channels,
                                    1,
                                    initializer=normc_initializer(0.01),
                                    activation_fn=None)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str,
                                       TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        features = []
        for elements in input_dict['obs'].unbatch_all():
            if elements:
                elem_tensor = []
                for elem in elements:
                    elem_tensor.append(torch.cat([v for k, v in elem.items()]))
                elem_tensor = torch.stack(elem_tensor)
                print(elem_tensor)
                features.append(
                    torch.mean(self._layers(elem_tensor), dim=0, keepdim=True))
            else:
                features.append(torch.zeros((1, self.feat_size)))

        self._features = torch.cat(features, dim=0)

        logits = self._logits(self._features)
        assert logits.shape[0] == input_dict['obs_flat'].shape[0]

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        features = self._features
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._layers(obs)
        return res


ModelCatalog.register_custom_model("custom-fc", CustomFC)
ModelCatalog.register_custom_model("custom-cnn", CustomCNN)
ModelCatalog.register_custom_model("semantic", SemanticNetwork)
