import math
import typing

import numpy as np
import torch


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    return fn_input * torch.tanh(torch.nn.functional.softplus(fn_input))


class Mish(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return mish(fn_input)


class WeightDropConv(torch.nn.Module):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size=1, bias=True, weight_dropout=0.1, groups=1,
                 padding=0, dilation=1, function=torch.nn.functional.conv1d, stride=1):
        super().__init__()
        self.weight_dropout = weight_dropout
        if in_features % groups != 0:
            print(f"[ERROR] Unable to get weight for in={in_features},groups={groups}. Make sure they are divisible.")
        if out_features % groups != 0:
            print(f"[ERROR] Unable to get weight for out={out_features},groups={groups}. Make sure they are divisible.")
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features // groups, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.stride = stride
        self._function = function

    def forward(self, fn_input):
        return self._function(fn_input,
                              torch.nn.functional.dropout(self.weight, self.weight_dropout, self.training),
                              bias=self.bias,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups,
                              stride=self.stride)


class SeparableConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size: typing.Union[int, tuple],
                 padding: typing.Union[int, tuple] = 0, dilation: typing.Union[int, tuple] = 1,
                 bias=False, dim=1, stride=1):
        super(SeparableConvolution, self).__init__()
        self.depthwise = kernel_size > 1
        function = getattr(torch.nn.functional, f'conv{dim}d')
        norm = getattr(torch.nn, f'BatchNorm{dim}d')
        if self.depthwise:
            self.depthwise_conv = WeightDropConv(in_features, in_features,
                                                 kernel_size,
                                                 padding=padding,
                                                 groups=in_features,
                                                 dilation=dilation,
                                                 bias=False,
                                                 function=function,
                                                 stride=stride)
            self.mid_norm = norm(in_features)
        self.pointwise_conv = WeightDropConv(in_features, out_features, 1, bias=bias, function=function)
        self.str = (f'SeparableConvolution({in_features}, {out_features}, {kernel_size}, '
                    + f'dilation={dilation}, padding={padding})')

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        if self.depthwise:
            fn_input = self.mid_norm(self.depthwise_conv(fn_input))
        return self.pointwise_conv(fn_input)

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.str


class BasicBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, stride, init_norm=False):
        super(BasicBlock, self).__init__()
        self.init_norm = torch.nn.BatchNorm2d(out_features) if init_norm else None
        self.init_conv = SeparableConvolution(in_features, out_features, 3, 1, stride=stride, dim=2)
        self.mid_norm = torch.nn.BatchNorm2d(out_features)
        self.end_conv = SeparableConvolution(in_features, out_features, 3, 1, dim=2)
        self.shortcut = (None
                         if stride == 1 and in_features == out_features
                         else SeparableConvolution(in_features, out_features, 3, 1, stride=stride, dim=2))

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out = self.init_conv(fn_input if self.init_norm is None else mish(self.init_norm(fn_input)))
        out = mish(self.mid_norm(out))
        out = self.end_conv(out)
        if self.shortcut is not None:
            fn_input = self.shortcut(fn_input)
        out = out + fn_input
        return out


class ConvNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_factor=15, depth=4, kernel_size=7, squeeze_heads=4, cat=True,
                 debug=True):
        super(ConvNetwork, self).__init__()
        hidden_size = 11 * hidden_factor
        self.net = torch.nn.ModuleList([BasicBlock(state_size, hidden_size, 1),
                                        *[BasicBlock(hidden_size, hidden_size, 2 - i % 2, True)
                                          for i in range(depth)]])
        self.init_norm = torch.nn.BatchNorm1d(hidden_size)
        self.linear0 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.mid_norm = torch.nn.BatchNorm1d(hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out = fn_input
        for module in self.net:
            out = module(out)
        out = out.mean((2, 3))
        out = self.linear1(mish(self.mid_norm(self.linear0(mish(self.init_norm(out))))))
        return out


def init(module: torch.nn.Module):
    if hasattr(module, "weight") and hasattr(module.weight, "data"):
        if "norm" in module.__class__.__name__.lower() or (
                hasattr(module, "__str__") and "norm" in str(module).lower()):
            torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
        else:
            torch.nn.init.orthogonal_(module.weight.data)
    if hasattr(module, "bias") and hasattr(module.bias, "data"):
        torch.nn.init.constant_(module.bias.data, 0)

class Residual(torch.nn.Module):
    def __init__(self, features):
        super(Residual, self).__init__()
        self.norm = torch.nn.BatchNorm1d(features)
        self.conv = WeightDropConv(features, 2 * features)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out, exc = self.conv(mish(self.norm(fn_input))).chunk(2, 1)
        exc = exc.mean(dim=1, keepdim=True).tanh()
        fn_input = fn_input * exc
        out = -out * exc
        return fn_input + out


def QNetwork(state_size, action_size, hidden_factor=16, depth=4, kernel_size=7, squeeze_heads=4, cat=False,
             debug=True):
    model = torch.nn.Sequential(torch.nn.Conv1d(2 * state_size, 11 * hidden_factor, 1, groups=11, bias=False),
                                Residual(11 * hidden_factor),
                                torch.nn.BatchNorm1d(11 * hidden_factor),
                                Mish(),
                                WeightDropConv(11 * hidden_factor, action_size, 1))
    print(model)
    if debug:
        parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters()))
        digits = int(math.log10(parameters))
        number_string = " kMGTPEZY"[digits // 3]

        print(f"[DEBUG/MODEL] Training with {parameters * 10 ** -(digits // 3 * 3):.1f}{number_string} parameters")
    model.apply(init)
    try:
        model = torch.jit.script(model)
    except TypeError:
        pass
    return model
