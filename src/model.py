import math
import typing

import numpy as np
import torch


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    return fn_input * torch.tanh(torch.nn.functional.softplus(fn_input))


class WeightDropConv(torch.nn.Module):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size=1, bias=True, weight_dropout=0.1, groups=1,
                 padding=0, dilation=1):
        super().__init__()
        self.weight_dropout = weight_dropout
        self.in_features = in_features
        self.out_features = out_features
        if in_features % groups != 0:
            print(f"[ERROR] Unable to get weight for in={in_features},groups={groups}. Make sure they are divisible.")
        if out_features % groups != 0:
            print(f"[ERROR] Unable to get weight for out={out_features},groups={groups}. Make sure they are divisible.")
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features // groups, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._kwargs = {'bias': self.bias, 'padding': padding, 'dilation': dilation, 'groups': groups}

    def forward(self, fn_input):
        if self.training:
            weight = self.weight.bernoulli(p=self.weight_dropout) * self.weight
        else:
            weight = self.weight

        return torch.nn.functional.conv1d(fn_input, weight, **self._kwargs)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class SeparableConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size: typing.Union[int, tuple],
                 padding: typing.Union[int, tuple] = 0, dilation: typing.Union[int, tuple] = 1,
                 norm=torch.nn.BatchNorm1d, bias=False):
        super(SeparableConvolution, self).__init__()
        self.depthwise_conv = WeightDropConv(in_features, in_features,
                                             kernel_size,
                                             padding=padding,
                                             groups=in_features,
                                             dilation=dilation,
                                             bias=False)
        self.mid_norm = norm(in_features)
        self.pointwise_conv = WeightDropConv(in_features, out_features, 1, bias=bias)
        self.str = (f'SeparableConvolution({in_features}, {out_features}, {kernel_size}, '
                    + f'dilation={dilation}, padding={padding})')

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.mid_norm(self.depthwise_conv(fn_input)))

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.str


def try_norm(tensor, norm):
    if norm is not None:
        tensor = mish(norm(tensor))
    return tensor


class Block(torch.nn.Module):
    def __init__(self, hidden_size, output_size, bias=False, cat=True, init_norm=False, out_norm=True,
                 kernel_size=7, squeeze_heads=4):
        super().__init__()
        self.residual = hidden_size == output_size
        self.cat = cat

        self.init_norm = torch.nn.BatchNorm1d(hidden_size) if init_norm else None
        self.linr = SeparableConvolution(hidden_size, output_size, kernel_size, padding=kernel_size // 2, bias=bias)
        self.out_norm = torch.nn.BatchNorm1d(output_size) if out_norm else None

        self.use_squeeze_attention = squeeze_heads > 0

        if self.use_squeeze_attention:
            self.squeeze_heads = squeeze_heads
            self.exc_input_norm = torch.nn.BatchNorm1d(squeeze_heads)
            self.expert_ranker = torch.nn.Linear(output_size, squeeze_heads, False)
            self.excitation_conv = SeparableConvolution(output_size, squeeze_heads, kernel_size,
                                                        padding=kernel_size // 2)
            self.linear_in_norm = torch.nn.BatchNorm1d(output_size * squeeze_heads)
            self.linear0 = torch.nn.Linear(output_size * squeeze_heads, output_size, False)
            self.exc_norm = torch.nn.BatchNorm1d(output_size)
            self.linear1 = torch.nn.Linear(output_size, output_size)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        batch = fn_input.size(0)
        fn_input = try_norm(fn_input, self.init_norm)
        out = self.linr(fn_input)
        out = try_norm(out, self.out_norm)

        if self.use_squeeze_attention:
            exc = self.excitation_conv(out)
            exc = torch.nn.functional.softmax(exc, 2)
            exc = exc.unsqueeze(-1).transpose(1, -1)
            exc = (out.unsqueeze(-1) * exc).sum(2)

            # Rank experts (heads)
            hds = exc.view(batch, self.squeeze_heads, -1)
            exc = self.exc_input_norm(hds)
            exc = self.expert_ranker(mish(exc))
            exc = exc.softmax(-1)
            exc = exc.bmm(hds)
            exc = exc.view(batch, -1, 1)

            # Fully-connected block
            nrm = self.linear_in_norm(exc).squeeze(-1)
            nrm = self.linear0(nrm).unsqueeze(-1)
            nrm = self.exc_norm(nrm)
            act = mish(nrm.squeeze(-1))
            exc = self.linear1(act).tanh()
            exc = exc.unsqueeze(-1)
            exc = exc.expand_as(out)

            # Merge
            out = out * exc

        if self.cat:
            return torch.cat([out, fn_input], 1)

        if self.residual:
            return out + fn_input

        return out


# class Residual(torch.nn.Module):
#     def __init__(self, m1, m2=None):
#         import random
#         super().__init__()
#         self.m1 = m1
#         self.m2 = copy.deepcopy(m1) if m2 is None else m2
#         self.name = f'residual_{str(int(random.randint(0, 2 ** 32)))}'
#
#     def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
#         double = fn_input.size(1) > 1
#         if double:
#             f0, f1 = fn_input.chunk(2, 1)
#             o0 = self.m1(f0)
#             o1 = self.m2(f1)
#             return torch.cat([o0, o1], 1) + fn_input
#         else:
#             return self.m1(fn_input) + self.m2(fn_input) + fn_input
#
#     def __str__(self):
#         return f'{self.__class__.__name__}(ID: {self.name}, M1: {self.m1}, M2: {self.m2})'
#
#     def __repr__(self):
#         return str(self)
#
#
# def layer_split(target_depth, features, split_depth=3, uneven: typing.Union[bool, int] = False):
#     layer_list = []
#
#     if target_depth > split_depth ** 2:
#         for _ in range(split_depth):
#             layer_list.append(layer_split(target_depth // split_depth, features // 2, split_depth, features % 2))
#         layer_list.append(layer_split(target_depth % split_depth, features // 2, split_depth, features % 2))
#     elif target_depth > split_depth:
#         for _ in range(target_depth // split_depth):
#             layer_list.append(layer_split(split_depth, features // 2, split_depth, features % 2))
#         layer_list.append(layer_split(target_depth % split_depth, features // 2, split_depth, features % 2))
#     else:
#         tmp_features = max(2, features)
#         f2, mod = tmp_features // 2, tmp_features % 2
#         layer_list = [Residual(Block(f2 + mod, f2 + mod), Block(f2, f2)) for _ in range(target_depth)]
#     layer = torch.nn.Sequential(*layer_list)
#     features = max(1, features + uneven)
#     layer = Residual(Block(features, features), layer)
#     return layer


class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_factor=15, depth=4, kernel_size=7, squeeze_heads=4, cat=True):
        """
        11 input features, state_size//11 = item_count
        :param state_size:
        :param action_size:
        :param hidden_factor:
        :param depth:
        :return:
        """
        super(QNetwork, self).__init__()
        observations = state_size // 11
        print(f"[DEBUG/MODEL] Using {observations} observations as input")

        out_features = hidden_factor * 11

        net = torch.nn.ModuleList([torch.nn.Conv1d(state_size, out_features, 1),
                                   *[Block(out_features + out_features * i * cat,
                                           out_features,
                                           cat=True,
                                           init_norm=not i,
                                           kernel_size=kernel_size,
                                           squeeze_heads=squeeze_heads)
                                     for i in range(depth)],
                                   Block(out_features + out_features * depth * cat, action_size,
                                         bias=True,
                                         cat=False,
                                         out_norm=False,
                                         init_norm=False,
                                         kernel_size=kernel_size,
                                         squeeze_heads=squeeze_heads)])

        def init(module: torch.nn.Module):
            if hasattr(module, "weight") and hasattr(module.weight, "data"):
                if "norm" in module.__class__.__name__.lower() or (
                        hasattr(module, "__str__") and "norm" in str(module).lower()):
                    torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
                else:
                    torch.nn.init.orthogonal_(module.weight.data)
            if hasattr(module, "bias") and hasattr(module.bias, "data"):
                torch.nn.init.constant_(module.bias.data, 0)

        net.apply(init)

        parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters()))
        digits = int(math.log10(parameters))
        number_string = " kMGTPEZY"[digits // 3]

        print(f"[DEBUG/MODEL] Training with {parameters * 10 ** -(digits // 3 * 3):.1f}{number_string} parameters")

        self.net = net

    def forward(self, fn_input: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        out = fn_input
        for module in self.net:
            out = module(out)
        return out
