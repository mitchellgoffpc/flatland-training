import typing

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
        self._kwargs = {'bias': self.bias, 'padding': padding, 'dilation': dilation, 'groups': groups, 'stride': stride}
        self._function = function

    def forward(self, fn_input):
        if self.training:
            weight = self.weight.bernoulli(p=self.weight_dropout) * self.weight
        else:
            weight = self.weight

        return self._function(fn_input, weight, **self._kwargs)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


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


def try_norm(tensor, norm):
    if norm is not None:
        tensor = mish(norm(tensor))
    return tensor


def make_excite(conv, rank_norm, ranker, linear_norm, linear0, excite_norm, linear1):
    @torch.jit.script
    def excite(out):
        batch = out.size(0)

        exc = conv(out)

        squeeze_heads = exc.size(1)

        exc = torch.nn.functional.softmax(exc, 2)
        exc = exc.unsqueeze(-1).transpose(1, -1)
        exc = (out.unsqueeze(-1) * exc).sum(2)

        # Rank experts (heads)
        hds = exc.view(batch, squeeze_heads, -1)
        exc = rank_norm(hds)
        exc = ranker(mish(exc))
        exc = exc.softmax(-1)
        exc = exc.bmm(hds)
        exc = exc.view(batch, -1, 1)

        # Fully-connected block
        nrm = linear_norm(exc).squeeze(-1)
        nrm = linear0(nrm).unsqueeze(-1)
        nrm = excite_norm(nrm)
        act = mish(nrm.squeeze(-1))
        exc = linear1(act).tanh()
        exc = exc.unsqueeze(-1)
        exc = exc.expand_as(out)

        # Merge
        out = out * exc
        return out

    return excite


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
            self.excite = make_excite(self.excitation_conv,
                                      self.exc_input_norm,
                                      self.expert_ranker,
                                      self.linear_in_norm,
                                      self.linear0,
                                      self.exc_norm,
                                      self.linear1)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        fn_input = try_norm(fn_input, self.init_norm)
        out = self.linr(fn_input)
        out = try_norm(out, self.out_norm)

        if self.use_squeeze_attention:
            out = self.excite(out)

        if self.cat:
            return torch.cat([out, fn_input], 1)

        if self.residual:
            return out + fn_input

        return out


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


# class QNetwork(torch.nn.Module):
#     def __init__(self, state_size, action_size, hidden_factor=15, depth=4, kernel_size=7, squeeze_heads=4, cat=False,
#                  debug=True):
#         """
#         11 input features, state_size//11 = item_count
#         :param state_size:
#         :param action_size:
#         :param hidden_factor:
#         :param depth:
#         :return:
#         """
#         super(QNetwork, self).__init__()
#         observations = state_size // 11
#         if debug:
#             print(f"[DEBUG/MODEL] Using {observations} observations as input")
#
#         out_features = hidden_factor * 11
#
#         net = torch.nn.ModuleList([torch.nn.Conv1d(state_size, out_features, 1),
#                                    *[Block(out_features + out_features * i * cat,
#                                            out_features,
#                                            cat=cat,
#                                            init_norm=not i,
#                                            kernel_size=kernel_size,
#                                            squeeze_heads=squeeze_heads)
#                                      for i in range(depth)],
#                                    Block(out_features + out_features * depth * cat, action_size,
#                                          bias=True,
#                                          cat=False,
#                                          out_norm=False,
#                                          init_norm=False,
#                                          kernel_size=kernel_size,
#                                          squeeze_heads=squeeze_heads)])
#
#         def init(module: torch.nn.Module):
#             if hasattr(module, "weight") and hasattr(module.weight, "data"):
#                 if "norm" in module.__class__.__name__.lower() or (
#                         hasattr(module, "__str__") and "norm" in str(module).lower()):
#                     torch.nn.init.uniform_(module.weight.data, 0.998, 1.002)
#                 else:
#                     torch.nn.init.orthogonal_(module.weight.data)
#             if hasattr(module, "bias") and hasattr(module.bias, "data"):
#                 torch.nn.init.constant_(module.bias.data, 0)
#
#         net.apply(init)
#
#         if debug:
#             parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, net.parameters()))
#             digits = int(math.log10(parameters))
#             number_string = " kMGTPEZY"[digits // 3]
#
#             print(
#                 f"[DEBUG/MODEL] Training with {parameters * 10 ** -(digits // 3 * 3):.1f}{number_string} parameters")
#
#         self.net = net
#
#     def forward(self, fn_input: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
#         out = fn_input
#         for module in self.net:
#             out = module(out)
#         return out

def QNetwork(state_size, action_size, hidden_factor=15, depth=4, kernel_size=7, squeeze_heads=4, cat=False,
             debug=True):
    # model = torch.nn.Sequential(WeightDropConv(state_size + 1, 11 * hidden_factor, bias=False),
    #                             torch.nn.BatchNorm1d(11 * hidden_factor),
    #                             Mish(),
    #                             WeightDropConv(11 * hidden_factor, action_size))
    model = torch.nn.Sequential(torch.nn.Conv1d(state_size + 1, 20, 1, bias=False),
                                torch.nn.BatchNorm1d(20),
                                torch.nn.ReLU6(),
                                torch.nn.Conv1d(20, action_size, 1))
    print(model)
    return torch.jit.script(model)
