import typing

import torch


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    return fn_input * torch.tanh(torch.nn.functional.softplus(fn_input))


@torch.jit.script
def nothing(x):
    return x


class Mish(torch.nn.Module):
    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return mish(fn_input)


class SeparableConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size: typing.Union[int, tuple],
                 padding: typing.Union[int, tuple] = 0, dilation: typing.Union[int, tuple] = 1,
                 bias=False, dim=1, stride: typing.Union[int, tuple] = 1, dropout=0.25):
        super(SeparableConvolution, self).__init__()
        self.depthwise = kernel_size > 1 if isinstance(kernel_size, int) else any(k > 1 for k in kernel_size)
        conv = getattr(torch.nn, f'Conv{dim}d')
        norm = getattr(torch.nn, f'InstanceNorm{dim}d')
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * dim
        if self.depthwise:
            self.depthwise_conv = conv(in_features, in_features,
                                       kernel_size,
                                       padding=padding,
                                       groups=in_features,
                                       dilation=dilation,
                                       bias=False,
                                       stride=stride)
            self.mid_norm = norm(in_features, affine=True)
        else:
            self.depthwise_conv = nothing
            self.mid_norm = nothing
        self.pointwise_conv = conv(in_features, out_features, (1,) * dim, bias=bias)
        self.str = (f'SeparableConvolution({in_features}, {out_features}, {kernel_size}, '
                    + f'dilation={dilation}, padding={padding}, stride={stride})')
        self.dropout = dropout * (in_features == out_features and (stride == 1 or all(s == 1 for s in stride)))

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.dropout:
            return fn_input
        if self.depthwise:
            fn_input = self.mid_norm(self.depthwise_conv(fn_input))
        return self.pointwise_conv(fn_input)

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.str


class BasicBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, stride, init_norm=False, message_box=None, double=True,
                 agent_dim=True):
        super(BasicBlock, self).__init__()
        self.activate = init_norm
        self.double = double
        self.agent_dim = agent_dim
        dim = 2 + agent_dim
        norm = getattr(torch.nn, f'InstanceNorm{dim}d')
        kernel = (3, 3) + ((1,) if agent_dim else ())
        pad = (1, 1) + ((0,) if agent_dim else ())
        stride = (stride, stride) + ((1,) if agent_dim else ())
        self.init_norm = norm(in_features, affine=True) if init_norm else nothing
        self.init_conv = SeparableConvolution(in_features, out_features, kernel, pad, stride=stride, dim=dim)
        if double:
            self.mid_norm = norm(out_features, affine=True)
            self.end_conv = SeparableConvolution(out_features, out_features, kernel, pad, dim=dim)
        else:
            self.mid_norm = nothing
            self.end_conv = nothing
        self.shortcut = torch.nn.Sequential()
        if stride[0] > 1:
            self.shortcut.add_module("1", getattr(torch.nn, f"MaxPool{dim}d")(kernel, stride, padding=pad))
        if in_features != out_features:
            self.shortcut.add_module("2", getattr(torch.nn, f"Conv{dim}d")(in_features, out_features, 1))
        self.message_box = int(out_features ** 0.5) if message_box is None else message_box

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out = self.init_conv(fn_input if self.activate is None else mish(self.init_norm(fn_input)))
        if self.double:
            out = self.end_conv(mish(self.mid_norm(out)))
        if self.agent_dim and self.message_box > 0:
            out[:, :self.message_box] = out[:, :self.message_box].mean(-1,
                                                                       keepdim=True).expand(-1, -1, -1, -1,
                                                                                            fn_input.size(-1))
        srt = self.shortcut(fn_input)
        out = out + srt
        return out


class ConvNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=15, depth=8, kernel_size=7, squeeze_heads=4,
                 decoder_depth=1, cat=True, debug=True, softmax=False):
        super(ConvNetwork, self).__init__()
        _ = state_size
        state_size = 2 * 21
        self.net = torch.nn.ModuleList([BasicBlock(state_size if not i else hidden_size,
                                                   hidden_size,
                                                   2,
                                                   init_norm=bool(i))
                                        for i in range(depth)])
        self.init_norm = torch.nn.InstanceNorm1d(hidden_size, affine=True)
        self.linear = torch.nn.Conv1d(hidden_size, action_size, 1)
        self.softmax = softmax

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out = fn_input
        for module in self.net:
            out = module(out)
        out = out.mean((2, 3))
        out = self.linear(mish(self.init_norm(out)))
        return torch.nn.functional.softmax(out, 1) if self.softmax else out

    @torch.jit.export
    def reset_cache(self):
        pass


class DecoderBlock(torch.nn.Module):
    def __init__(self, features, message_box=None):
        super(DecoderBlock, self).__init__()
        self.norm = torch.nn.InstanceNorm1d(features, affine=True)
        self.conv = torch.nn.Conv1d(features, features, 1, bias=False)
        self.message_box = int(features ** 0.5) if message_box is None else message_box

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out = self.norm(fn_input)
        out = mish(out)
        out = self.conv(out)
        if self.message_box > 0:
            out[:, :self.message_box] = out[:, :self.message_box].mean(-1, keepdim=True).expand(-1, -1, fn_input.size(-1))
        return out + fn_input


class GlobalStateNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=15, depth=8, kernel_size=7, squeeze_heads=4,
                 decoder_depth=1, cat=True, debug=True, softmax=False):
        super(GlobalStateNetwork, self).__init__()
        _ = state_size
        _ = kernel_size
        _ = squeeze_heads
        _ = cat
        _ = debug
        global_state_size = 2 * 16
        agent_state_size = 2 * 13

        self.net = torch.nn.Sequential(*[BasicBlock(global_state_size if not i else hidden_size,
                                                    hidden_size,
                                                    2,
                                                    init_norm=bool(i),
                                                    message_box=0,
                                                    double=False,
                                                    agent_dim=False)
                                         for i in range(depth)])
        self.decoder = torch.nn.Sequential(torch.nn.Conv1d(hidden_size + agent_state_size, hidden_size, 1, bias=False),
                                           *[DecoderBlock(hidden_size) for i in range(decoder_depth)],
                                           torch.nn.InstanceNorm1d(hidden_size, affine=True),
                                           torch.nn.Conv1d(hidden_size, action_size, 1))
        self.softmax = softmax

        self.register_buffer("base_zero", torch.zeros(1))
        self.encoding_cache = self.base_zero

    @torch.jit.export
    def reset_cache(self):
        self.encoding_cache = self.base_zero

    def forward(self, state, rail) -> torch.Tensor:
        if torch.equal(self.encoding_cache, self.base_zero):
            inp = self.net(rail)
            inp = inp.mean((2, 3), keepdim=True).squeeze(-1)
            self.encoding_cache = inp
        else:
            inp = self.encoding_cache
        inp = torch.cat([inp.expand(-1, -1, state.size(-1)), state], 1)
        out = self.decoder(inp)
        return torch.nn.functional.softmax(out, 1) if self.softmax else out


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
        self.conv = torch.nn.Conv1d(features, 2 * features, 1)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out, exc = self.conv(mish(self.norm(fn_input))).chunk(2, 1)
        exc = exc.mean(dim=-1, keepdim=True).tanh()
        fn_input = fn_input * exc
        out = -out * exc
        return fn_input + out


class QNetwork(torch.nn.Sequential):
    def __init__(self, state_size, action_size, hidden_factor=16, depth=4, kernel_size=7, squeeze_heads=4,
                 decoder_depth=1, cat=False, debug=True):
        super(QNetwork, self).__init__()
        _ = depth
        _ = kernel_size
        _ = squeeze_heads
        _ = cat
        _ = debug
        _ = decoder_depth
        self.model = torch.nn.Sequential(torch.nn.Conv1d(2 * state_size, 11 * hidden_factor, 1, groups=11, bias=False),
                                         Residual(11 * hidden_factor),
                                         torch.nn.BatchNorm1d(11 * hidden_factor),
                                         Mish(),
                                         torch.nn.Conv1d(11 * hidden_factor, action_size, 1))

    @torch.jit.export
    def reset_cache(self):
        pass

    def forward(self, *args):
        return self.model(*args)
