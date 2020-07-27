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


class NAFHead(torch.nn.Module):
    def __init__(self, hidden_size, action_size):
        super(NAFHead, self).__init__()
        self.action_size = action_size
        self.action = torch.nn.Conv1d(hidden_size, action_size, 1)
        self.value = torch.nn.Conv1d(hidden_size, 1, 1)
        self.triangular = torch.nn.Conv1d(hidden_size, action_size ** 2, 1)
        self.diagonal_mask = torch.ones((action_size, action_size)).diag().diag().unsqueeze_(0)

    def forward(self, fn_input, idx, prev_action):
        if idx == 0:
            return self.action(fn_input)
        if idx == 1:
            return self.value(fn_input)

        actions, value = self.action(fn_input), self.value(fn_input)

        batch = fn_input.size(0)
        triangular = self.triangular(fn_input).view(batch, self.action_size, self.action_size, -1)
        triangular = triangular.tril() + triangular.mul(self.diagonal_mask).exp()
        matrix = torch.bmm(triangular, triangular.transpose(1, 2))

        action_difference = (prev_action - actions).unsqueeze(2)
        advantage = torch.bmm(torch.bmm(action_difference.transpose(1, 2), matrix),
                              action_difference)[:, :, 0].div(2).neg()

        return advantage + value


class TripleClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, action_size):
        super(TripleClassificationHead, self).__init__()
        self.linear = torch.nn.Conv1d(hidden_size, action_size, 1)

    def forward(self, fn_input, idx, prev_action):
        _ = idx
        _ = prev_action
        return self.linear(fn_input)


class TailModel(torch.nn.Module):
    def __init__(self, tail):
        super(TailModel, self).__init__()
        self.tail = nothing if tail is None else tail
        self.no_tensor = torch.zeros(1)

    def _backbone(self, fn_input: torch.Tensor, potential_fn_input: torch.Tensor) -> torch.Tensor:
        raise UserWarning("Has to be implemented by child class")

    def forward(self, idx, prev_action, state, rail):
        out = self._backbone(state, rail)
        return self.tail(out, idx, prev_action)

    @torch.jit.export
    def reset_cache(self):
        pass


class ConvNetwork(TailModel):
    def __init__(self, state_size, hidden_size=15, depth=8, kernel_size=7, squeeze_heads=4,
                 decoder_depth=1, cat=True, debug=True, softmax=False, tail=None):
        super(ConvNetwork, self).__init__(tail)
        _ = state_size
        state_size = 2 * 21
        self.net = torch.nn.ModuleList([BasicBlock(state_size if not i else hidden_size,
                                                   hidden_size,
                                                   2,
                                                   init_norm=bool(i))
                                        for i in range(depth)])
        self.init_norm = torch.nn.InstanceNorm1d(hidden_size, affine=True)

    def _backbone(self, fn_input: torch.Tensor, trash: torch.Tensor) -> torch.Tensor:
        _ = trash
        out = fn_input
        for module in self.net:
            out = module(out)
        out = out.mean((2, 3))
        out = mish(self.init_norm(out))
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, message_box=None, init_norm=True):
        super(DecoderBlock, self).__init__()
        self.init_norm = init_norm
        self.residual = in_features == out_features
        self.norm = torch.nn.InstanceNorm1d(in_features, affine=True) if init_norm else nothing
        self.conv = torch.nn.Conv1d(in_features, out_features, 1, bias=False)
        self.message_box = int(out_features ** 0.5) if message_box is None else message_box

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        if self.init_norm:
            out = self.norm(fn_input)
            out = mish(out)
        else:
            out = fn_input
        out = self.conv(out)
        if self.message_box > 0:
            out[:, :self.message_box] = out[:, :self.message_box].mean(-1, keepdim=True).expand(-1, -1,
                                                                                                fn_input.size(-1))
        if self.residual:
            return out + fn_input
        return out


@torch.jit.script
def attention(tensor: torch.Tensor):
    query, key, value = tensor.chunk(3, 1)
    query = query.transpose(1, 2)  # B, F, S -> B, S, F
    key = torch.bmm(query, key).softmax(1)
    value = torch.bmm(value, key)
    return value


class GlobalStateNetwork(TailModel):
    def __init__(self, state_size, hidden_size=15, depth=8, kernel_size=7, squeeze_heads=4,
                 decoder_depth=1, cat=True, debug=True, softmax=False, memory_size=4, tail=None):
        super(GlobalStateNetwork, self).__init__(tail)
        _ = state_size
        _ = kernel_size
        _ = squeeze_heads
        _ = cat
        _ = debug
        global_state_size = 2 * 16
        agent_state_size = 2 * 13

        self.memory_size = memory_size

        self.net = torch.nn.Sequential(*[BasicBlock(global_state_size if not i else hidden_size,
                                                    hidden_size + (i == depth - 1) * hidden_size * 2,
                                                    2,
                                                    init_norm=bool(i),
                                                    message_box=0,
                                                    double=False,
                                                    agent_dim=False)
                                         for i in range(depth)])
        self.decoder_input = DecoderBlock(agent_state_size, 3 * hidden_size, 0, False)
        self.decoder = torch.nn.ModuleList([DecoderBlock(hidden_size, 3 * hidden_size, 0, True)
                                            for _ in range(1, decoder_depth)])
        self.end_norm = torch.nn.InstanceNorm1d(hidden_size, affine=True)

        self.memory_tensor = torch.nn.Parameter(torch.randn(1, 3 * hidden_size, memory_size))

    def _backbone(self, state: torch.Tensor, rail: torch.Tensor) -> torch.Tensor:
        inp = self.net(rail)
        inp = inp.mean((2, 3), keepdim=True).squeeze(-1)
        state = torch.cat([self.decoder_input(state) + inp, self.memory_tensor.expand(inp.size(0), -1, -1)], 2)
        state = attention(state)
        for block in self.decoder:
            state = attention(block(state))
        state = state[:, :, :-self.memory_size]
        out = self.end_norm(state)
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
        self.conv = torch.nn.Conv1d(features, 2 * features, 1)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        out, exc = self.conv(mish(self.norm(fn_input))).chunk(2, 1)
        exc = exc.mean(dim=-1, keepdim=True).tanh()
        fn_input = fn_input * exc
        out = -out * exc
        return fn_input + out


class QNetwork(TailModel):
    def __init__(self, state_size, hidden_factor=16, depth=4, kernel_size=7, squeeze_heads=4,
                 decoder_depth=1, cat=False, debug=True, tail=None):
        super(QNetwork, self).__init__(tail)
        _ = depth
        _ = kernel_size
        _ = squeeze_heads
        _ = cat
        _ = debug
        _ = decoder_depth
        self.model = torch.nn.Sequential(torch.nn.Conv1d(2 * state_size, 11 * hidden_factor, 1, groups=11, bias=False),
                                         Residual(11 * hidden_factor),
                                         torch.nn.BatchNorm1d(11 * hidden_factor),
                                         Mish())

    def _backbone(self, fn_input: torch.Tensor, trash: torch.Tensor) -> torch.Tensor:
        _ = trash
        return self.model(fn_input)
