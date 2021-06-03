import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi
import torch.distributed as dist
import numpy as np

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            if dist.is_initialized():
                input_gather = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
                dist.all_gather(input_gather, input)
                input = torch.cat(input_gather, dim=0)
            mean = input.mean(dim=[0, 2, 3])[None, :, None, None]
            std = input.std(dim=[0, 2, 3])[None, :, None, None]

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        return self.scale * (input + self.loc), logdet

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNormIdentity(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, input):
        log_det = 0.
        return input, log_det

    def reverse(self, output):
        return output


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle=False):
        super().__init__()
        self.num_channels = num_channels
        indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        for i in range(self.num_channels):
            indices_inverse[indices[i]] = i
        if shuffle:
            np.random.shuffle(indices)
            for i in range(self.num_channels):
                indices_inverse[indices[i]] = i
        self.register_buffer('indices', torch.from_numpy(indices))
        self.register_buffer('indices_inverse', torch.from_numpy(indices_inverse))

    def forward(self, input):
        assert len(input.size()) == 4
        det = 0.
        return input[:, self.indices, :, :], det

    def reverse(self, input):
        assert len(input.size()) == 4
        return input[:, self.indices_inverse, :, :]


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
                height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().float().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        w_p, w_l, w_u = torch.lu_unpack(*q.lu())
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.size(0)))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().float().inverse().unsqueeze(2).unsqueeze(3))


class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.scale = nn.Parameter(torch.zeros(out_channel))
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, input):
        out = self.linear(input)
        out = out * torch.exp(self.scale * 3)

        return out


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AdditiveCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        net_out = self.net(in_a)
        out_b = in_b + net_out
        logdet = 0.

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        net_out = self.net(out_a)
        in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        log_s, t = self.net(in_a).chunk(2, 1)
        # s = torch.exp(log_s)
        s = F.sigmoid(log_s + 2)
        # out_a = s * in_a + t
        out_b = (in_b + t) * s

        logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        log_s, t = self.net(out_a).chunk(2, 1)
        s = F.sigmoid(log_s + 2)
        in_b = out_b / s - t

        return torch.cat([out_a, in_b], 1)


class FlowStep(nn.Module):
    def __init__(self,
                 in_channel,
                 norm_layer,
                 permute_layer,
                 coupling_layer):
        super().__init__()

        self.actnorm = norm_layer(in_channel)
        self.invconv = permute_layer(in_channel)
        self.coupling = coupling_layer(in_channel)

    def forward(self, input):
        out, det1 = self.actnorm(input)
        out, det2 = self.invconv(out)
        out, det3 = self.coupling(out)

        log_det = det1 + det1 + det2

        return out, log_det

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_std):
    return -0.5 * log(2 * pi) - log_std - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)


def gaussian_sample(eps, mean, log_std):
    return mean + torch.exp(log_std) * eps


def squeeze2d(input):
    b_size, n_channel, height, width = input.shape
    squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
    out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
    return out


def unsqueeze2d(input):
    b_size, n_channel, height, width = input.shape

    unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
    unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
    unsqueezed = unsqueezed.contiguous().view(
        b_size, n_channel // 4, height * 2, width * 2
    )

    return unsqueezed


class FlowStage(nn.Module):
    def __init__(self,
                 z_shape,
                 n_flow,
                 norm_layer,
                 permute_layer,
                 coupling_layer,
                 dim_condition,
                 split=True):
        super().__init__()

        in_channel = z_shape[0]

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(FlowStep(
                squeeze_dim,
                norm_layer,
                permute_layer,
                coupling_layer))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            if dim_condition > 0:
                self.prior = ZeroLinear(dim_condition, np.prod(z_shape) * 8)
            else:
                self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input, condition=None):

        out = squeeze2d(input)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(out.size(0), -1).sum(1)

        else:
            if condition is not None:
                mean, log_std = self.prior(condition.float()).chunk(2, dim=1)
                mean, log_std = mean.view(out.size()), log_std.view(out.size())
            else:
                zero = torch.zeros_like(out)
                mean, log_std = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_std)
            log_p = log_p.view(out.size(0), -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False, condition=None):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                if condition is None:
                    zero = torch.zeros_like(input)
                    print(zero.size(), self.prior)
                    mean, log_std = self.prior(zero).chunk(2, 1)
                else:
                    mean, log_std = self.prior(condition.float()).chunk(2, dim=1)
                    mean, log_std = mean.view(input.size()), log_std.view(input.size())
                z = gaussian_sample(eps, mean, log_std)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        return unsqueeze2d(input)


class Flow(nn.Module):
    def __init__(self,
                 in_channel,
                 img_size,
                 n_flow,
                 n_block,
                 norm_layer,
                 permute_layer,
                 coupling_layer,
                 dim_condition=0,
                 n_bits=5,
                 ):
        super().__init__()

        self.n_bits = n_bits
        self.dim_condition = dim_condition

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            size = img_size // (2 ** (i + 1))
            self.blocks.append(FlowStage(z_shape=[n_channel, size, size],
                                         n_flow=n_flow,
                                         norm_layer=norm_layer,
                                         permute_layer=permute_layer,
                                         coupling_layer=coupling_layer,
                                         dim_condition=dim_condition))
            n_channel *= 2
        self.blocks.append(FlowStage(z_shape=[n_channel, size // 2, size // 2],
                                     n_flow=n_flow,
                                     norm_layer=norm_layer,
                                     permute_layer=permute_layer,
                                     coupling_layer=coupling_layer,
                                     dim_condition=dim_condition,
                                     split=False))

    def forward(self, input, condition=None, continuous=False):
        log_p_sum = 0
        log_det_sum = 0
        z_outs = []

        input = self.preprocess(input)
        if continuous:
            input.add_(torch.rand_like(input) / (2 ** self.n_bits))
        out = input

        for i, block in enumerate(self.blocks):
            out, det, log_p, z_new = block(out,
                                           condition if (i == len(self.blocks) - 1) else None)
            z_outs.append(z_new)
            log_det_sum += det
            log_p_sum += log_p

        return log_p_sum, log_det_sum, z_outs

    def preprocess(self, image):
        image = image.mul_(255.)

        if self.n_bits < 8:
            image.div_(2 ** (8 - self.n_bits))
            image.floor_()
            # print(image)
        image.div_(2.0 ** self.n_bits).sub_(0.5)
        return image

    def postprocess(self, image):
        image.add_(0.5).mul_(2 ** self.n_bits).floor_()
        image.mul_(2 ** (8 - self.n_bits))
        image.clamp_(0., 255.)
        image.div_(255.)
        return image

    def reverse(self, z_list, reconstruct=False, condition=None):
        output = z_list[-1]
        for i, block in enumerate(self.blocks[::-1]):
            condition = (condition if (i == 0) else None)
            output = block.reverse(output, z_list[-(i + 1)], reconstruct=reconstruct, condition=condition)
        self.postprocess(output)
        return output


from functools import partial

Glow = partial(Flow,
               norm_layer=ActNorm,
               permute_layer=InvConv2dLU,
               coupling_layer=AdditiveCoupling)
