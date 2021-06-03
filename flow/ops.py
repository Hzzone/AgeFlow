import torch
from flow.modules import squeeze2d


def compute_z_shapes(
        image_size=256,
        in_channels=3,
        n_stages=6):
    z_shapes = []
    for i in range(n_stages):
        in_channels *= 2

        if i == (n_stages - 1):
            in_channels *= 2
        image_size //= 2
        z_shapes.append([in_channels, image_size, image_size])
    return z_shapes


def sample_z(input_shape=(1, 3, 256, 256), n_stages=6, temp=0.7):
    z = torch.randn(*input_shape) * temp
    z_outs = []
    for _ in range(n_stages - 1):
        x, y = squeeze2d(z).chunk(2, dim=1)
        z = y
        z_outs.append(x)
    z_outs.append(squeeze2d(z))
    return z_outs
