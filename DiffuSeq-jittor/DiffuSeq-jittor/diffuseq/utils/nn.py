"""
Various utilities for neural networks.
"""
import jittor as jt
import jittor.nn as nn

class SiLU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def execute(self, x):
        return super().execute(x.float()).type(x.dtype)
    
def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool2d(kernel_size=(1, kwargs.get('kernel_size')[0]), stride=(1, kwargs.get('stride')[0]), padding=(0, kwargs.get('padding')[0]))
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().assign(targ.multiply(rate).add(src.multiply(1-rate)))


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().assign(jt.zeros_like(p))
    return module


def scale_module(module: nn.Module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().assign(p.multiply(scale))
    return module


def mean_flat(tensor: jt.Var):
    """
    Take the mean over all non-batch dimensions.
    """
    non_batch_dims = jt.NanoVector(list(range(1, len(tensor.shape))))
    # 对这些维度取平均值
    return tensor.mean(dims=non_batch_dims)

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps: jt.Var, dim: int, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jt.exp(
        -jt.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    )
    args = timesteps[:, None].float32() * freqs[None]
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding