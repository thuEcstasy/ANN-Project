"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""
import jittor as jt
from numpy import pi




def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, jt.Var):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Var"
    
    
    
    
    
    
    
    
    
    
    
    
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + jt.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * jt.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + jt.tanh(jt.sqrt(2.0 / pi) * (x + 0.044715 * jt.pow(x, 3))))


def discretized_gaussian_log_likelihood(x: jt.Var, *, means: jt.Var, log_scales: jt.Var):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    return discretized_text_log_likelihood(x, means, log_scales)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def gaussian_density(x, *, means: jt.Var, log_scales: jt.Var):
    from jittor.distributions import Normal
    normal_dist = Normal(means, log_scales.exp())
    logp = normal_dist.log_prob(x)
    return logp


def discretized_text_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = jt.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus: jt.Var = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = jt.log(jt.clamp(cdf_plus, min_v=1e-12))
    log_one_minus_cdf_min = jt.log(jt.clamp(jt.Var(1.0 - cdf_min), min_v=1e-12))
    cdf_delta: jt.Var = cdf_plus - cdf_min
    log_probs = jt.where(
        x < -0.999,
        log_cdf_plus,
        jt.where(x > 0.999, log_one_minus_cdf_min, jt.log(jt.clamp(cdf_delta, min_v=1e-12)))
    )
    assert log_probs.shape == x.shape
    return log_probs