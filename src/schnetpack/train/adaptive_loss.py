import torch
from typing import List, Dict, Optional
import numbers

import numpy as np
import mpmath
import os
from pkg_resources import resource_stream

def interpolate1d(x, values, tangents):
  r"""Perform cubic hermite spline interpolation on a 1D spline.

  The x coordinates of the spline knots are at [0 : 1 : len(values)-1].
  Queries outside of the range of the spline are computed using linear
  extrapolation. See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
  for details, where "x" corresponds to `x`, "p" corresponds to `values`, and
  "m" corresponds to `tangents`.

  Args:
    x: A tensor of any size of single or double precision floats containing the
      set of values to be used for interpolation into the spline.
    values: A vector of single or double precision floats containing the value
      of each knot of the spline being interpolated into. Must be the same
      length as `tangents` and the same type as `x`.
    tangents: A vector of single or double precision floats containing the
      tangent (derivative) of each knot of the spline being interpolated into.
      Must be the same length as `values` and the same type as `x`.

  Returns:
    The result of interpolating along the spline defined by `values`, and
    `tangents`, using `x` as the query values. Will be the same length and type
    as `x`.
  """
  # if x.dtype == 'float64' or torch.as_tensor(x).dtype == torch.float64:
  #   float_dtype = torch.float64
  # else:
  #   float_dtype = torch.float32
  # x = torch.as_tensor(x, dtype=float_dtype)
  # values = torch.as_tensor(values, dtype=float_dtype)
  # tangents = torch.as_tensor(tangents, dtype=float_dtype)
  assert torch.is_tensor(x)
  assert torch.is_tensor(values)
  assert torch.is_tensor(tangents)
  float_dtype = x.dtype
  assert values.dtype == float_dtype
  assert tangents.dtype == float_dtype
  assert len(values.shape) == 1
  assert len(tangents.shape) == 1
  assert values.shape[0] == tangents.shape[0]

  x_lo = torch.floor(torch.clamp(x, torch.as_tensor(0),
                                 values.shape[0] - 2)).type(torch.int64)
  x_hi = x_lo + 1

  # Compute the relative distance between each `x` and the knot below it.
  t = x - x_lo.type(float_dtype)

  # Compute the cubic hermite expansion of `t`.
  t_sq = t**2
  t_cu = t * t_sq
  h01 = -2. * t_cu + 3. * t_sq
  h00 = 1. - h01
  h11 = t_cu - t_sq
  h10 = h11 - t_sq + t

  # Linearly extrapolate above and below the extents of the spline for all
  # values.
  value_before = tangents[0] * t + values[0]
  value_after = tangents[-1] * (t - 1.) + values[-1]

  # Cubically interpolate between the knots below and above each query point.
  neighbor_values_lo = values[x_lo]
  neighbor_values_hi = values[x_hi]
  neighbor_tangents_lo = tangents[x_lo]
  neighbor_tangents_hi = tangents[x_hi]
  value_mid = (
      neighbor_values_lo * h00 + neighbor_values_hi * h01 +
      neighbor_tangents_lo * h10 + neighbor_tangents_hi * h11)

  # Return the interpolated or extrapolated values for each query point,
  # depending on whether or not the query lies within the span of the spline.
  return torch.where(t < 0., value_before,
                     torch.where(t > 1., value_after, value_mid))


def log_safe(x):
  """The same as torch.log(x), but clamps the input to prevent NaNs."""
  #x = torch.as_tensor(x)
  return torch.log(torch.min(x, torch.tensor(33e37).to(x)))


def log1p_safe(x):
  """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
  #x = torch.as_tensor(x)
  return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))


def exp_safe(x):
  """The same as torch.exp(x), but clamps the input to prevent NaNs."""
  #x = torch.as_tensor(x)
  return torch.exp(torch.min(x, torch.tensor(87.5).to(x)))


def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  #x = torch.as_tensor(x)
  return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))


def inv_softplus(y):
  """The inverse of tf.nn.softplus()."""
  #y = torch.as_tensor(y)
  return torch.where(y > 87.5, y, torch.log(torch.expm1(y)))


def logit(y):
  """The inverse of tf.nn.sigmoid()."""
  #y = torch.as_tensor(y)
  return -torch.log(1. / y - 1.)


def affine_sigmoid(logits, lo=0, hi=1):
  """Maps reals to (lo, hi), where 0 maps to (lo+hi)/2."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  #logits = torch.as_tensor(logits)
  #lo = torch.as_tensor(lo)
  #hi = torch.as_tensor(hi)
  alpha = torch.sigmoid(logits) * (hi - lo) + lo
  return alpha


def inv_affine_sigmoid(probs, lo=0, hi=1):
  """The inverse of affine_sigmoid(., lo, hi)."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  #probs = torch.as_tensor(probs)
  #lo = torch.as_tensor(lo)
  #hi = torch.as_tensor(hi)
  logits = logit((probs - lo) / (hi - lo))
  return logits


def affine_softplus(x, lo=0, ref=1):
  """Maps real numbers to (lo, infinity), where 0 maps to ref."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  #x = torch.as_tensor(x)
  #lo = torch.as_tensor(lo)
  #ref = torch.as_tensor(ref)
  shift = inv_softplus(torch.tensor(1.))
  y = (ref - lo) * torch.nn.Softplus()(x + shift) + lo
  return y


def inv_affine_softplus(y, lo=0, ref=1):
  """The inverse of affine_softplus(., lo, ref)."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  #y = torch.as_tensor(y)
  #lo = torch.as_tensor(lo)
  #ref = torch.as_tensor(ref)
  shift = inv_softplus(torch.tensor(1.))
  x = inv_softplus((y - lo) / (ref - lo)) - shift
  return x




def lossfun(x, alpha, scale, approximate=False, epsilon=1e-6):
  r"""Implements the general form of the loss.

  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.

  Args:
    x: The residual for which the loss is being computed. x can have any shape,
      and alpha and scale will be broadcasted to match x's shape if necessary.
      Must be a tensor of floats.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Must be a tensor of floats with the
      same precision as `x`. Varying alpha allows
      for smooth interpolation between a number of discrete robust losses:
      alpha=-Infinity: Welsch/Leclerc Loss.
      alpha=-2: Geman-McClure loss.
      alpha=0: Cauchy/Lortentzian loss.
      alpha=1: Charbonnier/pseudo-Huber loss.
      alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha. Must be a tensor of single-precision
      floats.
    approximate: a bool, where if True, this function returns an approximate and
      faster form of the loss, as described in the appendix of the paper. This
      approximation holds well everywhere except as x and alpha approach zero.
    epsilon: A float that determines how inaccurate the "approximate" version of
      the loss will be. Larger values are less accurate but more numerically
      stable. Must be great than single-precision machine epsilon.

  Returns:
    The losses for each element of x, in the same shape and precision as x.
  """
#   assert torch.is_tensor(x)
#   assert torch.is_tensor(scale)
#   assert torch.is_tensor(alpha)
  assert alpha.dtype == x.dtype
  assert scale.dtype == x.dtype
  assert (scale > 0).all()
  if approximate:
    # `epsilon` must be greater than single-precision machine epsilon.
    assert epsilon > np.finfo(np.float32).eps
    # Compute an approximate form of the loss which is faster, but innacurate
    # when x and alpha are near zero.
    b = torch.abs(alpha - 2) + epsilon
    d = torch.where(alpha >= 0, alpha + epsilon, alpha - epsilon)
    loss = (b / d) * (torch.pow((x / scale)**2 / b + 1., 0.5 * d) - 1.)
  else:
    # Compute the exact loss.

    # This will be used repeatedly.
    squared_scaled_x = (x / scale)**2

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x)
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha),
                             -torch.ones_like(alpha)) * torch.max(
                                 machine_epsilon, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == -float('inf'), loss_neginf,
        torch.where(
            alpha == 0, loss_zero,
            torch.where(
                alpha == 2, loss_two,
                torch.where(alpha == float('inf'), loss_posinf,
                            loss_otherwise))))

  return loss


def partition_spline_curve(alpha):
  """Applies a curve to alpha >= 0 to compress its range before interpolation.

  This is a weird hand-crafted function designed to take in alpha values and
  curve them to occupy a short finite range that works well when using spline
  interpolation to model the partition function Z(alpha). Because Z(alpha)
  is only varied in [0, 4] and is especially interesting around alpha=2, this
  curve is roughly linear in [0, 4] with a slope of ~1 at alpha=0 and alpha=4
  but a slope of ~10 at alpha=2. When alpha > 4 the curve becomes logarithmic.
  Some (input, output) pairs for this function are:
    [(0, 0), (1, ~1.2), (2, 4), (3, ~6.8), (4, 8), (8, ~8.8), (400000, ~12)]
  This function is continuously differentiable.

  Args:
    alpha: A numpy array or tensor (float32 or float64) with values >= 0.

  Returns:
    An array/tensor of curved values >= 0 with the same type as `alpha`, to be
    used as input x-coordinates for spline interpolation.
  """
  alpha = torch.as_tensor(alpha)
  x = torch.where(alpha < 4, (2.25 * alpha - 4.5) /
                  (torch.abs(alpha - 2) + 0.25) + alpha + 2,
                  5. / 18. * log_safe(4 * alpha - 15) + 8)
  return x


class Distribution():
  # This is only a class so that we can pre-load the partition function spline.

  def __init__(self):
    # Load the values, tangents, and x-coordinate scaling of a spline that
    # approximates the partition function. This was produced by running
    # the script in fit_partition_spline.py
    spline_file = (os.path.join(os.path.dirname(__file__), 'ressources/partition_spline_for_robust_loss.npz'))
    with np.load(spline_file, allow_pickle=False) as f:
        self._spline_x_scale = torch.tensor(f['x_scale'])
        self._spline_values = torch.tensor(f['values'])
        self._spline_tangents = torch.tensor(f['tangents'])

  def log_base_partition_function(self, alpha):
    r"""Approximate the distribution's log-partition function with a 1D spline.

    Because the partition function (Z(\alpha) in the paper) of the distribution
    is difficult to model analytically, we approximate it with a (transformed)
    cubic hermite spline: Each alpha is pushed through a nonlinearity before
    being used to interpolate into a spline, which allows us to use a relatively
    small spline to accurately model the log partition function over the range
    of all non-negative input values.

    Args:
      alpha: A tensor or scalar of single or double precision floats containing
        the set of alphas for which we would like an approximate log partition
        function. Must be non-negative, as the partition function is undefined
        when alpha < 0.

    Returns:
      An approximation of log(Z(alpha)) accurate to within 1e-6
    """
    alpha = torch.as_tensor(alpha)
    assert (alpha >= 0).all()
    # Transform `alpha` to the form expected by the spline.
    x = partition_spline_curve(alpha)
    # Interpolate into the spline.
    return interpolate1d(x * self._spline_x_scale.to(x),
                                      self._spline_values.to(x),
                                      self._spline_tangents.to(x))

  def nllfun(self, x, alpha, scale):
    r"""Implements the negative log-likelihood (NLL).

    Specifically, we implement -log(p(x | 0, \alpha, c) of Equation 16 in the
    paper as nllfun(x, alpha, shape).

    Args:
      x: The residual for which the NLL is being computed. x can have any shape,
        and alpha and scale will be broadcasted to match x's shape if necessary.
        Must be a tensor or numpy array of floats.
      alpha: The shape parameter of the NLL (\alpha in the paper), where more
        negative values cause outliers to "cost" more and inliers to "cost"
        less. Alpha can be any non-negative value, but the gradient of the NLL
        with respect to alpha has singularities at 0 and 2 so you may want to
        limit usage to (0, 2) during gradient descent. Must be a tensor or numpy
        array of floats. Varying alpha in that range allows for smooth
        interpolation between a Cauchy distribution (alpha = 0) and a Normal
        distribution (alpha = 2) similar to a Student's T distribution.
      scale: The scale parameter of the loss. When |x| < scale, the NLL is like
        that of a (possibly unnormalized) normal distribution, and when |x| >
        scale the NLL takes on a different shape according to alpha. Must be a
        tensor or numpy array of floats.

    Returns:
      The NLLs for each element of x, in the same shape and precision as x.
    """
    # `scale` and `alpha` must have the same type as `x`.
    #try:
    assert (alpha >= 0).all()
    assert (scale >= 0).all()
    #except:
      #print(alpha)
      #print(scale)
    float_dtype = x.dtype
    assert alpha.dtype == float_dtype
    assert scale.dtype == float_dtype

    loss = lossfun(x, alpha, scale, approximate=False)
    log_partition = torch.log(scale) + self.log_base_partition_function(alpha)
    nll = loss + log_partition
    return nll

  def draw_samples(self, alpha, scale):
    r"""Draw samples from the robust distribution.

    This function implements Algorithm 1 the paper. This code is written to
    allow
    for sampling from a set of different distributions, each parametrized by its
    own alpha and scale values, as opposed to the more standard approach of
    drawing N samples from the same distribution. This is done by repeatedly
    performing N instances of rejection sampling for each of the N distributions
    until at least one proposal for each of the N distributions has been
    accepted.
    All samples are drawn with a zero mean, to use a non-zero mean just add each
    mean to each sample.

    Args:
      alpha: A tensor/scalar or numpy array/scalar of floats where each element
        is the shape parameter of that element's distribution.
      scale: A tensor/scalar or numpy array/scalar of floats where each element
        is the scale parameter of that element's distribution. Must be the same
        shape as `alpha`.

    Returns:
      A tensor with the same shape and precision as `alpha` and `scale` where
      each element is a sample drawn from the distribution specified for that
      element by `alpha` and `scale`.
    """

    assert (alpha >= 0).all()
    assert (scale >= 0).all()
    float_dtype = alpha.dtype
    assert scale.dtype == float_dtype

    cauchy = torch.distributions.cauchy.Cauchy(0., np.sqrt(2.))
    uniform = torch.distributions.uniform.Uniform(0, 1)
    samples = torch.zeros_like(alpha)
    accepted = torch.zeros(alpha.shape).type(torch.bool)
    while not accepted.type(torch.uint8).all():
      # Draw N samples from a Cauchy, our proposal distribution.
      cauchy_sample = torch.reshape(
          cauchy.sample((np.prod(alpha.shape),)), alpha.shape)
      cauchy_sample = cauchy_sample.type(alpha.dtype)

      # Compute the likelihood of each sample under its target distribution.
      nll = self.nllfun(cauchy_sample,
                        torch.as_tensor(alpha).to(cauchy_sample),
                        torch.tensor(1).to(cauchy_sample))

      # Bound the NLL. We don't use the approximate loss as it may cause
      # unpredictable behavior in the context of sampling.
      nll_bound = lossfun(
          cauchy_sample,
          torch.tensor(0., dtype=cauchy_sample.dtype),
          torch.tensor(1., dtype=cauchy_sample.dtype),
          approximate=False) + self.log_base_partition_function(alpha)

      # Draw N samples from a uniform distribution, and use each uniform sample
      # to decide whether or not to accept each proposal sample.
      uniform_sample = torch.reshape(
          uniform.sample((np.prod(alpha.shape),)), alpha.shape)
      uniform_sample = uniform_sample.type(alpha.dtype)
      accept = uniform_sample <= torch.exp(nll_bound - nll)

      # If a sample is accepted, replace its element in `samples` with the
      # proposal sample, and set its bit in `accepted` to True.
      samples = torch.where(accept, cauchy_sample, samples)
      accepted = accepted | accept

    # Because our distribution is a location-scale family, we sample from
    # p(x | 0, \alpha, 1) and then scale each sample by `scale`.
    samples *= scale
    return samples

class AdaptiveLossFunction(torch.nn.Module):
  """The adaptive loss function on a matrix.

  This class behaves differently from general.lossfun() and
  distribution.nllfun(), which are "stateless", allow the caller to specify the
  shape and scale of the loss, and allow for arbitrary sized inputs. This
  class only allows for rank-2 inputs for the residual `x`, and expects that
  `x` is of the form [batch_index, dimension_index]. This class then
  constructs free parameters (torch Parameters) that define the alpha and scale
  parameters for each dimension of `x`, such that all alphas are in
  (`alpha_lo`, `alpha_hi`) and all scales are in (`scale_lo`, Infinity).
  The assumption is that `x` is, say, a matrix where x[i,j] corresponds to a
  pixel at location j for image i, with the idea being that all pixels at
  location j should be modeled with the same shape and scale parameters across
  all images in the batch. If the user wants to fix alpha or scale to be a
  constant,
  this can be done by setting alpha_lo=alpha_hi or scale_lo=scale_init
  respectively.
  """

  def __init__(self,
               num_dims: int,
               dtype: torch.dtype = torch.float32,
               alpha_lo: torch.Tensor = 0.001,
               alpha_hi: torch.Tensor = 1.999,
               alpha_init: Optional[torch.Tensor] = None,
               scale_lo: torch.Tensor = 1e-5,
               scale_init: torch.Tensor = 1.0):
    """Sets up the loss function.

    Args:
      num_dims: The number of dimensions of the input to come.
      float_dtype: The floating point precision of the inputs to come.
      device: The device to run on (cpu, cuda, etc).
      alpha_lo: The lowest possible value for loss's alpha parameters, must be
        >= 0 and a scalar. Should probably be in (0, 2).
      alpha_hi: The highest possible value for loss's alpha parameters, must be
        >= alpha_lo and a scalar. Should probably be in (0, 2).
      alpha_init: The value that the loss's alpha parameters will be initialized
        to, must be in (`alpha_lo`, `alpha_hi`), unless `alpha_lo` == `alpha_hi`
        in which case this will be ignored. Defaults to (`alpha_lo` +
        `alpha_hi`) / 2
      scale_lo: The lowest possible value for the loss's scale parameters. Must
        be > 0 and a scalar. This value may have more of an effect than you
        think, as the loss is unbounded as scale approaches zero (say, at a
        delta function).
      scale_init: The initial value used for the loss's scale parameters. This
        also defines the zero-point of the latent representation of scales, so
        SGD may cause optimization to gravitate towards producing scales near
        this value.
    """
    super(AdaptiveLossFunction, self).__init__()

    self.num_dims = num_dims
    self.alpha_lo = torch.as_tensor(alpha_lo)
    self.alpha_hi = torch.as_tensor(alpha_hi)
    self.scale_lo = torch.as_tensor(scale_lo)
    self.scale_init = torch.as_tensor(scale_init)

    self.distribution = Distribution()

    if alpha_lo == alpha_hi:
      # If the range of alphas is a single item, then we just fix `alpha` to be
      # a constant.
      self.fixed_alpha = alpha_lo.unsqueeze(0).unsqueeze(0).repeat(1, self.num_dims)
      # Assuming alpha_lo is already a torch.Tensor

      self.alpha = lambda: self.fixed_alpha
    else:
      # Otherwise we construct a "latent" alpha variable and define `alpha`
      # As an affine function of a sigmoid on that latent variable, initialized
      # such that `alpha` starts off as `alpha_init`.
      if alpha_init is None:
        alpha_init = torch.as_tensor((alpha_lo + alpha_hi) / 2.)
      latent_alpha_init = inv_affine_sigmoid(alpha_init, lo=alpha_lo, hi=alpha_hi)
      
      latent_alpha_init_1 = latent_alpha_init.clone().unsqueeze(0).unsqueeze(0).repeat(1, self.num_dims)
      self.register_parameter('latent_alpha', torch.nn.Parameter(latent_alpha_init_1,requires_grad=True))
      

      self.alpha = lambda: affine_sigmoid(self.latent_alpha, lo=alpha_lo, hi=alpha_hi)

    if scale_lo == scale_init:
      # If the difference between the minimum and initial scale is zero, then
      # we just fix `scale` to be a constant.
      self.fixed_scale = scale_init.unsqueeze(0).unsqueeze(0).repeat(1, self.num_dims)
      # self.fixed_scale = torch.tensor(
      #     scale_init, dtype=self.float_dtype,
      #     device=self.device)[np.newaxis, np.newaxis].repeat(1, self.num_dims)
      self.scale = lambda: self.fixed_scale
    else:
      # Otherwise we construct a "latent" scale variable and define `scale`
      # As an affine function of a softplus on that latent variable.

      self.register_parameter('latent_scale',torch.nn.Parameter(torch.zeros((1, self.num_dims)),requires_grad=True))
      self.scale = lambda: affine_softplus(self.latent_scale, lo=scale_lo, ref=scale_init)


  def lossfun(self, x, **kwargs):
    """Computes the loss on a matrix.

    Args:
      x: The residual for which the loss is being computed. Must be a rank-2
        tensor, where the innermost dimension is the batch index, and the
        outermost dimension must be equal to self.num_dims. Must be a tensor or
        numpy array of type self.float_dtype.
      **kwargs: Arguments to be passed to the underlying distribution.nllfun().

    Returns:
      A tensor of the same type and shape as input `x`, containing the loss at
      each element of `x`. These "losses" are actually negative log-likelihoods
      (as produced by distribution.nllfun()) and so they are not actually
      bounded from below by zero. You'll probably want to minimize their sum or
      mean.
    """
    #x = torch.as_tensor(x)
    assert len(x.shape) == 2
    assert x.shape[1] == self.num_dims
    #assert x.dtype == self.float_dtype
    return self.distribution.nllfun(x, self.alpha(), self.scale(), **kwargs)
  
  def forward(self,input,pred):
    if pred.ndim == 1:
      res = (input-pred)[:,None]
    else:
      res = input - pred
    return torch.mean(self.lossfun(res))