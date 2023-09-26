# Documentation

## Brownian motion

`BrownianInterval` can also be used as a standalone object, if you just want to be able to sample Brownian motion for any other reason.

The time and memory efficient sampling provided by the Brownian Interval was introduced in [\[1\]](https://arxiv.org/abs/2105.13493).

### Examples
**Quick example**

```python
from torchsde_brownian import BrownianInterval
bm = BrownianInterval(t0=0., t1=1., size=(4, 1))
dW = bm(0.2, 0.3)
```
Produces a tensor `dW` of shape `(4, 1)` corresponding to the increment `W(0.3) - W(0.2)` for a Brownian motion `W` defined over `[0, 1]`, taking values in `(4, 1)`-dimensional space.

(Mathematically: `W \in C([0, 1]; R^(4 x 1))`.)

**Example with `sdeint`**

### Arguments

- `t0` (float or Tensor): The initial time for the Brownian motion.
- `t1` (float or Tensor): The terminal time for the Brownian motion.
- `size` (tuple of int): The shape of each Brownian sample. If zero dimensional represents a scalar Brownian motion. If one dimensional represents a batch of scalar Brownian motions. If >two dimensional the last dimension represents the size of a a multidimensional Brownian motion, and all previous dimensions represent batch dimensions.
- `dtype` (torch.dtype): The dtype of each Brownian sample. Defaults to the PyTorch default.
- `device` (str or torch.device): The device of each Brownian sample. Defaults to the CPU.
- `entropy` (int): Global seed, defaults to `None` for random entropy.
- `levy_area_approximation` (str): Whether to also approximate Levy area. Defaults to `"none"`. Valid options are `"none"`, `"space-time"`, `"davie"` or `"foster"`, corresponding to different approximation types, see [below](#levy-area-approximation). This is needed for some higher-order SDE solvers.
- `dt` (float or Tensor): The expected average step size of the SDE solver. Set it if you know it (e.g. when using a fixed-step solver); else it will be estimated from the first few queries. This is used to set up the data structure such that it is efficient to query at these intervals.
- `tol` (float or Tensor): What tolerance to resolve the Brownian motion to. Must be non-negative. Defaults to zero, i.e. floating point resolution. Usually worth setting in conjunction with `halfway_tree`, below.
- `pool_size` (int): Size of the pooled entropy. If you care about
    statistical randomness then increasing this will help (but will
    slow things down).
- `cache_size` (int): How big a cache of recent calculations to use. (As new calculations depend on old calculations, this speeds things up dramatically, rather than recomputing things.) Set this to `None` to use an infinite cache, which will be fast but memory inefficient.
- `halfway_tree` (bool): Whether the dependency tree (the internal data structure) should be the dyadic tree. Defaults to `False`. Normally, the sample path is determined by both `entropy`, _and_ the locations and order of the query points. Setting this to `True` will make it deterministic with respect to just `entropy`; however this is much slower.
- `W` (Tensor): The increment of the Brownian motion over the interval
    `[t0, t1]`. Will be generated randomly if not provided.
- `H` (Tensor): The space-time Levy area of the Brownian motion over the
    interval `[t0, t1]`. Will be generated randomly if not provided.

### Important special cases

**Speed over memory**

If speed is important, and you're happy to use extra memory, then use
```python
BrownianInterval(..., cache_size=None)
```

**Fixed randomness**

If you want to use the same random seed to deterministically create the same Brownian motion, then use
```python
BrownianInterval(..., entropy=<integer>, tol=1e-5, halfway_tree=True)
```
If you're using a fixed SDE solver (or more precisely, if the locations and order of the queries to the Brownian interval are fixed), then just
```python
BrownianInterval(..., entropy=<integer>)
```
will suffice, and will be faster.

### BrownianPath and BrownianTree

`torchsde.BrownianPath` and `torchsde.BrownianTree` are the legacy ways of creating Brownian motions, corresponding to each of the two important special cases above (respectively).

These are still supported, but we encourage using the more flexible `BrownianInterval` for new projects.

### Levy area approximation
The `levy_area_approximation` argument may be either `"none"`, `"space-time"`, `"davie"` or `"foster"`. Levy area approximations are used in certain higher-order SDE solvers, and so this must be set to the appropriate value if using these higher-order solvers.

In particular, space-time Levy area is used in the stochastic Runge--Kutta solver.

# References

\[1\] Patrick Kidger, James Foster, Xuechen Li, Terry Lyons. "Efficient and Accurate Gradients for Neural SDEs". 2021. [[arXiv]](https://arxiv.org/abs/2105.13493)

