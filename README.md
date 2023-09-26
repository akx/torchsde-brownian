# torchsde_brownian

This library provides the Brownian motion generation code from [torchsde]
as a standalone package `torchsde_brownian`.

It is mainly meant for use in downstream packages such as `k-diffusion` and `diffusers`,
for where it is a drop-in replacement: the dependency just needs to be changed,
and the import for `torchsde` to `torchsde_brownian`.

## Installation

```shell script
pip install torchsde_brownian
```

**Requirements:** Python >=3.8 and PyTorch >=1.6.0.

## Documentation

Available [here](./DOCUMENTATION.md).

## Citation and references

Please find citations and references in [torchsde]'s readme.

[torchsde]: https://github.com/google-research/torchsde
