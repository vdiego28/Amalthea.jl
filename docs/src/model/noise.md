# Noise model

`prop_capillary`, `prop_gnlse` and `prop_fibre` add one-photon-per-mode
quantum noise to the input field when `shotnoise=true` (the default).
The input noise propagates through both the linear and nonlinear operators.
Use `shotnoise=false` to disable it, or `shotnoise=:input` as an explicit alias.

Pass a seeded random number generator for reproducible realizations:

```julia
using Amalthea, Random
output = prop_capillary(100e-6, 0.1, :He, 1;
    λ0=800e-9, τfwhm=10e-15, energy=1e-6,
    trange=400e-15, λlims=(200e-9, 4e-6),
    rng=MersenneTwister(42))
```

Each propagated mode receives independent random phases. Reusing a seed
reproduces the initial noise; use independent seeds for ensemble studies.
Noise at the one-photon level can seed nonlinear growth, so compare ensembles
and check numerical convergence when interpreting noise-sensitive results.

The former `:modified` model, which injected a constant noise field into the
nonlinear operator while excluding it from dispersion, was withdrawn for
incorrect physics in Luna #461 and is rejected in Amalthea 1.1.0. Low-level
`noise_field` source arguments are also rejected. Recompute results that
relied on that model. The standalone Python beta does not support quantum noise.
