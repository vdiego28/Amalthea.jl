# Amalthea native Python — internal development package

This package provides Julia-compatible time grids, private portable FFT bindings,
and `solve_precon` for spectral equations with a constant diagonal linear
operator and custom Python nonlinear RHS. Rust owns adaptive stepping and dense
output. The high-level GNLSE/capillary APIs are not implemented yet. This is **not** the
public Linux preview; that preview requires complete GNLSE and capillary support.

Build an internal CPU wheel from this directory with
`RUSTFLAGS="" maturin build --release`. Install the resulting wheel into an
isolated environment and run `python -m pytest tests`. Runtime imports require
NumPy and the included Rust extension, with no Julia or system FFTW dependency.
These local wheels are not yet validated manylinux release artifacts.

```python
from amalthea_native import EnvGrid, RealGrid

grid = EnvGrid(1.0, 800e-9, (400e-9, 1600e-9), 300e-15)
print(grid.t.shape, grid.omega.shape)
```

An independently checkable nonlinear example:

```python
import numpy as np
from amalthea_native import solve_precon

# y' = y², y(0) = 1; exact solution y(z) = 1/(1-z).
result = solve_precon(lambda z, field: field**2, [0j], [1+0j], 0.4,
                     dt=0.4, rtol=1e-10, saveN=41)
assert np.max(np.abs(result.field[0] - 1/(1-result.z))) < 1e-6
print(result.metadata["accepted_steps"], result.metadata["rejected_steps"])
```

`rhs(z, field)` and optional `step_filter(z, field)` receive owned NumPy arrays
and must return finite arrays of the same shape. Results add saved position as
the final axis. Filters run after accepted-step sampling, including the final
accepted endpoint beyond the requested output range, matching Julia's solve
loop. `locextrap=False` selects fourth-order stepping and corrected quartic
dense output; the default uses fifth-order stepping and quintic dense output.
`linop` must have exactly the input field's shape. Variable linear operators,
native nonlinear dispatch, and high-level pulse/material setup remain future
implementation units. The callback adapter currently copies field arrays at
each RHS evaluation; it makes no resident-native performance claim.
`examples/solver_analytic.py` runs a complete nonlinear solve with a nonzero
linear operator, deliberately rejected trials, and an analytic accuracy check.

The development-only `tools/export_grid_oracle.jl` script writes independent
Julia grid fixtures. Run it with the repository's Julia project, then set
`AMALTHEA_GRID_ORACLE` to the generated directory when running the Python tests.
The ordinary installed package never imports or provisions Julia.
`tools/export_solver_oracle.jl` similarly writes independently prepared solver
trajectories; set `AMALTHEA_SOLVER_ORACLE` to that directory for driver parity
tests. Cargo configuration in this distribution forces CPU-only engine builds.
With the engine dependency, maturin nests Rust sources and tests under
`python-native/` inside an extracted sdist. Run its tests with
`python -m pytest python-native/tests` from the extracted archive root.
