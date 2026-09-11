# Julia-free Python

`amalthea_native` provides GNLSE and capillary propagation with Python setup,
owned NumPy results and a Rust adaptive solver. It supports carrier and envelope
fields, multimode propagation, polarization, gas mixtures, plasma, molecular
Raman, changing profiles and custom Python models within the combinations
supported by the Julia implementation.

The distribution name is provisionally `amalthea-native`. This development
package is undergoing installed-artifact validation; see the
[current acceptance status](https://github.com/vdiego28/Amalthea.jl/blob/feat/julia-free-python/docs/dev/BACKLOG.md).
Its publication and platform availability are separate from the existing
Julia-based Python wrapper.

## Install a wheel

Use CPython 3.11–3.14 and a wheel matching your interpreter, operating system and
CPU architecture. Install the downloaded file in an environment of your choice:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: /path/to/amalthea_native-VERSION-TAGS.whl
```

Replace the path with the actual artifact. Installation resolves NumPy, SciPy,
CoolProp 7.2.0 and mpmath. The installed wheel requires no Julia, Rust compiler,
system FFTW or libcubature. Simulations are CPU-only. Dependencies must be
installed before going offline; no simulation downloads a runtime.

For HDF5 output, install the same wheel with its optional extra:

```sh
python -m pip install --only-binary=:all: '/path/to/amalthea_native-VERSION-TAGS.whl[hdf5]'
```

## Run both propagation APIs

Arguments preserve Julia's order and physical units. Use metres for lengths and
wavelengths, seconds for times, joules for energy, watts for peak power, kelvin
for temperature and **bar for gas pressure**. Angular frequency is in rad/s.
GNLSE `gamma` is in W⁻¹m⁻¹; `betas` starts with orders zero and one, with the
order-n coefficient in sⁿ/m. GNLSE `loss` is power attenuation in dB/m;
capillary `loss` is a boolean selecting the mode's attenuation.

```python
import numpy as np
from amalthea_native import prop_gnlse, prop_capillary

common = dict(lambda0=800e-9, lambda_lims=(600e-9, 1200e-9),
              trange=100e-15, tau_fwhm=20e-15, saveN=5)

fiber = prop_gnlse(0.01, 0.003, [0, 0, -20e-27], power=2000,
                   ramanmodel="sdo", **common)

# Radius, length, gas, pressure. Carrier propagation is the default.
capillary = prop_capillary(125e-6, 1e-5, "Ar", 2.0, energy=100e-6,
                           plasma="ADK", raman=False, **common)

for result in (fiber, capillary):
    print(result.field.shape, result.z)
    print(result.metadata["backend"], result.metadata["backend_reason"])
    print(result.metadata["accepted_steps"], result.metadata["rejected_steps"])
    assert np.all(np.isfinite(result.field))
```

Set `envelope=True` for envelope capillary propagation, and `plasma=False` to
disable carrier plasma. `raman=None` infers molecular Raman from the gas;
`raman=False` disables it. GNLSE supports `ramanmodel="sdo"` and `"SiO2"`.
Its instantaneous term retains the Julia `(1-fr)` factor even when Raman is
disabled; use `fr=0` for an entirely instantaneous response.

ASCII aliases include `lambda0`, `lambda_lims`, `tau_fwhm`, `delta_t`, `phi`
and `polarization`; the corresponding Greek aliases and `polarisation` are
accepted. Supplying two aliases for the same option raises an error. `dt` is an
alias for **temporal grid spacing**. Propagation step bounds are `init_dz`,
`min_dz` and `max_dz`. `rtol` and `atol` control local nonlinear error;
`saveN` controls output sampling. Default fifth-order stepping uses fifth-order
dense output; `locextrap=False` selects the compatible fourth-order scheme.

## Select evaluation and inspect results

| `backend` | Behavior |
|---|---|
| `"auto"` | Uses resident Rust evaluation when eligible; otherwise selects Python evaluation. Modal quadrature remains SciPy, with eligible temporal point arrays evaluated in Rust. |
| `"native"` | Requires complete resident Rust evaluation. Raises for modal quadrature, changing profiles, custom responses and other ineligible configurations. |
| `"python"` | Uses Python evaluation and SciPy modal quadrature with Rust stepping. |

Every path is Julia-free. Metadata records the selection, its reason, solver
counts and FFT implementation. Modal results also report `point_evaluator` and
`point_fft`; native point evaluation does not make the complete modal solve
resident in Rust.

`result.field` (also `Eomega` or `Eω`) has `(frequency, saved_position)` axes.
Modal results use `(frequency, mode, saved_position)`, including a collection
containing just one mode. `result.z` contains saved positions in metres.
`result.grid.t` and `result.grid.omega` describe the time/frequency axes.
Spectra retain the original unshifted FFT order. Use `temporal_field()` for the
matching inverse transform; carrier reconstruction is real.

```python
from amalthea_native.pulses import energy_t

final_time = capillary.temporal_field()[..., -1]
final_energy = energy_t(capillary.grid, final_time)
print("Final pulse energy (J):", final_energy)
assert np.isfinite(final_energy) and final_energy >= 0
```

`energy_t` applies the project's analytic-signal intensity and integration
conventions to one time-domain mode. For modal output, apply it separately to
each column and sum the modal energies. Callback fields below use physical
electric-field normalization; they are distinct from these modal amplitudes.

## Pulses and mode collections

Use `pulseshape="sech"` for a sech pulse, or `pulses=[GaussPulse(...),
SechPulse(...), DataPulse(...)]` for a coherent collection. Each pulse has its
own normalization and phase; their sum is not renormalized. `DataPulse` accepts
an independently sampled angular-frequency spectrum. A grid-matched array uses
`pulse=array, pulse_domain="time"` or `"frequency"`; construct the same
`EnvGrid`/`RealGrid` first to determine its shape. Supplied amplitude is retained
unless energy or peak power is requested. Modal input arrays put time/frequency
first and modes second.

```python
modal = prop_capillary(125e-6, 1e-5, "Ar", 2.0, modes=2,
                       envelope=True, energy=100e-6, plasma=False,
                       raman=False, init_dz=5e-6, max_dz=5e-6, **common)
assert modal.field.shape[1:] == (2, common["saveN"])
print(modal.metadata["point_evaluator"])
```

`modes=2` selects HE11 and HE12. A sequence such as `["HE21", "TE01", "TM01"]`
selects full spatial fields. Pulse `mode` and `polarisation` select their input
mode and polarization; circular/elliptical HE inputs use two orientations and
split the energy. `radial_integral_rtol`, `modal_atol` and `modal_maxevals`
control the global real/imaginary L2 quadrature criterion. Budget exhaustion
raises an error. Tighten quadrature and propagation controls independently when
checking convergence.

## Custom modes, profiles and responses

Subclass `Mode` and implement `neff(omega, *, z)`,
`field(coordinates, *, z)` and `dimlimits(*, z)`. `dimlimits` returns
`("polar"|"cartesian", (lower1, lower2), (upper1, upper2))`.
Spatial fields are real arrays with x/y first: `(2, points)` for batched
coordinates. The base supplies power normalization and dispersion; override
`N(*, z)` or `dispersion(order, omega, *, z)` when an analytic result is
available. Pass instances as `modes=[custom_mode, ...]`.

Radius and pressure may be functions of `z`. Pressure also accepts `(p0, p1)`
or `(positions, pressures)`. Functions are evaluated at the solver's requested
positions, including retries and output stages. They must cover the final
accepted step, which can extend beyond the last requested output. No sampled
replacement is made for arbitrary functions. Density, area, dispersion and
density-dependent responses refresh at the required position.

Variable linear propagation preserves Julia's endpoint exponential. Its
complete error may be first order even with fifth-order nonlinear stepping.
Refine `max_dz` to check convergence, separately from `rtol`.

For mixtures use a gas sequence and matching partial pressures, such as
`gas=("Ar", "Ne"), pressure=(2.0, 1.0)`. Each pressure may itself be a
profile. `species_options` supplies one response-option mapping per species.

Both APIs accept `responses=callback` or an ordered list. A callback receives
an owned complete oversampled time/component field and `ResponseContext`.
Return polarization of exactly the same shape: `(ntime, 1)` for scalar fields
or `(ntime, 2)` for vector modes. Carrier polarization must be real. Context
includes `z`, `t`, component names, gases, densities and modal coordinates.
Calls execute serially, invalid/nonfinite results raise, and original exceptions
propagate. Custom responses append to enabled built-ins and receive no implicit
density scaling.

Here is an equivalent scalar envelope Kerr replacement:

```python
from amalthea_native.materials import EPS0, gamma3

def envelope_kerr(field, context):
    coefficient = 0.75 * EPS0 * sum(
        gamma3(gas) * rho for gas, rho in zip(context.gases, context.densities))
    return coefficient * field * abs(field)**2

options = dict(common, envelope=True, energy=10e-6, plasma=False,
               raman=False, init_dz=5e-6, max_dz=5e-6)
builtin = prop_capillary(125e-6, 1e-5, "Ar", 2.0, **options)
custom = prop_capillary(125e-6, 1e-5, "Ar", 2.0, kerr=False,
                        responses=envelope_kerr, **options)
error = np.linalg.norm(custom.field-builtin.field) / np.linalg.norm(builtin.field)
assert error < 1e-13
print("Custom Kerr relative error:", error)
```

Vector Kerr requires coupling between components. GNLSE uses a different
polarization coefficient; use the complete examples and package API reference
when replacing those models.

## Save and read results

NPZ output is available in the base installation. It contains numeric arrays
and versioned JSON descriptions, with no pickle requirement:

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "capillary.npz"
    modal.save_npz(path)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["Eomega"], modal.field)
        print(json.loads(str(saved["metadata"])))
```

With the `hdf5` extra installed, HDF5 allows reading a field slice without
loading the complete trajectory:

```python
import h5py

with TemporaryDirectory() as directory:
    path = Path(directory) / "capillary.h5"
    modal.save_hdf5(path)
    with h5py.File(path, "r") as saved:
        np.testing.assert_array_equal(saved["Eomega"][..., -1], modal.field[..., -1])
        print(json.loads(saved["grid"].asstr()[()])["field_axes"])
```

Both methods overwrite the destination and preserve frequency/mode/save order.
They save completed results and callback descriptions, not live Python objects
or a serialized solver restart.

## Examples and supported combinations

The [complete example scripts](https://github.com/vdiego28/Amalthea.jl/tree/feat/julia-free-python/python-native/examples)
cover both APIs, custom modes/responses, multimode fields, supplied pulses,
profiles, mixtures, ADK/PPT, Raman, output processing and solver diagnostics.
Copy an example outside the checkout and run it with the installed interpreter.
PPT tables are generated locally with parameter-keyed caches; their first
construction can take longer than subsequent reuse.

The [capability matrix](https://github.com/vdiego28/Amalthea.jl/blob/feat/julia-free-python/docs/dev/native-port/PYTHON_SUPPORT_MATRIX.md)
records execution paths and Julia model restrictions. Envelope plasma, vector
Raman and carrier vector Kerr without THG are excluded. Quantum noise defaults
off and explicit requests fail. Python GPU execution, ensembles and separate
free-space/step-index interfaces are outside this release.
