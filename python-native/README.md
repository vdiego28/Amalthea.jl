# Amalthea native Python — internal development package

This package provides Julia-compatible time grids, private portable FFT bindings,
and `solve_precon` for spectral equations with a diagonal linear
operator and custom Python nonlinear RHS. Rust owns adaptive stepping and dense
output. `prop_gnlse` runs resident Rust evaluation with portable FFTs or Python
evaluation. It supports analytic/data/array pulse collections, custom input
propagators, Taylor dispersion, loss, Kerr, shock, and SDO/SiO₂ Raman. `prop_capillary` adds scalar/modal carrier and envelope
propagation, polarization, gas mixtures, ADK/PPT, molecular Raman, profiles and
custom Python modes/responses. The implemented capability matrix is in
`docs/dev/native-port/PYTHON_SUPPORT_MATRIX.md`; current artifact acceptance and
release status live in `docs/dev/BACKLOG.md`. This development package is not yet
the public preview.

Build an internal CPU wheel from this directory with
`RUSTFLAGS="" maturin build --release`. Install the resulting wheel into an
isolated environment and run `python -m pytest tests`. Runtime imports require
NumPy, SciPy, CoolProp 7.2.0, and the included Rust extension, with no Julia or system FFTW
dependency.
A local maturin build is a development artifact. Release wheels require the
maintained build, binary audit, installed numerical and offline gates; their
platform status is recorded in the backlog. For an installed-wheel walkthrough,
see `docs/src/python_native.md`.

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
`linop` may be an array or a callable `linop(z)` returning exactly the input
field's shape. Calls run serially at the requested stage/output positions;
returned arrays are copied and validated, and original exceptions propagate.
Only the immediately preceding linear position is cached, matching Julia.
The callback adapter currently copies field arrays at
each RHS evaluation; it makes no resident-native performance claim.
`examples/solver_analytic.py` runs a complete nonlinear solve with a nonzero
linear operator, deliberately rejected trials, and an analytic accuracy check.

Variable operators retain Julia's `exp(linop(z2)*(z2-z1))` endpoint convention.
This does not integrate a changing linear coefficient across an interval;
the complete variable-coefficient equation may have first-order error even
with fifth-order nonlinear stepping. Refine `max_dt` for convergence, as shown
in `examples/variable_solver.py`. The capillary profile API below uses this
driver. Its independent development fixture is
`AMALTHEA_VARIABLE_SOLVER_ORACLE`, exported by
`tools/export_variable_solver_oracle.jl`.

The development-only `tools/export_grid_oracle.jl` script writes independent
Julia grid fixtures. Run it with the repository's Julia project, then set
`AMALTHEA_GRID_ORACLE` to the generated directory when running the Python tests.
The ordinary installed package never imports or provisions Julia.
`tools/export_solver_oracle.jl` similarly writes independently prepared solver
trajectories; set `AMALTHEA_SOLVER_ORACLE` to that directory for driver parity
tests. `tools/export_gnlse_oracle.jl` adds optical setup/RHS/trajectory fixtures;
set `AMALTHEA_GNLSE_ORACLE` as well for the full acceptance suite. Cargo
configuration in this distribution forces CPU-only engine builds.
With the engine dependency, maturin nests Rust sources and tests under
`python-native/` inside an extracted sdist. Run its tests with
`python -m pytest python-native/tests` from the extracted archive root.

### Internal GNLSE API

Run `examples/gnlse.py` for a complete nonlinear pulse simulation. Arguments
`gamma`, fibre length, and Taylor beta coefficients retain their Julia order;
coefficients include beta0 and beta1. Required keywords are `lambda0`,
`lambda_lims`, and `trange`, all in SI units. Gaussian pulses need `tau_fwhm`
and either `power` (W) or `energy` (J); `pulseshape="sech"` also accepts `tau_w`.
`phi` contains spectral phase coefficients beginning at order zero. ASCII and
Unicode aliases are accepted; supplying two names for one keyword raises.

`backend="auto"` selects resident Rust GNLSE evaluation, including Raman and
accepted-step filtering, using RustFFT/RealFFT. `backend="native"` forces that
path; `backend="python"` uses NumPy evaluation with Rust stepping. Execution
metadata records the selected backend, FFT implementation, and stepper. Native
GNLSE calls no Python functions during stepping; custom input propagators run
during pulse setup. `shotnoise` defaults false; true raises.
`ramanmodel="sdo"` (default) and `ramanmodel="SiO2"` are supported. SiO₂ uses
the thirteen-component silica response; `tau1`/`tau2` apply only to SDO.
Pulse collections, supplied-frequency spectra, and custom input propagators
are supported as described below. Capillary propagation and custom Python
models are documented in the following sections.

An optional `pulse` is an owned complex array already on the constructed grid,
with `pulse_domain="time"` or `"frequency"`. Its amplitude is retained unless
`energy` or `power` is supplied for normalization. Results contain `field`
(frequency, saved position), `z`, `grid`, parameters, and execution metadata.
Use `temporal_field()` for inverse FFT reconstruction and `save_npz(path)` for
numeric arrays plus JSON metadata that load without pickle. Install the optional
extra with `pip install 'amalthea-native[hdf5]'` to use `save_hdf5(path)`.
Both formats preserve frequency/(mode)/saved-position axes and contain `Eomega`,
`z`, the grid axes/windows/mask, `format_version=1`, and JSON `grid`, `parameters`
and `metadata`. The grid description records the carrier/envelope choice and
axis labels. HDF5 uses lossless gzip arrays and UTF-8 JSON datasets; read a field
slice with `saved['Eomega'][..., -1]` and JSON with
`json.loads(saved['metadata'].asstr()[()])`. Both save methods overwrite existing
files. HDF5 is imported only on request.

Run `examples/output_processing.py` with the extra installed for complete
GNLSE and modal capillary simulations, exact NPZ/HDF5 round trips, temporal
reconstruction, per-mode energies and execution diagnostics. These are saved
results for analysis; they do not serialize live callbacks or solver state.

Solver controls are `init_dz`, `min_dz`, `max_dz`, `rtol`, `atol`, and `locextrap`.
`dt` retains Julia's alias for temporal grid spacing, not propagation step size.

### Multiple and custom input pulses

Use `pulses=GaussPulse(...)` or an ordered list such as
`pulses=[GaussPulse(...), SechPulse(...), DataPulse(...)]`. Each analytic pulse
supplies its own `lambda0`, duration, energy/peak power and `phi`. Pulses combine
coherently after individual normalization; the sum is not renormalized. As in
Julia, top-level simple pulse settings are ignored when `pulses` is supplied.

`DataPulse(omega, spectrum, energy=...)` accepts complex field samples on an
angular-frequency axis in rad/s. Alternatively pass `(omega, intensity, phase)`.
At least four unique positive frequencies are required. Cubic interpolation,
frequency masking, spectral-energy normalization, and the half-window centering
phase follow Julia's DataField. Unlike a grid-matched `pulse_domain="frequency"`
array, this represents an independently sampled input spectrum. Supply ascending
frequencies for meaningful complex-phase unwrapping; rows are sorted before
interpolation. Optional `phi` uses explicit `lambda0` or the source intensity's
mean frequency as its expansion center.

Pass `propagator=callback` to an analytic/data pulse, or wrap it with
`PropagatedPulse(pulse, callback)`. The callback receives `(field, grid)` once,
after input normalization and phase, and may mutate the owned field and return
None or return a complete array. Its grid is an isolated copy. Output shape and
finiteness are checked, and original exceptions propagate. Callbacks execute
serially in input order. Top-level `propagator=` applies to the simple input.
See `examples/pulses.py` for a complete multi-color/data/custom-input simulation.

Pulse objects also accept `RealGrid` through `pulse.spectrum(grid)`. The result
is an owned positive-frequency spectrum; reconstruct the real field with
`np.fft.irfft(spectrum, n=grid.t.size)`. Carrier-resolved Gaussian/sech pulses
use a cosine carrier and normalize energy or peak power after spectral phase,
using analytic-signal intensity. Real time-array inputs must match `grid.t`;
frequency-array inputs must match `grid.omega`. See `examples/real_pulse.py`
for few-cycle input preparation. The constant capillary API below also supports
carrier-resolved Kerr propagation with explicit `plasma=False`.

### Gas material setup

`amalthea_native.materials` provides `density`, inverse `pressure`,
`polarizability`, `refractive_index`, and default `gamma3` for the Julia gas
identifiers. Wavelengths are in metres, pressure in bar, temperature in kelvin,
and density in particles/m³. Numeric arrays broadcast and results own their
memory. CoolProp 7.2.0 and CODATA2014 normalization match PhysData.

```python
from amalthea_native.materials import density, refractive_index
rho = density("Ar", pressure=2.0, temperature=293.15)
n = refractive_index("Ar", [400e-9, 800e-9], pressure=2.0)
```

These functions retain the implemented Julia material conventions, including
the QuanfuHe expression for CH4/N2O/SF6. Air has no default gamma3 source.
The development `tools/export_material_oracle.jl` supplies the fifth fixture
variable, `AMALTHEA_MATERIAL_ORACLE`, for complete installed setup acceptance.

### Capillary mode setup

`MarcatiliMode(radius, gas=None, pressure=0, ...)` supports HE, TE, and TM
modes, `model="full"|"reduced"`, `loss=True|False`, and polarization angle
`phi` (also `ϕ` or `φ`). HE requires `n>=1`; TE/TM require `n=0`. Radial
order `m` starts at one. Gas-free construction uses a vacuum core.

```python
from amalthea_native import MarcatiliMode
import numpy as np
mode = MarcatiliMode(125e-6, "Ar", 2.0)
omega = 2*np.pi*299792458/800e-9
print(mode.beta(omega), mode.alpha(omega), mode.effective_area())
```

Frequency is rad/s; `beta` is rad/m, `alpha` is power attenuation per metre,
and effective area is m². `field((r, theta))` returns the raw transverse
vector with shape `(2, *coordinate_shape)` and x/y components first.
`normalized=True` divides it by `sqrt(mode.N())`. Radius and pressure may
be callables of propagation position `z`; methods evaluate them at the
requested position without sampling a replacement profile.

`core_index` and `cladding_index` callbacks receive `(omega, *, z)` with an
owned frequency array and return a finite scalar or an array of exactly
that shape. A core callback replaces gas setup. Exceptions propagate.
The default silica cladding uses Julia's bundled complex lookup table.

`dispersion(order, omega, z=...)` retains Julia's scaled adaptive finite
difference convention for orders 0–7. Group velocity and same-sample stencil
equivalence are tested at 1e-13. Higher derivatives inherit the accuracy
limits of Julia's float64 finite differences: small index differences are
amplified by cancellation, especially at orders 4–7. Independent 100-digit
analytic refinement tests verify the stencil convergence and bound the
float64 error; they do not imply 1e-13 accuracy for those high derivatives.
Capillary propagation acceptance remains separate from mode setup.

The development `tools/export_mode_oracle.jl` supplies
`AMALTHEA_MODE_ORACLE`, the sixth fixture variable for installed tests.

### Custom modes and modal propagation

Subclass `Mode` and implement `neff(omega, *, z)`,
`field((coordinate1, coordinate2), *, z)` and `dimlimits(*, z)`. The domain is
`("polar"|"cartesian", (lower1, lower2), (upper1, upper2))`. Spatial fields are
real, with x/y on axis zero and batched coordinates returning `(2, points)`.
Effective index may be complex; return a finite scalar or the frequency shape.
The base class supplies `beta`, `alpha`, `dispersion` and numerical power
normalization `N`. Override `N` when an analytic expression is available.
`Exy(coordinates, z=...)` returns the physical field divided by `sqrt(N)`.
See `examples/custom_mode.py` for an equivalent custom Marcatili construction.

The internal spatial integrator supports full polar/Cartesian and reduced
radial geometry, selected x/y components, and complete-array synthesis and
projection. It checks the global real/imaginary L2 error and raises if the
integration budget is exhausted. This requires SciPy 1.15 or later. Pass
constructed modes as `modes=[custom_mode, ...]` to `prop_capillary`; custom
dispersion and geometry are evaluated at the solver's actual positions.
`modal_components="y"` and `modal_full=False` explicitly select scalar radial
geometry. Custom modes otherwise use both components and the full integral.

`modes=2` selects HE11 and HE12. A sequence such as `["HE21", "TE01", "TM01"]`
selects full spatial propagation; mappings (`dict(kind="HE", n=1, m=10)`)
allow larger mode indices. A single ordinary signifier selects mode average.
Pulse objects accept `mode="HE12"` and `polarisation="linear"|"x"|"y"|"circular"`
or ellipticity in `[-1,1]`; `polarization` is an alias. The historical Julia x/y
selector ordering is preserved. Circular/elliptical HE inputs construct two
orientations and split their energy. Supplied modal arrays have time/frequency
first and modes second; normalization uses their total intensity.

Modal results use `(frequency, mode, saved position)` axes. Automatic selection
uses portable Rust point evaluation for constant built-in modes with ordinary
Kerr on either grid and scalar sampled Raman. Plasma, custom models/responses,
profiles and other THG choices retain Python point evaluation. `backend="python"`
selects Python points explicitly. Metadata reports `point_evaluator` and
`point_fft`; spatial integration remains SciPy and stepping remains Rust. Forced
native rejects this path. `radial_integral_rtol=1e-3`, `modal_atol=0`,
`modal_maxevals=100000` and `modal_rule="gk21"` control the global quadrature
criterion. Vector Kerr/plasma, scalar Raman, mixtures, gradients and tapers
are available. Vector Raman and envelope plasma are excluded; carrier vector
Kerr requires `thg=True`, matching Julia's valid combinations. Custom nonlinear
responses use the complete-array callback API described below.

Run `examples/modal_capillary.py` and `examples/custom_mode.py` for complete
simulations. The geometry fixture is `AMALTHEA_SPATIAL_ORACLE`, from
`tools/export_spatial_oracle.jl`; complete modal comparisons use
`AMALTHEA_MODAL_CAPILLARY_ORACLE`, from `tools/export_modal_capillary_oracle.jl`.

### Internal scalar capillary API

Run `examples/capillary.py` for a constant capillary with Kerr and loss:

```python
from amalthea_native import prop_capillary
result = prop_capillary(125e-6, .02, "Ar", 2., envelope=True,
                       lambda0=800e-9, lambda_lims=(400e-9, 1700e-9),
                       trange=300e-15, tau_fwhm=20e-15, energy=10e-6)
```

The positional arguments are radius [m], length [m], gas identifier, and
pressure [bar]. A single HE1m signifier (`modes="HE11"` by default) uses
mode-averaged propagation with `model="full"|"reduced"`, loss, Kerr and the
existing pulse inputs. Mode collections and polarization pairs use modal
propagation as described above.
Eligible constant configurations use resident Rust evaluation with portable
FFTs under `backend="auto"|"native"`. `backend="python"` uses NumPy evaluation
and Rust stepping; auto reports the path selected for each configuration. Results and solver
controls follow the GNLSE API. Noise remains disabled.

Use `envelope=True` for envelope propagation. For carrier-resolved propagation,
leave it false. Plasma is enabled by default and uses resident Rust evaluation
for supported ADK/PPT tables. Set `plasma=False` for Kerr/loss alone.
Carrier THG defaults on and runs natively; `thg=False` uses Python analytic
intensity under `backend="auto"|"python"`, with forced native execution rejected.
Results retain frequency-first ordering; `temporal_field()` returns real arrays
for carrier fields. See `examples/carrier_capillary.py` for both THG settings.

Molecular Raman is enabled by default for N2, H2, D2, N2O, CH4 and SF6 on
both grids, including carrier ADK/PPT combinations. Use `raman=False` to disable
it, or select components with `rotation` and `vibration`. Atomic gases infer
Raman off. Envelope THG uses Python evaluation when Kerr is enabled; carrier
THG-off Raman also uses Python. Auto reports these choices and forced native
rejects them. See `examples/raman_capillary.py` for simulations and controls.

Envelope plasma is unsupported by the Julia oracle. Multimode, polarized and
custom runtime models use the modal/callback interfaces above. Callable profiles
are evaluated at the requested positions. See the capability matrix for
combination restrictions and the backlog for pending platform acceptance.

`tools/export_capillary_oracle.jl` provides the seventh fixture directory,
`AMALTHEA_CAPILLARY_ORACLE`. It compares independently prepared setup and
trajectories for both backends, with separate Kerr/loss sensitivity checks.

`tools/export_real_capillary_oracle.jl` provides the ninth fixture directory,
`AMALTHEA_REAL_CAPILLARY_ORACLE`, for independent carrier setup/trajectory and
THG sensitivity checks. `AMALTHEA_REAL_PULSE_ORACLE` supplies the eighth input
fixture directory. These independent fixtures cover implemented propagation;
public preview acceptance is tracked in the backlog.

### Pressure and radius profiles

`prop_capillary(radius, length, gas, pressure, ...)` accepts callable radius
and pressure, a two-point pressure tuple `(p0, p1)`, or multipoint data
`(positions, pressures)`. Positions are metres and pressures are bar. Explicit
gradient data use square-root interpolation of pressure squared and clamp to
the first/last pressure outside their endpoints. Their density spline retains
Julia's 1024-node thermodynamic convention. Callable pressure uses direct
CoolProp evaluation; callable radius and pressure are evaluated at requested
positions without a sampled replacement.

```python
result = prop_capillary(lambda z: 125e-6*(1 + .1*z/.02), .02, "N2", (1., 3.),
                       envelope=True, lambda0=800e-9,
                       lambda_lims=(400e-9, 1700e-9), trange=300e-15,
                       tau_fwhm=20e-15, energy=10e-6)
```

Profiles run with Python evaluation and Rust stepping under `backend="auto"`
or `"python"`; forced native execution raises. Density, Raman linewidths,
effective area, dispersion and nonlinear normalization refresh at the required
positions. Both grids, supported THG choices and carrier plasma combinations
are available. Callbacks execute serially, must return finite real scalars
(positive radius, nonnegative pressure), and preserve their original exceptions.
They also receive the final accepted position beyond the requested length,
matching the solver's stopping convention. Gradient arrays and saved metadata
are copied. An all-zero structured gradient raises because Julia's density
spline has duplicate knots; scalar/callable vacuum remains available where its
selected molecular response is defined.

Run `examples/profile_capillary.py` for both grids with gradients, arbitrary
pressure and a taper. The independent fixture variable is
`AMALTHEA_PROFILE_CAPILLARY_ORACLE`; populate it with both
`tools/export_profile_capillary_oracle.jl` and `tools/export_profile_limits_oracle.jl`.

### Gas and response mixtures

Use a sequence of gas names with matching partial pressures:

```python
result = prop_capillary(125e-6, .02, ("Ar", "Ne"), (2., 1.),
                       envelope=True, lambda0=800e-9,
                       lambda_lims=(400e-9, 1700e-9), trange=300e-15,
                       tau_fwhm=20e-15, energy=10e-6)
```

Each pressure belongs to one species and may be a scalar, callable or gradient.
For example, `((1., 3.), (2., 4.))` specifies two gradients. A single gas name
still interprets `(p0, p1)` as one gradient. Mixture susceptibilities are added
before calculating the refractive index; density and nonlinear responses use
each species' partial pressure. Duplicate gases and one-species sequences are
supported.

Global response options apply to all species. `species_options` optionally
supplies one mapping per gas, overriding `kerr`, `raman`, `plasma`, `rotation`,
`vibration`, `PPT_options` or `preionfrac`. Raman and ADK/PPT defaults resolve
separately for every gas. For an N2/H2 carrier mixture, for example,
`species_options=[{"plasma": "PPT"}, {"plasma": "ADK", "preionfrac": .001}]`
selects independent rates and an initial H2 ionisation fraction. Grid, THG,
temperature and propagation controls remain common.

Constant Kerr-only mixtures can use resident Rust evaluation. Mixtures with
Raman or plasma use Python evaluation and Rust stepping; auto reports the path
and forced native execution rejects unsupported combinations. Changing pressure
refreshes every species' density and molecular linewidths at requested positions.
For changing profiles, refine `max_dz` to check convergence independently of
`rtol`: the variable linear operator retains Julia's endpoint exponential, so
nonlinear local-error control alone does not bound the complete equation's error.
`examples/mixture_capillary.py` runs both grids, native/Python Kerr comparisons,
profiles and molecular/plasma mixtures, and an NPZ round trip. The independent
fixture is `AMALTHEA_MIXTURE_CAPILLARY_ORACLE`, exported by
`tools/export_mixture_capillary_oracle.jl`.

### ADK ionisation setup

`IonRateADK("Ar")` constructs a rate evaluator for real electric fields in V/m;
calling it returns rates in 1/s with the same scalar/array shape. A numeric
first argument specifies the ionisation potential in joules. Options are
`occupancy=2`, `threshold=True` and `cycle_average=False`. Results own their
memory; invalid or nonfinite inputs fail explicitly. Zero field returns zero,
including when threshold detection is disabled.

`materials.ionisation_potential(material, unit="SI")` also accepts `"eV"` and
`"atomic"`. Potentials retain PhysData's material aliases and atomic hydrogen;
Air has no default potential. `tools/export_adk_oracle.jl` supplies the tenth
fixture directory, `AMALTHEA_ADK_ORACLE`. Plasma propagation has its own
trajectory checks described below.


### PPT rates and local caching

`IonRatePPT("Ar", 800e-9)` evaluates a PPT rate directly. Numeric construction
uses `(ionisation_potential_J, wavelength_m, Z, l)`. It supports Stark/dipole
corrections, `msum`, `cycle_average`, `sum_integral`, `sum_tol`, `Cnl` and
`occupancy`. An occupancy callable receives each magnetic quantum number,
serially for each field value; its exceptions propagate. `delta_alpha`/`Δα`
and `alpha_ion`/`α_ion` accept explicit SI corrections. Missing material
correction data means zero correction; missing quantum numbers fail explicitly.

`IonRatePPTAccel("Ar", 800e-9)` (also `IonRatePPTCached`) generates a 65536-node
table locally, then caches it under the user's cache directory. Supply
`cachedir=...`, `cache=False`, `N=...` or `Emax=...` to control construction.
Cache keys include every model/numerical option and the convention version;
corrupt entries are regenerated and writes are atomic. Occupancy callbacks
require `cache=False`. No Julia or runtime download is used. Large-argument
special functions use isolated high-precision mpmath contexts.

Tables interpolate log rates using Julia's normalized-knot cubic convention,
return zero below the first retained node, and clamp above the last. For custom
samples, use `IonRatePPTAccel.from_samples(field, rate)` with strictly increasing
positive fields and nonnegative rates; at least four positive rates are needed.
`field_nodes` and `rate_nodes` return owned copies. Direct rates and tables
accept real scalar/array fields in V/m and return rates in 1/s. Exactly zero
returns zero; invalid/nonfinite fields fail explicitly.

Run `examples/ppt.py` for default-size local construction and cache reuse.
`tools/export_ppt_oracle.jl` supplies the eleventh fixture directory,
`AMALTHEA_PPT_ORACLE`. Plasma trajectories have separate end-to-end acceptance
fixtures described below.

### Carrier plasma propagation

`prop_capillary(..., plasma="ADK"|"PPT")` adds plasma current and ionisation
loss. `plasma=None|True` selects Julia's default: ADK for H2/D2/N2O/CH4/SF6,
PPT for other gases with supported material data. `plasma=False` disables it.
`backend="auto"|"native"` uses resident Rust for standard ADK and locally built
or supplied PPT tables, with THG-on Kerr or Kerr disabled. Explicit Python
evaluation remains available. `result.metadata` reports the selected path.

Pass `PPT_options={"N": 65536, "cachedir": "./ppt-cache"}` (or `ppt_options`)
to configure local tables, or pass an existing `IonRateADK`, `IonRatePPT` or
`IonRatePPTAccel` object as `plasma`. Direct PPT, threshold-free ADK and custom
subclasses use Python under auto; forced native raises. THG-off Kerr also uses
Python. Tables whose polynomial scaling exceeds float64 representation retain
Python's normalized-coordinate evaluation. Nonempty PPT options require PPT table
construction. `preionfrac` accepts [0,1] and preserves Julia's formula
`preionfrac + 1 - exp(-integrated_rate)` exactly. Molecular Raman can be combined
with carrier plasma; envelope plasma is unsupported. See
`examples/plasma_capillary.py` for complete ADK/PPT simulations and controls.
`tools/export_plasma_capillary_oracle.jl` provides the twelfth independent
fixture directory, `AMALTHEA_PLASMA_ORACLE`.

### Molecular Raman setup

`MolecularRaman(time, gas)` constructs rotation/vibration responses for N2,
H2, D2, N2O, CH4 and SF6. Call it with number density [m^-3] to get owned
causal response samples. `rotation`, `vibration`, `minJ`, `maxJ` and
`temperature` control setup. Density-dependent linewidths refresh at each call.
`oscillators(density)` returns copied `omega`, `coupling` and `tau2` arrays;
`groups(density)` retains rotational/vibrational grouping. Setup retains the
project's absolute response normalization and tail window.

O2's selected Raman components lack lifetimes in the Julia model and raise.
Zero-density H2/D2/CH4 vibration also raises because the oracle response has a
nonfinite origin. Empty responses with both components disabled are supported.
See `examples/molecular_raman.py` for setup and `examples/raman_capillary.py`
for complete propagation. Independent setup fixtures come from
`tools/export_molecular_raman_oracle.jl` via `AMALTHEA_MOLECULAR_ORACLE`.
The fourteenth fixture, `AMALTHEA_RAMAN_CAPILLARY_ORACLE`, is exported by
`tools/export_raman_capillary_oracle.jl` and checks both grids, plasma mixtures,
component/temperature controls and supported THG choices end to end.

### Custom nonlinear responses

Both entrypoints accept `responses=callback` or an ordered list of callbacks.
Each `callback(field, context)` returns physical nonlinear polarization with
exactly the same shape as `field`. Responses append to the built-ins; set
`kerr=False`, `raman=False` or `plasma=False` to replace that term. Nonempty
custom responses select Python evaluation with Rust stepping; forcing
`backend="native"` raises before any response is invoked.

The field is an owned complete oversampled time array, `(ntime, 1)` for scalar
fields and GNLSE, or `(ntime, 2)` for vector modes. Carrier fields require real
polarization; envelope fields accept complex polarization. The library applies
the time window, FFT and spectral/modal normalization after accumulation.
There is no implicit density scaling of a custom response. Return finite numeric
arrays; incorrect shapes, nonnumeric values and imaginary carrier polarization
raise. Calls run serially, and original callback exceptions propagate.

`ResponseContext` exposes `z` (metres), `t` (seconds), `is_real`, `components`,
`gases` and `densities` (particles/m³ in gas order). GNLSE has empty gas/density
arrays. Modal callbacks also receive `coordinate_system` (`polar` or
`cartesian`) and the actual spatial `coordinates`; these are `None` for scalar
mode averages and GNLSE. Profiles refresh at the solver's requested positions,
including rejected trials and the final accepted step beyond the output range.
Fields and context arrays belong to each invocation and may be retained or
mutated. Returned polarization is copied before use. Saved parameters contain
callback names; results do not retain callback objects.

For scalar envelope capillaries, an equivalent Kerr response is:

```python
from amalthea_native.materials import EPS0, gamma3

def envelope_kerr(field, context):
    coefficient = 0.75 * EPS0 * sum(gamma3(gas)*rho
                                   for gas, rho in zip(context.gases, context.densities))
    return coefficient * field * abs(field)**2
```

Use this with `envelope=True, thg=False, kerr=False, responses=envelope_kerr`.
Carrier THG Kerr instead uses `EPS0 * sum(gamma3(gas)*rho) * field**3`.
Vector responses require cross-component coupling; see the complete equivalent
Kerr and illustrative delayed-convolution simulation in
`examples/custom_response.py`. This example uses every time sample at a spatial
node and compares its Kerr replacement with the built-in result.

GNLSE retains its distinct normalization: the instantaneous polarization
coefficient is `EPS0**2 * C * (1-fr) * gamma / (2*pi/lambda0)` multiplying
`field * abs(field)**2`; no gas density is present. Disabling built-in Raman
preserves Julia's `(1-fr)` instantaneous fraction. A custom model replacing the
entire GNLSE response can set `gamma=0` and return its full polarization.
Execution metadata reports `custom_response_calls`, callback names and the
selected Python path.
