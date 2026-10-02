# Standalone Python capability and execution matrix

This reference describes implemented `amalthea_native` behavior. Release state,
unfinished validation and immediate work belong in the
[BACKLOG resume queue](../BACKLOG.md#start-here--current-resume-queue-2026-09-06).
Numerical evidence is recorded in [PORT_LOG](PORT_LOG.md); acceptance rules are
in [TESTING](TESTING.md). The implementation contract is
[PYTHON_NATIVE_PLAN](PYTHON_NATIVE_PLAN.md). This matrix does not establish a
public preview or platform acceptance by itself.

## Execution paths

Rust owns adaptive stepping, accepted-step filtering and dense sampling for
both high-level entrypoints. Python owns physical setup, NumPy results and
serialization. No path imports or provisions Julia.

| Requested backend | Mode-averaged/GNLSE behavior | Modal behavior |
|---|---|---|
| `auto` | Resident Rust evaluation when the complete configuration is eligible; otherwise serial Python evaluation | Serial SciPy quadrature and mode projection; eligible point arrays use Rust |
| `native` | Resident Rust, or an explicit error for an ineligible configuration | Explicit error: the complete spatial evaluator uses Python/SciPy |
| `python` | Python response and linear-profile evaluation with Rust stepping | Python temporal responses, spatial projection and SciPy quadrature with Rust stepping |

Metadata records `backend`, `stepper`, selection reasons and, for modal solves,
`point_evaluator`/`point_fft`. Native points do not imply that the entire modal
simulation is native. Constant built-in modes, ordinary Kerr and scalar sampled
Raman are point-eligible. Profiles, constructed/custom modes, plasma, custom
nonlinear callbacks, carrier THG-off and envelope THG retain Python points.

## Physical setup and propagation

Paths below describe automatic selection; every supported configuration can
also request Python evaluation. The test references are files under
`python-native/tests/` and use independent fixtures exported by
`python-native/tools/export_*_oracle.jl` where applicable.

| Capability | Implemented evaluation | Acceptance coverage |
|---|---|---|
| Real/envelope grids, unshifted spectra, oversampling and windows | Python setup; portable RustFFT/RealFFT in resident kernels | `test_foundation.py`, `test_real_resident.py` |
| Gaussian/sech pulses, energy or peak power, spectral Taylor phase, coherent pulse collections | Python setup on both grids | `test_pulses.py`, `test_real_pulses.py`, GNLSE/capillary trajectories |
| Sampled data pulses, supplied time/frequency arrays, modal arrays and custom input propagators | Python setup with owned arrays and serial callbacks | `test_pulses.py`, `test_real_pulses.py`, `test_modal.py` |
| GNLSE Taylor dispersion, loss, Kerr, shock, SDO/SiO2 Raman | Resident Rust; Python with custom nonlinear responses | `test_gnlse.py`, `test_raman.py`, `test_resident.py`, `test_callbacks.py` |
| Gas density/inverse pressure, refractive index, polarizability, Kerr coefficient | Python/CoolProp 8.0.0, project constants and material data | `test_materials.py` against Julia/CoolProp 7.2.0 at 1e-13 |
| Marcatili HE/TE/TM, full/reduced model, attenuation, area and group velocity | Python mode setup | `test_modes.py`, scalar and modal propagation |
| Constant mode-averaged envelope/carrier Kerr | Resident Rust; Python for carrier THG-off/envelope THG | `test_capillary.py`, `test_real_capillary.py`, `test_modal.py` |
| Scalar molecular rotational/vibrational Raman and mixtures with Kerr | Resident Rust at supported constant density; Python for profiles/THG exceptions/mixtures | `test_molecular.py`, `test_raman_capillary.py`, `test_profiles.py`, `test_mixtures.py` |
| ADK/PPT rate setup, occupancies, cycle average, Stark/dipole options, direct/table/cached rates | Python setup; local high-precision branches and parameter-keyed tables | `test_adk.py`, `test_ppt.py` |
| Constant scalar carrier plasma, preionisation, Kerr and Raman | Resident Rust for eligible owned ADK/PPT data; Python for direct/custom/threshold-free rates and response mixtures | `test_plasma.py`, `test_native_plasma.py`, `test_raman_capillary.py`, `test_mixtures.py` |
| HE/TE/TM mode collections, orthogonal polarization pairs, full spatial fields | SciPy global-norm quadrature; eligible Rust points | `test_spatial.py`, `test_modal.py`, `test_native_points.py` |
| Linear/x/y/circular/elliptical input polarization, scalar/vector Kerr | Python input construction; eligible Rust scalar/vector points | `test_modal.py`, `test_native_points.py`, `test_callbacks.py` |
| Modal scalar Raman on both grids, including THG variants supported by Julia | SciPy quadrature; eligible scalar Rust convolution, Python otherwise | `test_modal.py`, `test_native_points.py` |
| Modal carrier plasma with scalar/vector fields, full projection, ADK/PPT and preionisation | Python temporal response and SciPy quadrature | `test_modal.py`, conditioned intermediate/refinement gates, `test_callbacks.py` |
| Two/multipoint pressure gradients, arbitrary pressure/radius functions, tapers | Python at the actual requested positions; fresh density, dispersion, area and response coefficients | `test_variable_solver.py`, `test_profiles.py`, `test_modal.py`, `test_callbacks.py` |
| Gas mixtures with per-species Kerr/Raman/plasma and partial-pressure profiles | Native collapsed constant scalar Kerr; Python per-species responses otherwise; eligible modal Kerr points | `test_mixtures.py`, `test_modal.py`, `test_native_points.py`, `test_callbacks.py` |
| Custom mode dispersion/spatial fields, polar/Cartesian domains, analytic or generic normalization | Python mode protocol and SciPy quadrature | `test_spatial.py`, `test_modal.py`, `test_callbacks.py` |
| Complete-array custom nonlinear response mixtures | Serial Python callbacks on owned time/component fields and context snapshots | `test_callbacks.py`, including Cartesian mode + profile + response combinations |

All combinations remain subject to the current Julia model's domain. The
same-input point/RHS/dense checks isolate numerical transport; complete solves
and nonzero physics controls establish representative combined behavior.
An arbitrary custom function cannot be certified from a few sample values;
its shape, finite results, exceptions and lifecycle are checked at invocation.

## Conventions and explicit exclusions

- Units follow Julia: metres, seconds, joules, watts, kelvin; gas pressure is
  in bar and angular frequency in rad/s. Existing Greek/ASCII aliases and
  duplicate-alias rejection are retained.
- Mode-averaged fields have frequency/save axes. Modal results add a mode
  axis between them, including a one-element mode collection. Selected
  polarization pairs retain Julia's ordering and energy split.
- GNLSE retains the `(1-fr)` Kerr scaling even with Raman disabled. Density
  and sampled molecular response scaling retain the Julia response convention.
- Pressure functions are evaluated directly, without inferred constancy or
  sampled substitution. Explicit gradient data retains Julia's thermodynamic
  spline/endpoint convention. Variable linear propagation uses Julia's endpoint
  exponential; convergence may require refining the maximum propagation step.
- Envelope plasma and vector Raman are unsupported by the Julia oracle.
  Carrier vector Kerr with THG disabled is also an explicit oracle restriction.
  O2 molecular lifetime data and undefined zero-density vibrational responses
  raise errors where Julia lacks a valid model. Material domains still apply.
- Quantum noise is disabled, defaults false, and explicit requests reject.
  Seeded ensembles, Python GPU execution, and separate free-space/step-index
  interfaces remain outside this release contract.
- Julia scan/output objects, RNG objects and live callbacks are not serialized
  into Python result files. The distribution supplies owned arrays and the
  output interface below; this is not a drop-in Julia object wrapper.

## Solver, results and installation contract

`solve_precon` supports constant or callable diagonal operators, scalar/matrix
fields, nonzero start positions, fourth/fifth-order controls, rejected trials,
dense sampling, restart and accepted-step filters. Callback exceptions retain
their identity. Output sampling/filter order and initial-step bounds match the
repaired Julia driver. See `test_solver.py` and `test_variable_solver.py`.

`PropagationResult` exposes owned spectra/positions/grid/parameters/metadata,
`temporal_field()`, pickle-free `save_npz` and optional `save_hdf5`. Both formats
retain original numeric arrays and use versioned JSON descriptions. The `hdf5`
extra is lazy; base import/simulation/NPZ does not require h5py. Independent
readers and complete examples are covered by `test_output.py`.

Wheel installation is CPU-only and must require no Julia, compiler, system
FFTW/libcubature or runtime download. The explicit Linux build baseline is
manylinux_2_28. CPython 3.11–3.14 and Linux x86_64/ARM64, Apple Silicon and
Windows x86_64 are acceptance targets, not inferred support claims. Their
actual build/install/offline status lives only in BACKLOG and dated PORT_LOG
evidence. The provisional distribution name must be checked before publication.
