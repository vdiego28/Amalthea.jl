# Julia-free Python distribution

Design authorized by the 2026-09-07 implementation roadmap. Current execution
status belongs in [BACKLOG](../BACKLOG.md); evidence belongs in
[PORT_LOG](PORT_LOG.md). This replaces the Python-specific applicability of
the parked CLI recommendation in [PLANS §4](PLANS.md#4-standalone-cli-luna-cli-backlog-s6-item-3),
without changing the existing Julia architecture or authorizing a CLI port.

## Release contract

Create a separate `amalthea-native` distribution, imported as `amalthea_native`,
alongside the existing `python/` Julia wrapper. The name is provisional until
availability is checked immediately before publication. No publication is part
of preparing local artifacts.

The first public Linux preview requires both GNLSE and full Julia-valid
capillary coverage, including custom Python models. Earlier wheels are internal.
Stable release additionally requires Linux x86_64/ARM64, Apple Silicon, and
Windows x86_64 wheels, with CPython 3.11–3.14 installation tests. CPU-only,
portable flags; use an explicit manylinux baseline (initial target:
manylinux_2_28). Audit actual binary dependencies and compatible dependency
wheels before accepting that target. No runtime downloads, Julia provisioning,
compiler, system FFTW, or libcubature may be required on wheel platforms.

Quantum noise defaults off and an explicit request for it raises an error.
Envelope plasma and vector Raman are excluded physics features, not Python
fallback cases. Separate free-space/step-index entrypoints, GPU Python support,
seeded noise, and ensemble APIs follow this release.

## Ownership and implementation seams

Python owns setup and results: NumPy arrays, SciPy numerical setup/quadrature,
and CoolProp 7.2.0 thermodynamics initially. Copy physical constants and model
coefficients from `PhysData.jl`, not library defaults. Preserve the grid and
pulse conventions in `Grid.jl`/`Fields.jl`, including oversampling, windows,
spectral halves, normalization, and field-axis ordering.

Rust owns adaptive stepping for both execution paths. Build a safe driver around
the existing corrected resident engine, retaining the Julia C ABI. Port the
outer lifecycle from `RK45.jl` and `Luna.jl`: rejected-attempt preservation,
deferred FSAL, fourth-order starting-derivative recomputation, extra-stage dense
output, accepted-step filtering, field resynchronization, stopping, and output
sampling. Do not substitute `stepper.rs`'s standalone implementation without
first verifying that it has all these repaired semantics. Construction must
validate array dimensions and numerical values before reaching unsafe kernels.

Introduce RustFFT/RealFFT behind an explicit portable FFT implementation;
retain FFTW for Julia. Reuse plans and scratch, including per-worker modal
scratch. Normalize at the existing engine's normalization seams. Test complex
and real transforms, odd/even lengths, DC/Nyquist, spectrum ordering, inverse
input preservation, oversampling, Raman convolution, and Hilbert conventions.
No hidden process-global FFT selection may change a Julia handle's behavior.

The PyO3/maturin extension lives in a separate `python-native/` package. Keep
the existing crate usable by Julia without Python dependencies by isolating
bindings in a separate crate or optional feature. The sdist must contain the
Rust engine sources needed to build the extension; verify by rebuilding an
extracted sdist outside the checkout. Never copy a prebuilt checkout `.so`
into the wheel as a substitute for building a portable artifact.

## API and callback contracts

Preserve positional arguments:
`prop_gnlse(gamma, flength, betas, ...)` and
`prop_capillary(radius, flength, gas, pressure, ...)`, physical units, and
ASCII/Unicode keyword aliases. Reject conflicting aliases, unknown options,
and unsupported combinations explicitly. In particular, preserve the current
GNLSE `(1-fr)` Kerr scaling even with `raman=false`; changing this convention
would be a separate scientific change.

`backend="auto"` chooses eligible native evaluation or Python evaluation and
reports the actual path and reasons in result metadata. `"native"` rejects
unsupported configurations; `"python"` uses Python model evaluation with the
same Rust step driver. Fallback must never mean Julia provisioning or dropping
a response. Native eligibility must cover the entire configuration.

Return owned NumPy arrays for spectral fields, saved positions, grid axes,
parameters, and execution metadata, preserving frequency/mode/save ordering.
Provide NPZ output and optional HDF5 via `h5py`. Callback objects and transient
views must not leak into serialized parameters; record descriptive metadata.

Custom modes supply dispersion and spatial fields with explicit normalization
and integration domains. Pressure/radius callables are evaluated at requested
positions, including linear quadrature and dense stages; never infer constancy
from a finite probe or silently replace a callable with a sampled spline.
Array pulse inputs retain their specified axes and energy conventions.
Nonlinear callbacks receive complete scalar/vector time-field arrays rather
than scalar samples. Execute callbacks serially within a simulation, preserve
their original Python exceptions, and reject wrong shapes/nonfinite outputs.
Keep callback references alive for the solve; release them on every exit path.

Initially use SciPy quadrature for modal integration, with vectorized Rust
point evaluation where applicable. The driver must aggregate component error
estimates and enforce the global criterion used by `NonlinearRHS.TransModal`;
per-component SciPy success alone is insufficient. Check convergence against
independently refined integrations. Python callback paths must refresh density,
Raman broadening, mode normalization, and response coefficients at each
required position, including mixed species.

## Milestones and gates

1. Establish the repaired baseline: inspect hosted checks for the exact repair
   commit, retain numerical evidence, preserve installer/validation edits,
   and inventory the capability matrix below. The lead explicitly authorized
   implementation while hosted CI runs on 2026-09-07; inspect its output after
   making Python progress. Hosted success remains an acceptance gate and is
   not inferred from local tests.
2. Build the safe driver, portable FFTs, and internal extension wheel. Run an
   analytic simulation without Julia/FFTW/libcubature; verify rejection,
   restart, fourth/fifth-order dense output, filters, and stopping behavior.
3. Port GNLSE and basic capillary setup: grids, pulses, Taylor dispersion,
   loss, Kerr/shock, SDO/SiO2 Raman, gases, Marcatili, ADK/PPT. Generate PPT
   tables locally with stable high-precision branches and caches keyed by
   every rate-affecting parameter, model version, and table bounds. Compare
   independently prepared Julia and installed Python simulations.
4. Complete modal/polarization, molecular, gradient/taper, mixture, plasma,
   and custom protocols. Every promised combination needs an end-to-end
   oracle case; native guard gaps use Python evaluation. Reject invalid Julia
   combinations explicitly.
5. Build distributions and examples for both APIs, custom models, processing,
   and diagnostics. Run clean-environment installed-artifact tests outside the
   repository, including wheels rebuilt from sdist and complete examples with
   networking disabled after installation. Linux preview waits for all
   scientific coverage; stable waits for all promised platforms.

For each numerical unit use [TESTING](TESTING.md)'s same-input (~1e-13) and
full-solve (normally 1e-6) tiers, with measured feature sensitivity greater than
the asserted tolerance. Compare setup quantities before trajectories. Cover
callback errors and repeated construction/destruction. Final regression gate:
`python3 test/validate.py --all`, Python tests, and affected FFI scripts.
Shared CUDA changes require the existing strict hardware gate.

## Planned capability matrix

These are implementation assignments, **not claims of Python support**.
All Python rows require new installed-artifact evidence. Source inspection
uses `Interface.jl::{_prop_gnlse_args,_prop_capillary_args,makeresponse}` and
`RK45.jl::RustNativeStepper` guards; the existing
[native matrix](NATIVE_SUPPORT_MATRIX.md) summarizes Julia dispatch only.
Python evaluation is the coverage backstop for any Julia-valid combination
outside native eligibility. Examples are candidate oracle fixtures, not proof
that their complete runs have been repeated for this migration.

| Capability | Planned evaluator | Source / oracle fixtures |
|---|---|---|
| Grids, windows, Gaussian/sech/supplied pulses, units and aliases | Python setup | `Grid.jl`, `Fields.jl`, `Interface.jl`; `test_interface.jl` |
| GNLSE Taylor dispersion, loss, Kerr, shock | Native | `SimpleFibre.jl`, `norm_mode_average_gnlse`; `test_gnlse.jl`, `gnlse_sol.jl` |
| GNLSE SDO and SiO2 Raman | Native | `Raman.jl`; `test_native_raman_sio2.jl`, `gnlse_ssfs.jl` |
| Gas properties and Marcatili setup, full/reduced model, mode loss | Python setup | `PhysData.jl`, `Capillary.jl`; `test_interface.jl` |
| Mode-averaged RealGrid/EnvGrid Kerr, THG variants | Native where eligible, otherwise Python | `Nonlinear.jl`, mode-averaged guards; `basic_modeAvg_env_THG.jl` |
| Mode-averaged real plasma, PPT and ADK, preionization | Native | `Ionisation.jl`, `PlasmaCumtrapz`; `test_native_phase2.jl` |
| Molecular rotation/vibration, scalar Raman, density broadening | Native at supported constant density, Python otherwise | `Raman.jl`; `test_native_raman_env_rotational.jl` |
| Constant-density mixtures, Kerr | Native | `Et_to_Pt!`; `test_mixtures.jl`, `test_native_modal_mixture.jl` |
| Julia-valid mixtures with plasma/Raman | Python | per-species `makeresponse`/`Et_to_Pt!`; `mixture_modeAvg.jl` (extend fixture for non-Kerr effects) |
| Multimode HE/TE/TM, full spatial fields, 1–2 polarizations, plain Kerr | Python quadrature + native point evaluation | modal guards; `basic_modal_full_bothpolarisations.jl` |
| Modal scalar RealGrid Raman | Python quadrature + native point evaluation when eligible | `test_native_modal_raman.jl` |
| Modal scalar EnvGrid Raman, no-THG/EnvGrid-THG guard gaps | Python | `TransModal`, `RamanPolarEnv`, modal response guards |
| Modal RealGrid plasma, scalar/vector field and full projection | Python | `test_transmodal_vector_plasma.jl`, `modal_vector_plasma.jl` |
| Two/multi-point pressure gradients, arbitrary pressure/radius functions, tapers | Python position evaluation, native kernels where valid | `test_gradient.jl`, `test_tapers.jl`, `gradient_modeAvg_env.jl` |
| Custom modes, dispersion, spatial fields and nonlinear array responses | Python | generic `Modes`/`TransModal` contracts; built-in/custom equivalence fixtures required |
| Owned results, backend diagnostics, NPZ/optional HDF5 | Python | `Output.jl`, `Luna.jl::backend_report` conventions |
| Envelope plasma, vector Raman, quantum noise | Reject | explicit release exclusions; EnvGrid `makeresponse` plasma rejection |

Notable source finding: modal guards reject `Kerr_field_nothg` and
`Kerr_env_thg`, even where broad Kerr summaries suggest support. Callable
tapers must also bypass existing sampled native taper tables. Both belong in
Python coverage, not in a looser native eligibility check.

## First implementation unit: grids and portable transforms

Implement independently usable `RealGrid`/`EnvGrid` setup and a private PyO3
portable transform class in the mixed `python-native/` package. This unit has
no propagation entrypoints and must not present itself as a preview release.
Grid construction mirrors `Grid.jl` arithmetic, including one-based crop-index
rounding before conversion to Python slicing. Window construction mirrors
`Maths.planck_taper`'s centered coordinates. Reject nonfinite/nonpositive setup
parameters and invalid grid bounds before allocation. An explicit sample limit
may reject oversized allocations; it must never coarsen a requested grid.

Each private FFT object owns RustFFT/RealFFT plans, reusable scratch, and input
copies. Both directions are unnormalized, matching the resident engine seam.
Python views are never retained. Exercise input preservation and FFT ordering
against NumPy and independently generated Julia data, including odd/even
lengths. This isolated module will subsequently be integrated into the engine;
its existence alone does not establish a portable resident simulation.
Generate grid oracle fixtures by directly invoking Julia constructors in a
development-only script; runtime imports must never load Julia. Record every
axis/window relative error and exact selection-mask equality. Wheel-build and
outside-checkout imports are internal artifact checks only.

## Performance sequence

## Callback driver implementation unit (2026-09-08)

Use the existing corrected `ffi::precon_step_ffi` kernel through an owning
Rust RAII adapter in the Python extension. This is the first safe standalone
driver backend, not a new DOPRI implementation or a claim of resident-native
physics. Its field, stages, scratch handle, PI state, and outer solve loop live
in Rust. Link the engine as a Cargo path dependency; maturin's sdist must bundle
that dependency and rebuild outside the checkout. The extension's build must
force the engine's CPU-only policy even if host CUDA is available.

Expose `solve_precon(rhs, linop, field, zmax, ...)` as a development-level
spectral solver. This initial driver accepts constant diagonal `linop`; custom
position-dependent nonlinear RHS is evaluated at actual requested stages.
Python adapters pass complete owned NumPy arrays with original shape and
Fortran flattening at the boundary. Reject shape/nonfinite callback outputs;
capture Python exceptions in the callback context, stop invoking callbacks on
error, and rethrow the original exception after returning from the C boundary.
Catch Rust panics inside each extern callback so no unwind crosses C.

Port the Julia quintic coefficient table and its two extra stages; false mode
uses the corrected quartic weights. Preserve deferred FSAL and stage buffers
until all dense requests for an accepted interval are finished. Match Julia's
strict-interior output cadence and loop through the first accepted endpoint
beyond zmax. Apply an optional `(z, field) -> field` filter only after accepted
steps and output sampling; the next attempt reads that filtered field without
refreshing fifth-order FSAL (fourth mode recomputes it). Preserve legacy PI
minimum-step forced acceptance but reject any nonfinite state regardless.
Bound total attempts and consecutive rejections; reject floating-point time
stagnation. Return owned `(field axes..., saved z)` arrays with accepted/rejected
counts and backend metadata explicitly naming Rust stepping/Python evaluation.

Validate independent nonlinear analytic solutions, fourth/fifth dense
convergence, nonzero linear propagation, rejected retries, filter cadence,
callback exceptions/invalid outputs and restart consistency. Generate an
independent Julia fixed-step and full adaptive oracle with `shotnoise` absent.
No public preview, GNLSE/capillary eligibility or custom spatial protocol is
implied by this low-level solver.

Saved-position refinement: ordinary NumPy linspace can straddle an accepted
filter discontinuity on the opposite side from Julia's compensated range.
Recover small exact-roundtrip rational endpoints (continued fractions bounded
by 2^24, as in Julia's Float64 range rational-detection strategy), then form
samples with rational arithmetic and one final Float64 rounding. If no small
roundtrip rational exists, interpolate exact binary endpoint fractions. This
fallback is accurately rounded but does not claim bitwise equivalence to every
Julia double-double range edge case. Assert exact coordinates and filtered
trajectories for common rational endpoints against a separate Julia oracle;
never hide a discontinuity with an epsilon in the sampling comparison.

Validation harness repair discovered during this unit: the existing
`test_transmodal_julia_threading.jl` top-level `return` does not prevent later
testitem expressions from being evaluated by the installed Julia/TestItemRunner
combination. A one-thread worker records a skip then runs assertions requiring
threaded scratch. Put the body in the explicit `else` branch of the thread
guard. Verify a one-thread skip and a real four-thread pass, then rerun the
affected group. No production modal threading changes or weaker assertions.

Preserve the frozen CPU audit. Create a separate post-repair snapshot measuring
import, first simulation, setup, warmed solve, output, memory, step counts, and
callback overhead against Julia and existing Rust paths. Complete coverage
before optimizing the largest measured complete-workload bottleneck. Retain an
optimization only for at least 5% complete-solve improvement or removal of a
demonstrated regression. QDHT, Raman SIMD, allocation, batching, and threading
work already completed must not be restarted. Collect Apple hardware evidence
during platform validation; cross-compilation is not runtime evidence.
