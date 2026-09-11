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
The later source audit records actual paths and acceptance-test assignments in
[PYTHON_SUPPORT_MATRIX.md](PYTHON_SUPPORT_MATRIX.md); release/platform status
remains in BACKLOG.
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

## First optical Python evaluator and hosted artifact gate (2026-09-09)

Advance coverage through the already verified callback driver while resident
portable integration remains pending. Add an internal `prop_gnlse(gamma, flength,
betas, **kwargs)` with `backend=auto|python`; `native` rejects explicitly until
its complete evaluator is available. Metadata records Python/NumPy evaluation
and Rust stepping. This unit covers Gaussian/sech pulses, owned complex time or
spectral arrays already on the constructed grid, Taylor phase/dispersion, loss,
Kerr, shock, and SDO Raman. SiO2 Raman, arbitrary-axis DataField interpolation,
pulse collections/custom propagators, and capillary entrypoints remain explicit
unsupported cases, not silent substitutions or preview-ready claims.

Reproduce the actual Julia pipeline before simplifying physics: CODATA2014
permittivity and speed of light; power normalization by peak |E|² and energy by
NumericalIntegration's extended Simpson weights (17,59,43,49)/48 at each end;
phase coefficients include orders zero and one; Taylor dispersion includes beta0
and beta1, then subtracts the moving frame. Clamp attenuation to [0,3000]/m as
LinearOps does. Preserve `(1-fr)` Kerr even with Raman disabled. Evaluate on the
oversampled envelope grid with both spectral halves and No/N scaling, physical
field normalization, causal doubled-grid SDO convolution with dt and the same
response tail window, polarization time window, then GNLSE normalization and
frequency window on sidx. Retain the raw polarization outside sidx exactly as
TransModeAvg does; accepted-step windows subsequently suppress those bins.

Use NumPy FFTs for the Python evaluator; the separately tested Rust portable
plans remain available for the subsequent resident integration. The accepted
filter is fft(twin*ifft(omega_win*field)), matching Amalthea.run. Preserve solver
output cadence and original field axes. New results expose owned spectral field,
z, grid, parameters, metadata, temporal reconstruction, and pickle-free NPZ
serialization. All keywords are checked, with the source's ASCII/Unicode aliases
and explicit alias-conflict errors. Quantum noise defaults off and true rejects.
High-level solver step controls use `init_dz`, `min_dz`, `max_dz` so `dt` remains
the existing alias for the temporal spacing `delta_t`.

Generate independent Julia setup, RHS, fixed-step interior, and full adaptive
trajectories with the pure Julia backend. Cover ordinary and oversampled grids,
energy/power and spectral phase, both pulse shapes, loss/Kerr/shock/Raman
sensitivity above 1e-6, and zero-nonlinearity analytic propagation. Do not weaken
~1e-13 same-input and 1e-6 full-solve assertions. Wrong/nonfinite pulse arrays,
unsupported options, serialization ownership and alias conflicts need tests.

Add a dedicated Linux hosted job: build/install the standalone wheel, export all
Julia development-only oracles, and test from outside the checkout; build an
sdist, extract it, rebuild/install that wheel, and run extracted tests/examples.
Keep ordinary wheel/runtime checks separate from Julia oracle preparation in
scope descriptions: no release platform/manylinux claim follows from this job.
Use explicit oracle environment variables so acceptance cannot pass with skipped
oracle tests. Preserve the existing wrapper CI job and CPU-only build policy.

## SiO2 Raman in the Python GNLSE evaluator (2026-09-09)

Port the fixed thirteen-component Hollenbeck–Cantrell coefficient table directly
from `PhysData.raman_parameters(:SiO2)`: convert wavenumbers with the same `200πc`
(center) and `100πc` (Gaussian/Lorentzian widths) factors. The causal response is
`sum(Ai*exp(-gamma_i*t)*exp(-Gamma_i²*t²/4)*sin(omega_i*t))`, scaled to unit
continuous integral on [0,1e-9] s, then multiplied by the existing `chi3R*epsilon0`.
Unlike CombinedRamanResponse/SDO, Julia's intermediate-broadening response has
no tail taper. Do not apply the SDO window or discretely renormalize the samples.
SDO tau1/tau2 are unused for SiO2 (including None), matching the Julia interface.
Retain the existing doubled-grid causal convolution, its half-intensity drive,
dt factor, response mixture scaling, and complete Python evaluator metadata.

Use SciPy's complex scaled complementary error function for normalization:
each component's integral to infinity equals
`Ai*sqrt(pi)/Gamma_i * imag(erfcx((gamma_i-i*omega_i)/Gamma_i))`.
Completing the square in the complex exponential gives this expression. For
these fixed coefficients the difference from Julia's finite 1 ns upper bound
is bounded by `sum(Ai*exp(-Gamma_i²*T²/4)/(Gamma_i²*T/2))`; the slowest Gaussian
already has exponent below -1e6 there. Thus the finite-tail correction is below
Float64 representability, not an altered model/truncation tolerance. Cache only
the fixed table's scalar normalization, independent of grid and nonlinear scale;
return fresh response arrays per construction. Add SciPy >=1.14 as a runtime
dependency and install it in the standalone CI job (wheels still select a version
compatible with each supported Python). Official API references:
[erfcx](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html),
[quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html).

Check the closed-form normalization against independently refined time-domain
quadrature and the Julia normalization. Quad's default absolute tolerance is
inappropriate for ~1e-12 s integrals; integrate in femtosecond coordinates with
explicit early-time breakpoints and epsabs=0, enforce a global relative error
estimate, and refine both tolerance and breakpoints. Include a resolved tail
before the 1 ns endpoint so quadrature cannot miss the narrow response near zero.
The response need not have unit *sampled* integral on a coarse time grid.

Expand the independent Julia fixtures with SiO2 on ordinary and oversampled
grids plus a long-window case exposing its untapered tail. Compare response,
RHS and single/fixed-step fields at 1e-13, full trajectories at 1e-6, and actual
high-level entrypoints. Prove SiO2 differs from Raman-off and SDO by more than the
asserted full-solve tolerance. Test direct causal convolution, zero Raman
fraction/zero gamma, repeated construction and independence of returned arrays,
ignored SDO widths, invalid model rejection, NPZ model metadata, and source-
rebuilt installed artifacts with offline examples. Re-run SDO/control fixtures
and recorded CPU Rust/propagation regression groups; no shared numerical Rust,
Julia C ABI, CUDA source, or solver-control changes are part of this unit.

Normalization refinement finding: Julia's default `hquadrature` returns
3.752586750599693e-12, while independent refinements at reltol=1e-11 and 1e-13
return 3.752586750631897e-12; the closed form is 3.752586750631900e-12.
The default quadrature error is 8.58e-12 relative, within its requested 1e-8 but
above the same-input 1e-13 gate. Do not hard-code the approximate oracle number
or relax that gate. In development-only SiO2 fixtures retain the original response
and actual unmodified high-level run, then replace just its response normalization
with a separate Julia hquadrature refinement at 1e-13 for response/RHS/fixed-step
comparisons. Export both normalizations and error estimates. The independent
Python setup still computes its own closed form; the refinement is never derived
from Python's answer. Assert 1e-13 against the refined oracle, and the unchanged
1e-6 full-solve tier against both refined trajectories and the actual default
Julia entrypoint. Record the unmodified response discrepancy separately and
check that it is explained by the independently measured scalar normalization.
This is evidence of quadrature convergence, not a Julia production-source change.

## Rich pulse inputs and custom input propagators (2026-09-09)

The active requested sequence includes rich pulse inputs, resident portable FFT
integration, and basic capillary support; completing pulse inputs alone does not
close that sequence or the broader release roadmap. This unit implements the
first dependency, with the later units retained in BACKLOG.

Add reusable Python `GaussPulse`, `SechPulse`, `DataPulse` and `PropagatedPulse`
objects. `pulses=` accepts one object or a nonempty ordered list/tuple and sums
independently prepared complex spectra coherently, without normalizing the sum.
Per-pulse lambda0, duration, power/energy and Taylor phase follow Fields.jl.
Offset pulse carrier frequency is `exp(+i*(omega_pulse-omega_grid)*t)` before
spectral phase and normalization. Preserve the existing simple pulse/array API.
The Julia interface ignores its simple pulse keywords when pulses is supplied;
follow that for built-in pulse defaults but reject simultaneous singular `pulse`
array and `pulses` to avoid ambiguous new Python-only inputs.

Extract existing shared energy/vector/phase helpers to the pulse module rather
than duplicate formulas. DataPulse accepts (omega, complex spectrum) or
(omega, intensity, phase), copying input arrays. Use SciPy FITPACK `splrep(k=3,
s=0)` and `splev` to match Dierckx and Julia's spline evaluation (despite those
SciPy interfaces' legacy label); defaults of newer interpolators are not an
unverified substitute. Require at least four unique positive frequencies;
accept unordered axes by sorting complete rows before interpolation. Complex
input phase is unwrapped in frequency order. Duplicate coordinates, nonfinite
arrays, negative intensities, or zero remaining spectrum are errors.

Follow DataField exactly: interpolate intensity and phase separately, clip
negative interpolated intensity, mask strictly inside the source bounds,
multiply intensity by grid frequency window, sqrt and phase, normalize via
Fields' spectral-energy sum, then apply `exp(-i*omega*Nt*dt/2)` to center the
pulse. Extra Taylor phase uses explicit pulse lambda0 or the source intensity-
weighted mean frequency (discrete moment), preserving that order. Supplied
spectra are not treated as time-grid-aligned arrays; DataPulse and the existing
`pulse_domain="frequency"` have distinct, documented centering semantics.

Custom `propagator(field, grid)` receives owned spectral field and an isolated
copy of grid setup. It may mutate and return None (Julia convention), or return
an array. Validate final exact field shape and finite values; propagate the
original Python exception. Invoke once per pulse after its normalization and
phase; retain ordered serial execution and copy outputs so retained user arrays
cannot alias solver state. Top-level propagator applies to the simple input;
collections carry per-pulse propagators as Julia does. Serialize descriptive
pulse/propagator metadata, never live callbacks or pickle-only objects.

Validate independently generated Julia spectra and nontrivial propagation for
multi-color coherent pulses, nonuniform supplied spectra and unwrapped phase,
custom propagators, and mixtures. Require setup/fixed interval ~1e-13 and full
trajectories 1e-6; compare the oracle to omitted-pulse/propagator controls above
tolerance. Check interpolation refinement, alias validation, ownership, errors,
repeated construction, and installed/source-rebuilt artifacts. Extend the
standalone CI's oracle inputs and offline example to exercise these paths.
References: SciPy [splrep](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html)
and [splev](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html);
local `Fields.jl::{DataField,PulseField,PropagatedField}` and `Maths.BSpline`.

Source-order refinement: DataField's complex-data constructor unwraps phase in
supplied order before BSpline sorts coordinates. Preserve that ordering for exact
compatibility (ascending frequency input is preferable for meaningful phase
continuity); explicit intensity/phase inputs simply sort complete rows. This
supersedes the initial frequency-order unwrapping choice above.

Unicode keyword refinement: Python normalizes identifier spelling `ϕ` to `φ`
(NFKC), while dictionary-expanded strings retain their original spelling. Accept
both spellings in the shared alias map and reject duplicates after resolving
them; verify direct Unicode keyword calls as well as dictionary aliases.

## Resident portable FFT integration (2026-09-09)

The next implementation connects the portable transforms to `CpuNativeSim`,
then exposes safe resident GNLSE construction and the existing native attempt
kernel through the standalone driver. This does not replace resident stepping
with a Python callback which happens to call a Rust RHS. The callback solver
remains the custom-model path. Basic capillary setup/responses follow this unit;
the full requested roadmap and release gates remain unchanged.

Introduce `amalthea::transforms` with immutable 1-D complex/real plan enums:
retained FFTW wrappers or RustFFT/RealFFT plans. Transforms receive explicit,
exclusive `FftScratch` references. Portable transforms preserve all inputs,
use standard unshifted spectrum ordering, and remain unnormalized in both
directions. Ignore imaginary DC/even-N Nyquist in real inverse transforms to
match FFTW. Allocate transform work buffers from plan requirements and reuse
across calls; no global selection switch, mutex in the transform loop, or
per-transform planner creation. Reuse this implementation from the private
Python transform classes, removing their duplicated algorithms.

`CpuNativeSim` owns ordinary, Raman-convolution, and Hilbert scratch. ModalRO
holds read-only plans; every ModalScratch owns transform/Hilbert scratch.
Parallel radial transforms use disjoint persistent scratch entries (one per
column), preserving the existing FFTW parallel behavior. Existing free-space
3-D FFTW wrappers and all CUDA code remain unchanged. FFTW construction remains
the Julia default and preserves its C ABI/wisdom semantics. Explicit portable
construction is handle-local and cannot invalidate an already configured
simulation's plan dependencies. Factory selection for Raman/Hilbert plans
uses that same handle policy; they must never attempt to load FFTW on Python.

First validate the transform seam independently (direct DFT, odd/even sizes,
input preservation, repeated use, concurrent shared plans with separate
scratch), and resident RHS/steps with portable plans. Then add a safe typed CPU
configuration interface and standalone native driver wiring. Transfer prepared
physical arrays, causal Raman samples, and accepted-step windows without
changing their formulas. Preserve existing dense-order, rejection, FSAL/resync,
stop, and output-cadence semantics through the shared driver lifecycle. Forced
native rejects configurations outside its implemented coverage; auto reports
its actual path. Once this GNLSE path passes, auto can select it.

Acceptance retains 1e-13 same-input RHS/interval and 1e-6 complete-trajectory
gates against independently prepared Julia fixtures, plus Python/native
comparisons and independently significant Kerr/Raman/shock/window effects.
Test both fourth- and fifth-order controls, rejection/restart/dense output,
modal scratch concurrency, and repeated construction. Run the full recorded
CPU gate and FFI checks because shared CPU FFT call sites change. Rebuild the
installed wheel from its sdist and run complete examples with networking
disabled and no Julia/FFTW/libcubature loaded. No performance gain is claimed
until complete-workload measurements exist.

The safe envelope facade accepts owned linop, complete output prefactor,
oversampled time window, base-grid accepted windows, Kerr coefficient (before
the resident 3/4 envelope factor), input amplitude scale, and optional causal
Raman samples plus dt. It validates dimensions/finiteness before invoking any
existing raw setter. The complete prefactor already includes sidx behavior;
transfer it using unit beta/window and active mask, preserving the identity
outside physical sidx. Factor the resident Raman setup into a sample-to-spectrum
helper shared by the existing coefficient setter and this facade. Native
filtering uses the base-grid portable plan and resynchronizes without refreshing
FSAL, matching the existing Julia lifecycle. The facade owns its CpuNativeSim;
no raw pointer or borrowed NumPy memory escapes to Python. A shared standalone
outer loop selects callback/native attempts and extra stages; dense polynomial
evaluation and output sampling remain one implementation for both paths.

## Gas material setup for capillaries (2026-09-09)

Implement the thermodynamic and gas optical foundation before constructing
capillary trajectories. Add a Python `materials` module with SI wavelength,
number-density, and susceptibility functions, pressure in bar and temperature
in kelvin. `density`, inverse `pressure`, gas `polarizability`, `refractive_index`,
and `gamma3` reproduce PhysData for its sixteen gas identifiers, including
He/HeJ/HeB and Ar/ArB aliases. Pin CoolProp 7.2.0 and call its Python PropsSI
DMOLAR/P interface. Multiply by the project's CODATA2014 Avogadro constant,
not the post-2019 SI value. Zero pressure/density returns exactly zero without
calling CoolProp. Validate names and finite nonnegative pressure/density and
positive temperatures/wavelengths; broadcast numeric arrays and return owned
results. Reference-density and scalar gamma3 caches are parameter-keyed, never
substitutes for evaluating requested pressure/temperature values.

Transcribe each gas's exact Sellmeier and reference-density convention from
`PhysData.jl`: Börzsönyi/JCT coefficients are divided by 1 bar/273.15 K density
before summation; Peck, Zhang and QuanfuHe use their documented atm/temperature
reference states. Preserve the implemented QuanfuHe expression (which returns
its refractive-index expression divided by density) rather than silently
converting it to n²−1. Its unusual convention needs direct oracle coverage and
is a separate physics issue if changed. Default gamma3 source selection follows
Lehmeier/Shelton/Zahedpour/Wahlstrand; Air has no default nonlinear source and
must fail explicitly. Do not invent Bishop or other unsupported models.

Export independent Julia fixtures for every gas over zero/ordinary/high
pressures and varied temperatures, wavelengths across the capillary optical
range, susceptibility/refractive index, default gamma3, and inverse pressure.
Assert 1e-13 setup equivalence, pressure round trips at the actual CoolProp
accuracy, physical pressure/temperature sensitivity, aliases and invalid inputs.
Record package version and CODATA constants in fixture metadata. These are
setup checks; they do not establish capillary propagation. The following work
still needs Marcatili full/reduced dispersion/loss/spatial fields/effective area
(including the existing silica lookup data), RealGrid pulse support and native
facade, capillary RHS normalization, ADK/PPT tables, and end-to-end oracle tests.
Full multimode/profile/custom coverage and all release gates remain required.

## Marcatili mode construction (2026-09-09)

Add `MarcatiliMode(radius, gas=None, pressure=0, ...)` for HE(n>=1), TE/TM(n=0),
positive integer radial order, rotation phi, full/reduced model and loss switch.
Methods expose neff, beta/alpha (and Unicode aliases), dispersion(order,omega),
field((r,theta)), normalization N, effective_area/Aeff, radius and dimlimits.
Frequency is rad/s, coordinates/radius are metres, z is metres. Field arrays
retain `(2, *coordinate_shape)` polarization ordering, with x then y. The raw
spatial field formula and normalization follow Capillary.jl; normalization is
not folded into field unless a caller explicitly requests it. Do not mask the
raw field outside its declared domain, since Julia does not do so.

Use SciPy Bessel functions/zeros and scalar quadrature for the radius-independent
Aeff integral. Enforce the requested scalar error estimate and verify independent
refinement, including high radial/azimuthal orders. The TE/TM integrand uses
J_(n-1)^4 as written in Julia (J_-1^4=J_1^4), while its numerator/normalization
uses J_2 at the J_1 zero. Cache only integer-mode-dependent constants. Radius
may be a callable and is evaluated at each requested z without sampling.
Pressure may similarly be callable for the built-in gas core. Custom core and
cladding index callbacks receive the complete omega array and keyword z; accept
a scalar constant or an exactly matching array, preserve exceptions, reject
nonfinite/wrong shapes, and copy incoming arrays so callback mutation cannot
change the caller's axis. Profiles must return finite scalars (positive radius,
nonnegative pressure); do not infer their constancy.

The default cladding is the existing complex SiO2 lookup, not its analytic
Sellmeier approximation. Bundle the literal three-column data from
`src/data/lookup_tables.jl`, retain CODATA2014 photon-energy conversion, and fit
real/imaginary cubic FITPACK splines (s=0, extrapolation enabled), matching
Maths.BSpline/Dierckx. Keep source provenance beside the data and verify table
and interpolated values, including irregular samples and nonzero absorption.

Port the existing Maths.derivative convention for orders 1–7: scaled frequency,
central stencil min(order+6,11), one adaptive bound estimate using p+2 points,
exact rational construction of coefficients, float64 evaluation, and the
same step-size/roundoff rules as FiniteDifferences.jl. Include its MIT license
for adapted algorithm code. This avoids silently selecting a different moving
frame. Test stencils/step estimates against Julia and independent analytic
functions. Derivative estimates are cancellation-sensitive; diagnose any
oracle discrepancy using exported samples and independent refinement before
changing acceptance. Do not widen the established same-input or trajectory
tolerances to hide a setup discrepancy.

Export independent Julia mode fixtures across gases, HE/TE/TM, full/reduced,
loss on/off, rotations, arbitrary radius/pressure profiles, custom complex
cladding, and near-cutoff behavior. Cover Bessel roots, silica data/index,
neff/beta/alpha, N/Aeff, fields and derivatives before capillary trajectories.
Refine Julia's Aeff quadrature for the 1e-13 setup gate while retaining the
unmodified default value as an independently bounded reference, as for SiO2
Raman. Validate N/Aeff independently by integration of spatial fields and check
radius/pressure/loss sensitivity. This is a setup unit: full capillary solves,
RealGrid inputs/driver, plasma and all later coverage remain required.

The independent default-constructor comparison exposed a Julia spline query
history defect. Repair and regression scope are documented in
[PLANS §21](PLANS.md#21-spline-finder-initialization-repair-discovered-by-the-python-mode-oracle).
Regenerate all setup/trajectory fixture sets against that repair, retaining
the preceding fixtures as provenance; do not adapt Python to the incorrect
history-dependent spline result.

For derivative refinement, independently evaluate the exponential's known
derivatives using 100-digit Decimal arithmetic and exact rational stencil
weights. Halve the stencil spacing and verify convergence to the analytic
derivative below 1e-13. Separately bound the production float64 derivative's
error by high-precision truncation error plus floating-point sample, weight,
coordinate and summation error. Require that bound to resolve the derivative
signal, so a zero or wrong-order derivative fails. Retain unchanged 1e-13
Julia identical-sample/coefficient/step gates and the physical beta1 gate.
This distinguishes the inherited finite-difference accuracy limit from a
porting error; it does not claim independently evaluated high-order mode
derivatives agree at 1e-13 or substitute those derivatives into propagation.

## Constant mode-averaged capillary envelope (2026-09-09)

The first capillary propagation unit adds `prop_capillary(radius, flength,
gas, pressure, ...)` for a single HE1m mode, constant radius/pressure, an
envelope without THG/plasma, Kerr, loss, and the existing rich pulse inputs.
Keep Julia's entrypoint positional order and keyword aliases. This is an
internal slice: carrier-resolved propagation, molecular Raman, gradients,
tapers, multimode/vector fields and custom runtime models remain subsequent
units and must be implemented before the public preview. Unsupported requests
raise explicitly; arbitrary callbacks must never be classified as constant
by sampling. Noise stays disabled. Native/auto use the resident CPU envelope;
Python uses the NumPy evaluator with the same Rust solver lifecycle.

Extract the existing envelope transforms/filter/solver adapter into a shared
private `_EnvelopeModel`, preserving GNLSE behavior and tests. Its physical
amplitude scale is configured per model: NLSCALE for GNLSE and
NLSCALE*sqrt(Aeff) for a capillary. Capillary polarization has coefficient
3/4*density*epsilon0*gamma3. On selected frequency bins the RHS prefactor is
`-i*omega^2/(4*NLSCALE*c)/beta*sqrt(Aeff)*omega_win`; leave the prefactor one
outside the selection, matching TransModeAvg. Prepare the constant operator
with beta1 at the reference frequency and beta0 subtraction for envelopes,
matching `LinearOps.make_const_linop` (not the variable-operator builder).
Preserve its power-loss clamp to [0,3000] per metre and zero non-selected
operator bins. No Taylor expansion
replaces the actual Marcatili frequency dependence.

Initially require `envelope=True`, scalar linear polarization and the
explicitly supported response set. Infer plasma=false for envelopes and
Raman=false for atomic gases as Julia does; molecular Raman requests raise
until implemented. `raman=False` allows the molecular gas's Kerr-only case.
Do not reject a supported configuration merely because it is slower; add the
Python path for each remaining Julia-valid combination in subsequent units.
The final public entrypoint retains Julia's carrier-resolved default.

Export independent Julia setup, RHS, single-interval, fixed/adaptive and
high-level trajectories for full/reduced and loss/Kerr controls. Compare raw
neff/beta/beta1, Aeff, density and initial energy at 1e-13. Frame subtraction
can amplify roundoff in the constant linear operator; use identical exported
linear-operator input for the 1e-13 interval gate and independently prepared
operators for the 1e-6 complete trajectory gate. Measure Kerr and loss effects
above the full-solve tolerance. Test both explicit backends, native callback
avoidance, aliases/invalid configurations, repeated construction and installed
examples before claiming the unit complete.

## Carrier-resolved pulse preparation (2026-09-10)

Extend the existing pulse objects and supplied-array path to RealGrid without
changing envelope conventions. Analytic real pulses use the cosine carrier,
then an rFFT; apply Taylor phase on the positive-frequency spectrum, invert
with irFFT, and normalize after phase. Energy and peak power use the analytic
signal intensity |Hilbert(E)|², not instantaneous real-field squared.
Use NumPy FFT construction of the analytic signal with DC/Nyquist retained
once and positive interior frequencies doubled; preserve input ownership.
Time integration retains the extended Simpson endpoint weights already used
by the envelope. Real spectral energy is 2*pi/omega_max² times the same
extended Simpson integral over the positive-frequency intensity.

Real time-domain array inputs must be real and exactly match grid.t;
frequency-domain arrays must match grid.omega. Follow irFFT's real endpoint
projection for complex DC/Nyquist, as the existing FFTW path does. Leave
frequency-domain DataPulse interpolation, masking, centering phase and custom
propagators intact, selecting the correct real spectral-energy normalization.
Custom input propagators still receive an owned complete spectral array and
isolated grid. This unit only prepares inputs; the resident RealGrid facade
and ADK/PPT propagation remain next.

Export independent Julia RealGrid pulse fixtures for Gaussian/sech, CEP/GDD,
power/energy, mixed pulses, nonuniform DataPulse data and custom propagators.
Compare axes, fields, analytic intensity and energies at 1e-13. Exercise
few-cycle pulses so the analytic-intensity convention matters beyond the
asserted tolerance. Add analytic Hilbert/DC/Nyquist tests, shape/type/exception
checks and an installed example. Retain all envelope and capillary regressions.

## Resident RealGrid and constant carrier capillaries (2026-09-10)

Generalize the safe resident envelope facade to a mode-averaged facade with an
explicit real/envelope representation. Keep one attempt/dense/filter lifecycle
and the existing Julia C ABI. Real setup validates Nomega=Nt/2+1, even base and
oversampled time lengths, exact owned array lengths and finite coefficients.
Use the existing portable real mode-averaged RHS and exclusive reusable real
filter scratch. Accepted filtering applies the spectral window, normalized
irFFT, time window and rFFT, retaining Julia's FSAL resynchronization rule.
Real Raman/plasma configuration is added in subsequent response units; reject
those requests explicitly until implemented.

Extend the constant scalar capillary entrypoint with RealGrid using the same
mode/material/pulse setup. Its linear operator is -i*(beta-beta1*omega)-alpha/2,
with the existing 3000/m loss clamp and mask. Prefactor and physical amplitude
scale are the same as the envelope; Kerr's real coefficient is rho*epsilon0*
gamma3 without the envelope's 3/4 factor. Real transforms pad the positive
spectrum with Nt_over/Nt scaling and crop with its inverse. Preserve Julia's
default carrier plasma selection: during this internal Kerr-only unit require
explicit plasma=False rather than silently disabling plasma. THG defaults on.
For thg=False use the Julia-valid analytic-intensity Kerr response in Python,
3/4*rho*epsilon0*gamma3*|Hilbert(E)|^2*E. Auto reports this Python path; forced
native rejects it while the resident kernel only implements E^3. This is a
coverage assignment, not a claim that all carrier responses are finished.
Results retain frequency-first ordering and reconstruct real fields via irFFT.

Independent Julia fixtures cover THG on/off, Kerr/loss controls, full/reduced
mode, HE12, fine sampling and fourth-order compatibility. Compare setup/RHS
and identical-input dense intervals at 1e-13, independently prepared complete
fixed/adaptive/high-level trajectories at 1e-6. Require nonzero Kerr, THG and
loss effects above the asserted trajectory tolerance. Add real resident
rejection/restart/filter tests and independent analytic dense convergence,
configuration validation, native callback avoidance and installed examples.
Run the shared CPU gate and preserve all envelope and pulse regressions.
No CUDA execution is authorized for this work.

## Standalone ADK setup (2026-09-10)

Port PhysData's first ionisation potentials and IonRateADK's closed-form setup
to Python, preserving CODATA2014 constants and arithmetic conventions. Expose
`materials.ionisation_potential(material, unit="SI"|"eV"|"atomic")` and
`IonRateADK(material_or_ionpot, occupancy=2, threshold=True,
cycle_average=False)`. Numeric ionisation potentials are joules. Preserve gas
aliases and the atomic-H potential; Air has no default and must fail explicitly.
Rate calls accept real scalar/array electric fields in V/m, use absolute field,
return owned finite nonnegative rates in 1/s, and reject invalid/nonfinite inputs.
Keep threshold discovery's sequential 1.01 field increments from 1e3 V/m and
default occupancy/cycle-average convention rather than fitting a table.

At exactly zero field return the physical zero limit even with threshold=False;
Julia's direct threshold-free formula produces Inf*0 there. Record that edge
case separately rather than claiming NaN parity. For positive fields retain
the production formula and its floating-point underflow convention. Stable
logarithmic evaluation is only a numerical control initially, not a silent
replacement that changes the threshold. Validate finite setup coefficients
before returning a constructed object. Plasma wiring remains a subsequent
unit alongside PPT, with the model/default selection from Interface.makeplasma!.

Export independent Julia potentials, coefficients, thresholds and signed field
rate sweeps for every supported potential, both threshold modes, cycle average
and occupancies. Compare coefficients at 1e-13 and resolved rates pointwise at
1e-13; record subnormal/underflow samples separately with an absolute bound
derived from floating-point spacing rather than normalizing by tiny rates.
Add independent high-precision positive-field formula checks, occupancy and
cycle-average sensitivity, zero/invalid/array/ownership tests, and installed
CPU-only execution. Setup evidence does not establish plasma trajectories.

## Standalone PPT rates and local tables (2026-09-11)

Port IonRatePPT and its material setup, including quantum numbers, static
neutral/ion polarisabilities, Stark shift, dipole correction, m-level sums,
occupancy (integer or serial callable), cycle average, sum_integral, sum_tol
and optional Cnl. Numeric setup takes ionisation potential [J], wavelength [m],
Z and integer l; material setup uses the exact PhysData values and missing-data
convention (missing corrections become zero). Evaluate finite real scalar or
array fields with owned output; use zero at exactly zero field and explicit
errors for invalid configurations/results. Do not replace unsupported material
quantum numbers with guessed values. Keep ADK as Interface's default for
H2/D2/N2O/CH4/SF6 and PPT for other supported plasma gases when wiring follows.

Preserve the multiphoton series start ceil(v), additive iteration order and
Maths.converge_series criterion 2*abs(new-old)/abs(new+old)<sum_tol. Array
evaluation may advance active fields together, but each field keeps its own
stopping decision. Guard against nonconvergence explicitly. For m=0 use
Dawson's integral. For nonzero m use Kummer's transformation to evaluate the
scaled hypergeometric expression without overflowing exp(x*x):
phi = sqrt(pi)*x**(2*abs(m)+1)*Gamma(abs(m)+1)/(2*Gamma(abs(m)+3/2))
      * 1F1(abs(m)+1, abs(m)+3/2, -x*x).
Use SciPy's vectorized hyp1f1 on the validated moderate range and an isolated
mpmath context with increasing precision for the large-argument branch and
nonfinite fallback; require successive precision results to agree before
rounding to float64. The branch must be tested against independent scaled
integral quadrature and precision refinement, including either side of x=26.
Keep numerical changes separate from Julia's unmodified rate comparison and
diagnose any inherited oracle error rather than relaxing tolerances.
References: [SciPy hyp1f1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html),
[mpmath hypergeometric functions](https://mpmath.org/doc/current/functions/hypergeometric.html).

Construct the acceleration table locally on Julia's linear Emax/5000..Emax
grid (default N=65536, Emax=2*barrier_suppression). Remove zero-rate nodes,
interpolate log(rate) with Maths.CSpline's normalized-knot derivative convention, return
zero below the retained minimum and clamp above the maximum. Cache numeric
arrays as pickle-free NPZ with a versioned SHA-256 key over every physical,
model, grid and numerical option plus constants/convention version. Use atomic
replacement of uniquely named temporary files, validate loaded metadata and
arrays, and recompute corrupt/incompatible entries. Arbitrary occupancy
callbacks require cache=False rather than an unstable callable identity key.
Tests use explicit temporary cache directories; no Julia provisioning, runtime
download or shared writable global numerical context is allowed.

Export independent Julia setup, phi values, direct rates, table samples and
interpolated queries. Cover all supported PPT materials, numeric/custom setup,
all option branches and threshold/clamp/zero behavior. Compare resolved setup
and same-input evaluations at 1e-13, retain measured underflow limits, and
validate table interpolation by independent refinement. Prove Stark/dipole,
cycle average and m-level options change the reference beyond their asserted
tolerance. Include actual default-size local table construction, cache hits,
parameter isolation, corruption/replacement, callable errors and installed
offline construction. Plasma trajectory acceptance remains a separate gate.


PPT zero-field oracle finding (2026-09-11): direct Julia IonRatePPT(0) enters
an unbounded multiphoton series because its gamma/alpha/beta intermediates
become nonfinite and NaN never satisfies converge_series' stopping criterion.
The independent exporter must not call that undefined branch: mark zero as
unevaluated (NaN sentinel plus explicit metadata), preserve positive/signed
resolved-field comparisons, and separately test Python's physical zero limit.
This changes neither finite-field oracle tolerances nor Julia's existing table
path, which already returns zero below Emin. Record the interrupted export as
diagnostic evidence; do not mistake it for an oracle acceptance run.

PPT spline finding (2026-09-11): IonRatePPTAccel uses Maths.CSpline, unlike the
FITPACK-backed Maths.BSpline used by pulse data. Its derivative system has
diagonal [2,4,...,4,2], off-diagonals 1, and RHS three times adjacent endpoint
differences / centered interior differences. Solve that tridiagonal system,
then evaluate the same cubic in local normalized coordinate t, returning
literal log samples at exact knots. This is the natural cubic for uniform
knots but deliberately retains Julia's normalized-knot convention for supplied
nonuniform samples too. A FITPACK not-a-knot spline differs near the table
ends (measured 1.249e-6 relative rate error) and is not acceptable. Version the
cache convention accordingly and include nonuniform supplied-table checks.

## Constant carrier plasma through Python evaluation (2026-09-11)

Extend the validated constant scalar HE1m RealGrid capillary slice with ADK
and PPT plasma using the existing Rust callback stepper. Keep the resident
native plasma handoff as a separate next unit: auto reports Python plasma
execution and forced native rejects plasma before expensive setup. No shared
Rust/Julia/CUDA implementation or C ABI change is required for this slice.

Accept plasma=False, True/None (Julia's carrier default), 'ADK', 'PPT', or
an existing IonRateADK/IonRatePPT/IonRatePPTAccel object. Default gases
H2/D2/N2O/CH4/SF6 use ADK; other gases use PPT with exact existing material
eligibility. PPT_options (ASCII alias ppt_options) is a mapping passed to local
PPT table construction, including cache controls; reject nonempty options when
PPT construction is not selected rather than silently ignore them. Explicit
rate objects use their ionisation potential for loss; supplied-sample tables
without a model use the gas's potential. Keep envelope plasma unsupported,
quantum noise disabled, and molecular Raman/profile/modal restrictions explicit.

At each RHS, evaluate the scalar physical oversampled field and compute three
cumulative trapezoids with initial zero and Julia's operation ordering:
F = preionfrac + 1 - exp(-cumtrapz(W,dt)); phase = F*(e^2/m_e)*E;
J = cumtrapz(phase,dt) + Ip*W*(1-F)/E for nonzero E;
P = cumtrapz(J,dt). Add density*P to Kerr polarization before time window/FFT.
Preserve the inherited preionfrac convention exactly, including fractions above
one; do not clip or replace it with a different depletion model. Validate
preionfrac in [0,1], finite real input/rates/results, and exact field-array
shapes. Output arrays are owned. Do not retain a rate callback's returned array
or mutate its input. Callback evaluation is serial within a simulation.
The internal plasma evaluator accepts complete-array rate callables to support
later custom-response wiring; public capillary selection in this unit accepts
the three validated rate model types only.

Export independent Julia input, linop, RHS, plasma rate/fraction/current/P,
same-input dense intervals and full fixed/adaptive trajectories. Use non-vacuous
ADK/PPT controls with plasma off, Kerr off, THG off, preionisation, a finer grid
and fourth-order stepping; require 1e-13 same-input and 1e-6 independently
prepared complete solves. Compare default and explicit model choice, built-in
versus constructed rate objects, cache reuse, metadata/NPZ, original callback
exceptions and invalid arrays. Add an installed offline plasma example. Keep
shared CPU acceptance evidence unchanged for this Python-only unit and append
new numerical/installed evidence before declaring this slice complete.

Plasma intermediate conditioning finding (2026-09-11): for the symmetric
physical control, the final polarization norm is 8.4552e-30 while current norm
is 1.9931e-14. Exp/library rounding produces a 4.8334e-16 relative fraction
difference, an 8.8324e-15 current difference, and a 2.7629e-12 polarization
difference through cancellation. Feeding Julia's fractions into the current
formula or its currents into the last trapezoid gives exact agreement. Keep
these independent component discrepancies visible. Validate each integral at
1e-13 with identical intermediate inputs; bound the independently propagated
polarization difference by the integral of absolute current perturbations plus
a floating-point summation bound. Require this bound below 1e-6 of the actual
polarization signal to exclude vacuous agreement. Also compare the last
integral with 100-digit Decimal arithmetic. Retain the unchanged 1e-13 full RHS
and same-input dense interval gates and 1e-6 complete-trajectory gates. This
isolates inherited conditioning without modifying the physics or discarding
the symmetric test control.

Rejected-plasma-trial oracle (2026-09-11): use identical controls in Julia and
Python for the large initial-step test. Comparing rtol=1e-12/max_dt=length with
the ordinary rtol=1e-9/max_dt=1e-5 fixture changes accepted-window cadence and
gives a 1.116e-5 field difference; a finer adaptive solve is not a valid
reference for this per-accepted-step filtered problem. Export the same large
initial step and tight controls, assert actual rejections and retain 1e-6
trajectory equivalence. Report accepted/rejected counts for both paths.

## Owned resident ADK/PPT plasma handoff (2026-09-11)

Extend `resident.rs::ModeAverageConfig` with optional validated `PlasmaConfig`.
Own the selected boxed `AdkIonizationRate` or `PptIonizationRate` inside
ResidentModeAverage, after the CpuNativeSim field in declaration/drop order.
Call the existing CPU setters only after mode-average geometry is configured.
The boxes keep borrowed engine pointers stable when the facade moves; no
Python-owned storage, transient Vec pointers, or external handles are retained.
The resident wrapper is CPU-only and does not call rate_vector's GPU dispatch.
No existing Julia C ABI, shared ionisation formula or CUDA implementation is
changed; the strict CUDA gate remains blocked by the documented host pair.

ADK transfers its seven already validated Python coefficients, requiring
finite positive coefficients and threshold > 0. Threshold-free ADK retains
Python evaluation: the old shared scalar formula gives NaN at exactly zero,
where Python defines the physical zero limit. Direct IonRatePPT and subclasses
with custom behavior also retain Python evaluation; forced native rejects with
the specific reason. Do not silently replace a direct formula with a LUT.

PPT transfers positive increasing field nodes, finite log rates and normalized
knot derivatives from the validated Python table. Build owned SplineSegments
using b=d0/h, c=(3*(y1-y0)-2*d0-d1)/h^2 and
 d=(2*(y0-y1)+d0+d1)/h^3, preserving Maths.CSpline's nonuniform convention.
Include the final literal log-rate endpoint. Do not call the existing Rust
from_samples constructor, whose variable-knot natural-spline convention is
different. Validate every coefficient and bound before engine construction.
If scaling the polynomial cannot be represented finitely, auto keeps the
Python normalized-coordinate evaluator and native rejects; no node sampling
or tolerance change is allowed. Finite trial fields above Emax retain clamp
behavior. Bound node counts and reject mismatched/nonfinite/private inputs.

Transfer ionpot, e_ratio, preionfrac, dt and density as finite values with
positive potentials/time/e_ratio, nonnegative density and preionfrac in [0,1].
The eight-element existing mode-average tuple stays unchanged. Private
solve_real/real_rhs gain an optional trailing plasma tuple:
(kind, data, log_rates, derivatives, ionpot, e_ratio, preionfrac, density).
For ADK, data holds the seven coefficients and both other arrays are empty;
for PPT, data holds field nodes. dt is the existing mode-average time spacing.
Envelope bindings continue with no plasma parameter. Python uses native for
eligible constant scalar carrier plasma with THG-on Kerr (or Kerr disabled);
THG-off Kerr keeps the existing Python fallback. Report actual backend and
selection reason. Explicit backend=python always retains callback evaluation.

Acceptance extends all eight independent plasma fixture cases to both paths
where eligible, retaining 1e-13 RHS/dense and 1e-6 complete/rejected trajectories.
Assert that native solves avoid Python response and filter callbacks. Cover
cycle-average/occupancy ADK, supplied nonuniform PPT tables, threshold/clamp/
zero behavior, original configuration ownership, moved facades and repeated
construction/destruction. Reject invalid lengths, enum tags, coefficients,
preionisation and density. Preserve Python fallback/custom-class semantics.
Run the recorded full CPU/FFI gate, the complete twelve-fixture Python suite,
and extracted-sdist/offline installed examples; do not repeat the known failed
CUDA gate without a compatible toolchain. Append measured evidence before
closing this native slice; full modal/molecular/custom coverage remains next.

## Molecular Raman setup and density broadening (2026-09-11)

Add `MolecularRaman(time, gas, *, rotation=True, vibration=True,
minJ=0, maxJ=50, temperature=ROOMTEMP)` as owned Python setup, independently
of capillary wiring. Port the six complete molecular parameter sets in
PhysData (N2/H2/D2/N2O/CH4/SF6) with CODATA2014 constants, preserving all literal
frequencies, nuclear weights, polarizabilities, reduced masses and linewidths.
O2 has missing rotation/vibration lifetimes in the oracle: explicitly reject
selected missing components; both components disabled remains an empty response.
No guessed parameters. Atomic gases have no molecular Raman model.

Construct rotational energies 2*pi*hbar*c*(B*J*(J+1)-D*(J*(J+1))^2), truncate
at the first decreasing energy as Julia does, then normalize nuclear-weighted
Boltzmann populations. Preserve its starting-level truncation/guard convention.
Build J->J+2 frequencies and couplings in the original arithmetic order,
retaining rotational and vibrational groups separately for response summation.
Vibrational K=(4*pi*epsilon0)^2*(dalpha/dQ)^2/(4*mu*Omega). Do not apply an
additional n2/epsilon0 normalization or multiply density during setup.

Evaluate on tt=arange(N)*(time[1]-time[0]); zero causal origin, and apply the
same Planck tail taper between .7*tt[-1] and tt[-1]. Refresh lifetimes at each
supplied density, using fixed tau or 1/(pi*(C+A/(rho/amg)+B*rho/amg)). Preserve
rotational rho=0 => tau=Inf. Selected density-dependent vibrational branches
at rho=0 produce a nonfinite origin in Julia; reject that undefined case
explicitly rather than claiming a finite oracle comparison. Do not cache an
arbitrary density profile or silently sample it. Return owned response and
flattened oscillator arrays, and expose copied parameter/group data for setup
validation and later resident/Python response wiring.

Validate finite real uniform time axes, positive temperature, scalar finite
nonnegative density, boolean component flags and integer J limits. Export
independent Julia literal parameters/constants, group oscillator frequencies,
couplings, damping at multiple densities/temperatures, and complete response
samples/tapers. Retain 1e-13 same-input setup/response gates, explicit O2/zero
oracle failures, component toggles and density sensitivity above tolerance.
Use a 100-digit independent Boltzmann/coupling control for an analytic rotor
case. Run the full installed suite and source-rebuilt/offline setup example;
capillary Raman trajectories and native acceleration are separate next units.
The current resident-plasma CPU gate may continue because this adds Python
setup/export/tests only, without modifying its shared Rust/Julia engine.

## Constant molecular capillary Raman trajectories (2026-09-11)

Wire MolecularRaman into constant scalar HE1m capillaries for all six complete
molecular gas models, with Julia's default Raman selection and rotation,
vibration and temperature controls. Keep zero-density/missing-data failures
from the setup gate explicit. At constant density construct h=rho*response(rho)
and its double-length causal FFT once; retain the Planck tail window and the
same physical field normalization. Do not add a silica/GNLSE response scale.

Envelope drive is .5*abs(E)^2. Carrier drive is E^2 with THG on or
.5*abs(analytic_signal(E))^2 with THG off. Raman polarization is E times the
first Nt samples of the zero-padded convolution, including dt exactly once.
Add it after Kerr/plasma and before the polarization time window. Make the
Python envelope evaluator's polarization calculation an overridable internal
method so the capillary can preserve the existing envelope Raman path while
adding Julia's supported envelope THG Kerr term when requested:
rho*epsilon0*gamma3/4 * exp(2i*omega0*to)*E^3, on top of the normal .75 term.
Envelope THG uses Python when Kerr is enabled. Pass its flag to EnvGrid even
when Kerr is disabled: Julia's grid selection increases the oversampling for
THG independently of response selection. Omitting that flag gives 1024 rather
than 2048 response samples in the independent N2 THG fixture. Without Kerr the
THG polarization term is zero, but the finer grid still applies.
Envelope plasma and vector Raman remain explicitly unsupported.

For native carrier THG-on Raman, reuse CpuNativeSim's portable double-length
Raman FFT scratch and samples, adding a CPU mode-average real convolution
branch alongside its retained ADE response. Extract the existing envelope
convolution execution into a shared internal method without changing its
normalization or summation order. Real drive is E^2; final accumulation uses
the real physical field. The safe facade accepts real Raman samples; the
existing private tuple remains unchanged and denotes THG-on for carrier
Raman. THG-off carrier Raman retains Python (even with Kerr disabled), with a
specific forced-native rejection. The Julia C ABI, retained Julia ADE path
and CUDA implementation are unchanged. This new branch is reached only when
real FFT Raman samples have been configured; existing real setups leave the
flag false. Validate the shared CPU change with the full recorded gate.

Export independent Julia setup/RHS, impulse samples, identical-input dense
intervals and complete fixed/adaptive/high-level trajectories. Cover each gas
on both grids, component toggles, Raman-off controls, real THG-off and envelope
THG, Kerr-off Raman, temperature, fourth-order stepping, finer sampling and
ADK/PPT+Raman mixtures. Use both native/Python paths where eligible. Retain
1e-13 setup/RHS/dense and 1e-6 complete solve gates; show Raman/plasma/component/
THG effects exceed 1e-5. Independently check the portable real convolution
against direct causal summation on an artificial nontrivial field, including
tail padding and accepted windows. Require native callback avoidance, fallback
reporting, owned output/NPZ and installed offline complete molecular examples.
No frozen performance audit changes; this is scientific coverage work.

## Position-dependent callback linear operator (2026-09-11)

The next dependency for exact pressure/radius profiles is the low-level
`solve_precon` driver. Accept `linop(z)` alongside the current constant array;
it returns a complete complex array with exactly the original field shape.
Keep the callable alive for the solve, evaluate serially, copy returned values,
validate shape/finiteness and preserve original exceptions, including failures
inside stage, FSAL-frame and dense-output propagation. Do not infer constancy,
sample a replacement function, or cache more than the immediately preceding
linear-operator position, matching Julia's exact `lastt2` equality cache.

Source inspection of `RK45.jl::make_prop!` establishes the oracle convention:
multiply by exp(L(t2)*(t2-t1)); its inverse uses the same L(t2) and opposite
interval sign. Despite the local name `linop_int`, this is endpoint evaluation,
not quadrature of L over the interval. Reproduce this convention explicitly.
Do not introduce a higher-order variable-L integrator during frontend migration.
For variable L, the DOPRI nonlinear order is not a claim of fifth-order global
accuracy of the complete variable-coefficient ODE; demonstrate its endpoint
linear error through independent analytic refinement and document the limit.

Extend only the Python extension's serial Context and public Python adapter;
reuse the corrected FFI stepping kernel and its existing error boundaries.
Keep constant-array private bindings backward compatible by adding an optional
trailing linear callback argument. Native resident solves continue to use the
constant array and unchanged facade. RHS forward/back transforms must select
the same requested stage position, even for the backward exponent. Preserve
all current rejection, deferred FSAL, dense sampling and accepted-filter seams.
Do not change the shared engine or Julia ABI while the Raman CPU gate runs.

Export independent Julia variable-operator trajectories and callback positions
for fourth/fifth-order fixed and adaptive solves, nonzero restarts, nonlinear
rejections and nontrivial filters. Compare same-input intervals/fixed nodes at
1e-13 and complete adaptive trajectories at 1e-6, with variable-vs-constant
and nonlinear controls greater than the asserted tolerance. Use matrix fields
to test Fortran flattening. Verify a constant callable matches a literal array,
exact duplicate-position caching, owned retained outputs, invalid shapes and
nonfinite results, original exceptions at construction/stages/dense output,
serial callbacks and repeated teardown. Add an installed/offline variable-L
example with an independent piecewise endpoint formula and refinement check.
This unit enables profile integration but does not yet broaden prop_capillary;
profile/density/broadening/normalization wiring needs its own numerical gate.

Oracle inspection during validation: Julia returns the left endpoint directly
and recomputes quintic extra stages for every interior output query. For a
variable linear callback, retain that call sequence (including return visits
to the extra-stage position) so callback positions agree exactly for fixed
steps. Constant-array/resident paths retain their existing extra-stage reuse.
The tight fifth-order adaptive fixture with an accepted multiplicative filter
hits Julia's default step repetition limit: its retained FSAL derivative was
evaluated before filtering. Preserve this as an explicit matching failure,
with fourth-order adaptive and both fixed-order filtered trajectories as
positive controls. Do not weaken tolerances or alter FSAL semantics to hide it.

## Exact scalar capillary profiles and gradients (2026-09-11)

Extend scalar HE1m capillaries on both grids to callable radius/pressure,
two-point pressure `(p0,p1)` and multipoint `(positions,pressures)` data. Use
the validated Python variable-operator driver; auto reports Python profile
evaluation and forced native rejects it. Preserve constant scalar native
eligibility. Do not infer constant functions from probes or replace arbitrary
functions with sampled tables. Call functions at actual requested positions,
including dense stages and the accepted endpoint beyond the requested length;
only explicit gradient data are clamped outside their specified endpoints.
Reject invalid scalar results, nonfinite/nonpositive radii, negative pressures,
wrong gradient dimensions or unsorted/duplicate positions. Copy profile data
and serialize descriptive callable metadata without retaining callback objects
in output parameters.

Preserve `Capillary.gradient`: pressure interpolates as
sqrt(p0^2 + fraction*(p1^2-p0^2)); density uses `PhysData.densityspline`'s
1024 uniformly spaced pressure nodes and normalized-knot Maths.CSpline. The
sample bounds are min/max supplied pressures, except equal pressures start
at zero. All-zero gradient nodes give an invalid oracle density spline and
must raise explicitly (ordinary constant vacuum remains supported). Build
these thermodynamic samples locally with CoolProp 7.2.0. This is the existing
material interpolation convention, not sampling the user profile in position.
Arbitrary pressure callables instead use direct thermodynamics at each call.
Extract the already matched normalized spline algorithm from PPT into a
private reusable helper without changing coefficients, endpoint values, PPT
cache conventions or native handoff derivatives; retain all PPT gates.

For varying pressure, construct a Marcatili core callback from the same density
provider used by nonlinear responses, preserving the polarizability/index
conventions. At every RHS position recompute effective area, input amplitude
normalization, beta-dependent spectral prefactor, Kerr density, plasma density
and molecular Raman response/damping plus its causal FFT. Reuse static grid,
pulse, cladding, ionization-table and molecular oscillator setup. Linear
evaluation independently recomputes neff, beta1 and (for envelope) beta0 at
its requested position. It must not leave nonlinear coefficients stale through
the linear driver's position cache.

Match the generic Julia variable `LinearOps.make_linop`: conjugate the effective
index, floor its real part at 1e-3 and clamp its imaginary part to
3000*c/omega before multiplying by omega/c. This caps amplitude attenuation
at 3000/m, while the existing constant path clamps power alpha at 3000/m and
uses alpha/2. Retain this observable convention rather than unifying the two.
`Interface.setup` passes the default thg=false to linear construction even
when envelope THG polarization is selected; retain its beta0 reference.
Subtract the envelope beta0 term after forming the first phase difference,
in Julia's arithmetic order, to preserve the same-input cancellation gate.

Export independently prepared Julia setup at several off-grid positions,
same-input RHS/dense intervals, fixed and adaptive complete trajectories,
and high-level entrypoints for structured gradients/tapers. For arbitrary
pressure callables, use Julia's valid low-level setup with the same direct
thermodynamic density/core functions. Cover rising/falling/multipoint pressure,
nonlinear radius, simultaneous profiles, full/reduced modes, loss clamping,
both grids and THG choices, Raman component/temperature controls, ADK/PPT+Raman,
fourth-order controls and zero-density limits where the oracle is defined.
Retain 1e-13 setup/RHS/same-input dense and 1e-6 full-solve gates, with profile,
Kerr, plasma and Raman controls changing Julia by more than 1e-5. Check
gradient knots/density spline against identical-data Julia evaluation and an
independently refined thermodynamic grid. Any cancellation in the assembled
linear phase requires same-input isolation, not weaker field tolerances.
Test constant-callable equivalence where the two loss-clamp conventions agree,
actual stage-position evaluation, callback exceptions/invalid results, output
ownership and repeated destruction. Add installed/offline examples and the
full installed/source-rebuilt suite. This unit changes Python setup/evaluation
only; the completed shared CPU/FFI gate remains applicable.

The export subsequently exposed a shared Julia metadata conversion defect for
carrier N2O/CH4/SF6 gradients. Repair/validation is specified in [PLANS §22](PLANS.md#22-real-valued-complex-gas-coefficient-in-gradient-metadata);
this exception requires a new full CPU/FFI gate before accepting this unit.


## Scalar gas and response mixtures (2026-09-11)

The next scalar coverage unit accepts a nonempty sequence of gas names in the
existing gas positional argument and a matching sequence of partial-pressure
specifications. Preserve the single-gas interpretation of `(p0,p1)`; with a gas
sequence the outer pressure axis is always species, so gradients are nested.
Each species accepts constant, structured-gradient or callable pressure, with
the exact profile conventions above. The common radius may be a callable.
Copy arrays, validate species/pressure counts and reject empty or malformed
collections explicitly. A one-species sequence and duplicate species are valid.

Use the existing low-level Julia mixture convention as the oracle:
`PhysData.ref_index_fun(gases, pressures)` sums species susceptibilities before
`sqrt(1 + chi1)`, while `NonlinearRHS.Et_to_Pt!` visits each species' response
tuple with that species' density. The high-level Julia `makeresponse` does not
supply this collection interface; independently construct the low-level mode,
density vector and response tuples, following `test/test_mixtures.jl` and
`examples/low_level_interface/mixtures/mixture_modeAvg.jl`. Do not average
refractive indices or treat total partial pressure as a pure gas. Density is
computed independently with CoolProp for each supplied partial pressure.

Expose optional `species_options`, a sequence of mappings matching the gas
axis. Global response options supply defaults; each mapping may override
`kerr`, `raman`, `plasma`, `rotation`, `vibration`, `PPT_options` and
`preionfrac`. Resolve default Raman and ADK/PPT selection separately for every
gas. Temperature, grid, THG and propagation controls remain common. Reject
unknown per-species keys and mismatched counts. Existing single-gas controls
retain their current meanings, including envelope-plasma rejection and
explicit undefined molecular responses. A custom ionisation model may be
supplied through the already supported plasma option. General nonlinear
array callbacks remain the subsequent custom-response protocol unit.

Factor scalar response construction and evaluation so one geometry transform
synthesizes the physical time field, each species evaluates that complete field,
and their polarizations accumulate in the documented species/response order.
Refresh each species' density, Kerr coefficient and Raman damping at every RHS
position. Plasma fractions, ionisation loss and preionisation are evaluated
separately for each species using its own ionisation model/potential. Never
replace multiple plasma responses with an effective rate or average potential.
Linear dispersion, beta-dependent normalization and area use the combined
mixture mode. Keep a scalar total Kerr coefficient only for the native handoff;
constant Kerr-only mixtures may use the resident scalar engine because this
sum is algebraically identical. Raman/plasma mixtures initially use Python,
even where a narrower algebraic reduction might be possible. Native forcing
rejects the entire unsupported combination before invoking custom profiles.
No native C ABI or CUDA changes are required.

For acceptance export both grids with distinct gases and density-equivalent
split-gas controls, constant and varying partial pressures, common tapers,
full/reduced loss, fourth-order and supported THG choices. Include molecular
rotation/vibration mixtures and carrier mixtures combining PPT/ADK, Raman,
Kerr and nonzero preionisation. Compare the independently prepared core index,
densities, mode area/dispersion, complete RHS and identical-input dense interval
at 1e-13; preserve the identical-linear-input isolation for phase cancellation.
Compare full fixed/adaptive trajectories at 1e-6, with equivalent split gas
and single-species controls, and require species/response/profile effects above
1e-5. Constant eligible cases must run through both Python/native paths and
prove native callback avoidance. Verify original profile exceptions, fresh
per-species density evaluation, malformed input rejection and owned metadata.
Add one complete installed/offline example plus the full installed and
source-rebuilt Python gates. Reuse the completed shared CPU gate when this unit
changes only Python setup/evaluation; rerun it if shared engine code changes.

The mixed PPT/ADK + Raman + multipoint gradient/taper stress case needs a
separate adaptive refinement gate. Its 1e-5 m maximum-step trajectories differ
by 2.50e-6 even though fixed trajectories agree at 3.44e-12 and same-input RHS/
dense checks pass at 1e-13. Recorded accepted/rejected counts differ between
backends; removing accepted filtering alone does not resolve the discrepancy.
The already documented endpoint-exponential variable-L scheme depends on the
accepted step positions, so a local nonlinear rtol is not a global variable-L
accuracy guarantee. Preserve the coarse result as diagnostic evidence. Refine
both backends at 2.5e-6 and 1e-6 m, keep the full-solve comparison gate at 1e-6,
and use the latter bound for this stress case's adaptive acceptance. Export
actual step counts/positions and the ordinary default-control trajectory too.
Do not change response physics or solver/FSAL semantics to force equal adaptive
paths, and do not present the coarse run as meeting the field tolerance. This
is backend-equivalence refinement, not a claim that either trajectory has
1e-6 absolute error against the continuously varying physical equation.

## Modal geometry, quadrature and propagation (2026-09-11)

Implement the remaining modal path around the existing Rust array callback
solver. Preserve frequency/mode/save axes, real scalar spatial mode functions,
one or two Cartesian field components, the full polar/Cartesian integral and
the reduced radial integral. The synthesis matrix contains
`field(mode, coordinates; z)/sqrt(N(mode; z))`, with mode rows and selected x/y
columns. Synthesize the physical field with a matrix product, apply the full
time-array response, and project with the ordinary transpose of that real
matrix. Retain Julia's interior-domain boundary rules, polar Jacobian r (or
2*pi*r for the reduced integral), temporal/spectral windows, oversampling and
modal factor -i*omega/4. Do not reuse mode-averaged area/beta normalization.

Use SciPy `cubature` (available since 1.15; raise the dependency floor) with
workers=1 and bounded callback batches. Its component criterion is additive;
Cubature.L2 instead requires norm(error) <= max(atol, rtol*norm(value)), over
all real and imaginary field entries. Evaluate that global criterion explicitly
using scaled Euclidean norms. A zero-subdivision pilot estimates the global
scale; if necessary rerun SciPy with a common component absolute budget smaller
than the global budget divided by sqrt(component count). Recheck the actual
returned global estimate/error and refine again if the scale moved. Bound
evaluations and reject nonconvergence rather than returning unchecked fields.
Form relative budgets by multiplying the largest real/imaginary component by
rtol before restoring the scaled norm. A finite array's total norm can overflow
even when its relative error budget is representable; an infinite budget must
never admit an otherwise failing integral. Reject unrepresentable final error
metrics explicitly and test both accepted/rejected extreme finite arrays.
Use public SciPy integration only; do not introduce a native quadrature engine
or depend on private SciPy internals. Gk21 is the default, with Genz-Malik as an
independent full-integral refinement option. Evidence must compare independent
spatial refinement, not just SciPy's success flag.
Sources: [SciPy cubature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html),
[SciPy 1.15 release notes](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html),
[Cubature convergence rule](https://github.com/stevengj/cubature/blob/master/converged.h).

The reusable mode protocol supplies `neff(omega, *, z)`,
`field((coordinate1, coordinate2), *, z)`, `dimlimits(*, z)`, and optionally
`N(*, z)`, `beta(omega, *, z)`, `alpha(omega, *, z)` and
`dispersion(order, omega, *, z)`. Provide a `Mode` base with generic numerical
normalization and dispersion defaults; Marcatili retains its analytic methods.
Raw spatial fields return x/y on axis zero, including batched coordinate arrays.
Copy/validate callback outputs and reject nonzero imaginary spatial fields,
nonfinite values, invalid domains and nonpositive normalization. Complex neff
remains valid. Compute normalization at each requested position and reuse it
only within that evaluation; no process-global cache of custom mode objects.
The full integral uses the common domain; reject mismatched mode domains rather
than silently clipping one mode with another's boundary. The lower-level
spatial helper supports explicit x, y or xy selection and full/reduced rules.
Custom modes default to the full integral; a caller may explicitly declare
radial symmetry. Full custom-model propagation must be tested independently
against an equivalent built-in mode, not inferred from geometry tests alone.

Public capillary selection retains a lone HE1m string as mode average. A
positive mode count denotes HE11...HE1m modal propagation; a sequence of
signifiers/dictionaries or constructed modes denotes a modal collection,
including length one. Dictionaries disambiguate indices above nine. HE
polarization pairs use phi=0 then pi/(2*n); TE/TM have one orientation.
Constructed modes are already specified and must not be silently duplicated.
The public modal entrypoint checks power orthonormality at setup, as Julia's
`Interface.check_orth` does; generic spatial synthesis remains usable for
nonorthogonal test fields. Include the numerical integration error in this
check and reject duplicate/nonorthogonal public mode collections. Explicit
component/full-integral controls allow custom scalar or vector fields without
inferring symmetry from a few sampled points.
Like Julia, perform the orthonormal-set check only for multiple modes. Include
a 32-epsilon rounding allowance in addition to quadrature error: independent
HE11...HE18 power integrals show defects up to 2e-15 while embedded-rule error
estimates can be smaller. This guards normalization/matrix arithmetic rounding;
it does not alter the 1e-13 node/step or 1e-6 trajectory comparison gates.
Expose pulse mode/polarisation metadata and preserve Julia's observable input
conventions, including its historical x/y selector ordering. Circular and
elliptical fields use the documented relative phase and energy split, with
the existing carrier/envelope normalization performed separately per component.
Allow complete grid-matched modal pulse arrays with frequency/mode axes.

Evaluate independent per-species responses on each complete physical time-field
array. Scalar modal responses reuse the factored scalar implementation. Vector
Kerr uses Julia's cross-polarization formulas; envelope THG retains Julia's
componentwise expression. Carrier vector plasma uses hypot(Ex,Ey) for rates
and fractions, then componentwise current and ionization loss E/|E|^2. Reject
vector Raman and envelope plasma as excluded physics; preserve any additional
combination that the independent Julia oracle rejects as an explicit error.
Refresh density/broadening and spatial normalization at actual RHS positions.
Custom nonlinear callbacks receive owned complete time-by-polarization arrays
plus position/coordinates and density context, and return the same shape; retain
original exceptions and execute serially. General scalar/GNLSE custom response
wiring follows the same shape/ownership contract.

Constant modal linear operators preserve Julia's all-frequency outside-window
values and common reference-mode frame; variable operators retain the selected
frequency mask and endpoint-exponential driver conventions. Explicitly compare
these different paths instead of unifying their loss/phase behavior. Automatic
selection reports Python quadrature and the point evaluator actually used.
Add safe vectorized Rust point evaluation for eligible existing kernels after
the independent Python mathematical path is verified; do not label Python
quadrature as fully native. Forced native requires the entire evaluation path
to satisfy native eligibility, and rejects the initial SciPy modal path.

Acceptance covers field matrices/normalization/domains, identical-input node
polarization and projection/dense intervals at 1e-13, independently refined
spatial integrals, and full trajectories at 1e-6. Include HE/TE/TM, both grids,
scalar/vector fields, polarization pairs, excited higher modes and transfer,
full/reduced integrals, response mixtures, Raman/plasma, profiles, fourth-order
and default adaptive controls, and all custom protocols. Every physics control
must change Julia by more than 1e-5. Geometry/quadrature acceptance is an internal
dependency and does not establish complete modal propagation. Run installed
and source-rebuilt suites plus offline complete examples. Shared Rust/Julia
changes require the recorded CPU/FFI gate; Python-only units retain its current
evidence. Do not retry the unchanged incompatible CUDA toolchain for this work.

### Modal frontend wiring details

Expose `radial_integral_rtol` (Julia's existing name), `modal_atol`,
`modal_maxevals`, `modal_rule`, `modal_full` and `modal_components`; defaults
are 1e-3, 0, 100000, gk21 and automatic geometry/component selection. A mode
count or sequence always selects modal output, even for one mode. A lone
ordinary signifier retains mode-average behavior; extend that existing path
to the other valid HE/TE/TM signifiers. Orthogonal polarization requests turn
a signifier into its modal pair. Constructed mode collections are evaluated
at requested positions using the variable linear callback; never infer a
custom mode is constant from probes. Explicit controls select reduced/scalar
custom geometry. Preserve the reference frame of the first mode.

Pulse objects gain mode (default `lowest`) and polarisation (default `linear`)
metadata, with ASCII spelling aliases. For each selected component construct
its spectral phase and amplitude before invoking its input propagator; retain
the original callback object and serial execution. Zero ellipticity creates a
zero-energy component without normalizing a zero field. Grid-matched modal
arrays have time/frequency first and modes second; optional energy/peak-power
normalization uses total modal intensity, preserving relative components.
A propagator for such an array receives the complete owned modal array.
Reject conflicting array/duration or pulse-collection specifications and wrong
mode counts, and keep result parameters JSON-compatible without retaining mode
or callback objects. Each modal RHS refreshes species coefficients once, then
evaluates complete time-by-polarization arrays at each spatial node serially.
The current unit wires built-in scalar/vector responses and constructed custom
modes; general nonlinear user-array callbacks remain required by the following
custom-response unit. They are not claimed from the spatial callback helper.
The independent response probe confirms carrier vector Kerr with thg=false
fails Julia's scalar-planned Hilbert scratch with a BoundsError. Reject that
combination explicitly while retaining scalar modal THG-off and vector
plasma-only THG-off. Envelope THG is valid for vector fields and retains its
componentwise expression rather than the ordinary cross-polarization Kerr law.
Constant envelope modal operators evaluate loss even outside the physical
window, including negative grid frequencies. Preserve Julia's signed silica
spline extrapolation there; gas susceptibility is even in wavelength. Factor
the existing Marcatili index algebra to accept already-prepared indices for
this internal assembly, while retaining positive-frequency validation on the
public material/mode APIs. Test the whole linear array, not just its spectral
window. Constructed custom modes use the variable masked operator instead.

Modal default-control tests expose an overly restrictive standalone driver
guard: Julia accepts a positive initial dt outside min_dt/max_dt, then applies
those bounds when selecting subsequent attempts. Remove only the two initial
dt-versus-bound checks in the Python extension; retain finite positive controls,
ordered bounds and step-progress validation. Cover both sides of the bounds,
both solution orders, callback/resident paths and independent Julia output.
Custom/built-in equivalence tests must select the same variable-operator path
(a callable constant radius suffices). The deliberately different constant
outside-window operator can produce measurable differences for broadband inputs;
test each separately against its Julia counterpart, without changing the rules.
Scalar pulse collections validate mode selection against the single capillary
mode (GNLSE uses only `lowest`) and permit only linear/x selection; polarization
requiring two components must use modal propagation. Explicit pulse collections
own their polarization, as in Julia, even when a top-level default differs.
Vector plasma retains the scalar plasma acceptance method for cancellation in
the final cumulative current integral: compare rates/fractions/current and
same-current integration at 1e-13, bound propagated current differences plus
summation rounding analytically, and independently refine selected integrals
with 100-digit arithmetic. Export per-species point components. Verify FFT and
projection from identical polarization inputs at 1e-13 and keep fixed-node
dense and complete-solve gates unchanged. A conditioned point polarization
bound must itself remain below 1e-6; a bare relaxed tolerance is insufficient.
The envelope THG effect control must hold the grid and initial field fixed:
Julia expands the envelope time window when THG is enabled. Export a paired
THG-disabled response on the enabled grid, with identical linear operator,
initial field and solver controls. Comparing different grids would conflate
setup changes with the nonlinear feature under test.
Near-zero plasma fractions also exhibit cancellation in Julia's `1-exp(-I)`:
the gradient fixture has a same-field relative fraction difference of 6.12e-11
from an absolute rounding error near machine epsilon. Preserve that convention.
Check rates and integrated rates from identical input at 1e-13; bound fraction
error by integrated-rate differences, positive-sum rounding and four epsilons
for the exponential/subtraction. Bound the propagated current error from its
phase integral and ionization-loss terms, then the final polarization as above.
Same-fraction current and same-current polarization retain their 1e-13 gates.
Strict refined plasma spatial comparisons may use the oracle's two-million
node budget; public defaults remain unchanged and exhaustion remains an error.
The refined PPT case exposes unnecessary subdivision from SciPy's component
criterion: Julia satisfies global L2 in 32767 nodes while the first conservative
SciPy component budget exhausts 100000 nodes without returning an estimate for
the global check. Request bounded subdivision passes (4, 16, 64, ...), checking
the required global norm after each public cubature result, including results
whose component criterion has not converged. Keep the conservative component
budget, count every restarted evaluation against the shared node limit, and
cap eight passes. Acceptance remains solely the unchanged global L2 test;
this avoids treating SciPy's stronger per-component criterion as mandatory.
Installed full-solve comparisons exercise the public 1e-3 spatial default
against the independently refined 1e-8 Julia trajectories, retaining the 1e-6
field gate. Setup RHS checks still request 1e-8; focused full-envelope,
Raman and full-plasma cases additionally compare complete default/refined
Python trajectories. Retain the expensive 1e-8 PPT complete solve as diagnostic
evidence. This separates default-use acceptance from quadrature refinement
without changing either field or fixed-node mathematical tolerances.
The 500-uJ N2/H2 modal plasma stress probe requires initial steps below 0.4 um
and estimates hours for a 10-um adaptive solve at rtol=1e-9. Preserve its point
and fixed-trajectory data as stress diagnostics. Use 100 uJ for the combined
mixture acceptance fixture, retaining the grid, geometry, species, response
choices, solver/spatial tolerances and all mathematical gates. Export a
plasma-disabled trajectory with identical initial field to prove the plasma
effect exceeds 1e-5. This is an explicitly recorded fixture-amplitude choice,
not a claim of completed high-energy stress acceptance.


## General nonlinear response callbacks (next coverage unit)

This builds on the implemented modal frontend. Modal artifact validation may
continue against its frozen source/wheels while this independent Python unit is
prepared; do not promote either unit until its gates pass. Add `responses=None` to both high-level
entrypoints. A callable or sequence of callables appends custom polarization
responses to the configured built-ins; an empty sequence has no effect. Users
replace a built-in by disabling that response explicitly. Validate the entire
sequence before entering the solver. Nonempty custom responses make the whole
simulation use Python evaluation; forced native rejects before invoking any
custom callback. Rust continues to own stepping and callback exception transport.

Each callable has signature `response(field, context)` and returns polarization
with the same shape. `field` is an owned complete oversampled time-by-component
array, `(ntime, 1)` for scalar/GNLSE and `(ntime, 2)` for vector modes. Carrier
fields/polarizations are real; envelope fields/polarizations may be complex.
Reject wrong shapes, nonzero imaginary carrier polarization, nonfinite values
and nonnumeric arrays before accumulation. Copy the result so later mutation of
user-retained arrays cannot change solver state. Never deepcopy callback objects
or invoke callbacks concurrently within one simulation. Preserve original
exceptions through the existing Rust callback transport.

Expose a lightweight `ResponseContext` containing the actual `z`, an owned time
axis in seconds, grid kind, gas names and an owned array of current number
densities. GNLSE has empty gas/density arrays. Modal nodes additionally provide
coordinate-system and spatial-coordinate values; scalar averages/GNLSE use
`None`. Context arrays and fields belong to that invocation, so user mutation
cannot corrupt the grid, another response, quadrature state or saved output.
Snapshot already-refreshed density coefficients; do not call arbitrary pressure
profiles a second time merely to create context. Modal points receive the
actual RHS position and coordinates, including rejected trials and the final
step beyond the requested end.

The returned array is physical nonlinear polarization before the time window,
FFT and existing spectral normalization. There is no implicit extra density or
energy scaling. Built-ins and callbacks accumulate on the same physical field;
preserve the established GNLSE 4/3 convention, capillary envelope 3/4 scaling,
and modal projection. Document an equivalent Kerr callback using these exact
conventions and a nonlocal-in-time example to establish complete-array access.
Result metadata reports custom response evaluation/call counts and the actual
Python execution path. Parameters serialize callback names only, retaining no
callback/context/mode objects in the result.

Acceptance uses equivalent built-in/custom Kerr, Raman-like convolution and
selected plasma kernels on both grids, scalar/vector modal fields and mixtures
with changing density/radius. Compare identical-input RHS/fixed-node dense
checks at 1e-13 and independently prepared Julia/custom complete solves at 1e-6,
with feature effects above 1e-5. Include a Cartesian custom mode in complete
propagation, and combine custom mode, pressure/radius and response protocols.
Check serial ordering, complete array shapes, mutation isolation, original
exceptions, malformed/nonfinite results, nested simulation construction and
repeated teardown without retained callback references. Run installed and
source-rebuilt suites plus a complete offline custom-response example. No
native eligibility is inferred from callback identity or a few sampled values.
Native modal point batching remains a separate following implementation unit.

The independent Cartesian callback fixture uses two orthogonal polynomial modes
on a tapering rectangle, generic numerical power normalization in Python and
analytic normalization in Julia. Both spatial components carry energy. Their
custom index varies at the requested position; Ar/Ne partial-pressure callables
refresh density directly. Compare the built-in Julia Kerr response with a
complete-array Python Kerr callback on both grids, including fixed-node dense
intervals with transferred linear samples. Linear-only and constant-profile
Julia controls hold the input fixed and establish nonzero effects. This fixture
is synthetic protocol coverage, not a new physical waveguide model. Reuse the
existing modal ADK/preionisation/gradient oracles for a custom plasma callback
written with SciPy cumulative integration; same-input built-in/callback checks
isolate callback transport from the previously bounded cancellation.

The Cartesian assembled RHS gate refines Julia's spatial estimate from 1e-9
to 1e-11 before the 1e-13 comparison: the first 1e-9 integral differed from
Python's polynomial-exact tensor rule by 1.30e-13. Retain coarse/refined arrays
and both reported error norms to establish convergence. Complete trajectories
continue to use 1e-9 spatial integration; fixed-node dense and field tolerances
are unchanged.

## Safe native modal point interface — implementation design

After custom-response numerical acceptance, add a bounded reusable point
interface without moving adaptive spatial integration into Rust. Python still
constructs physical modal spectra and projects the resulting polarization;
SciPy still enforces the global integration criterion and Rust owns stepping.
This is the native point-evaluation assignment in the capability matrix above,
not a claim that the complete modal evaluator is native.

The first eligible slice is constant-profile built-in modes with ordinary
Kerr on RealGrid/EnvGrid and scalar sampled Raman where the resident scalar
kernel already supports it. Plasma, arbitrary profiles/constructed modes,
custom nonlinear responses, carrier no-THG Kerr/Raman and envelope THG retain
Python point evaluation. Vector Raman remains explicitly unsupported. Select
native points only for `backend="auto"`; `backend="python"` keeps the complete
independent Python evaluator. Forced `backend="native"` still rejects SciPy
modal integration before invoking callbacks. Report `backend="python"`,
`point_evaluator="rust"`, the portable FFT choice, and the selection reason.
Never infer eligibility from sampled callable values or callback identity.

Expose a private PyO3 point object with owned validated configuration, reusable
FFT plans/scratch, and a batch method accepting complete physical spectra.
The Python boundary retains `(point, frequency, component)` order; explicitly
transpose/copy into the engine's component-major scratch and restore that order
on return. Check grid dimensions, one/two components, every configuration/input
length, finite values, checked products and allocation limits before indexing.
Input and returned arrays are owned. No pointers borrowed from NumPy, Julia,
FFTW or libcubature may be retained. Use CPU-only portable construction and
preserve every existing Julia C export.

Reuse `ResidentModeAverage` for scalar physical point spectra: unit amplitude
scale, modal spectral prefactor `-i*omega*omega_window/4`, physical Kerr
coefficient and optional already-density-scaled impulse. Initialization already
computes the RHS; expose a safe batch facade around that path instead of
recreating FFT plans for each node. Keep batch outputs independent of previous
initializations. This uses the existing carrier/envelope sampled-Raman kernels
and preserves the established instantaneous 3/4 factor.

For vector Kerr, extract the existing temporal subsection of
`CpuNativeSim::modal_pointcalc` into a helper with an explicit read-only temporal
configuration and exclusive `ModalScratch`. The original Julia modal path calls
that helper between its unchanged mode synthesis and projection. A portable
standalone facade supplies physical spectra directly to the same helper. Do not
copy the vector Kerr formulas into a second implementation. Preserve transform
normalization/cropping/windowing and arithmetic order during extraction. Start
with serial native batches; the existing per-worker scratch architecture remains
available for separately measured threading work. The point object owns its
scratch and must not admit concurrent mutable calls.

Validate direct point arrays against independently exported Julia modal nodes
and explicit Python spectra at 1e-13 for both grids and scalar/vector fields,
including zero input and repeated calls with changing arrays. Validate full
RHS, fixed-node dense intervals and complete trajectories against the existing
modal oracles at the unchanged gates; enforce nonzero feature controls. Add
boundary/lifetime tests and prove automatic dispatch bypasses Python temporal
responses while forced Python calls them. Extend all eligible oracle cases to
both point implementations, including sampled scalar Raman and constant Kerr
mixtures. Run the full recorded shared CPU/FFI gate because vector extraction
changes the existing CPU modal kernel. CUDA implementation and ABI remain
unchanged; any later shared CUDA change still requires strict hardware tests.

Record matched setup/warmed complete-solve timings, steps, quadrature nodes and
point-boundary overhead using a new post-repair evidence directory. Preserve
the frozen CPU audit. Retain automatic acceleration only with a demonstrated
5% complete-solve improvement or removal of a measured regression. The broader
import/first-solve/output/memory/platform performance snapshot and bottleneck
selection remain subsequent roadmap work. If conversion costs prevent that
improvement, measure and resolve the boundary cost before promoting dispatch.

Boundary details: infer the oversampled length from the time window; validate
an even time length, oversampled length at least the base time length, and the
matching real/envelope spectral count before constructing plans. Retain the
existing 2^23-sample construction limit and cap each copied batch at 2^24
complex values using checked size multiplication. A zero-point batch is valid
only with empty input. The private binding converts point/frequency/component
NumPy arrays to point/component/frequency owned flat storage and back. The
PyO3 object remains bound to its creating thread and keeps exclusive mutable
scratch; no GIL-release/threading behavior is introduced in this slice.

The initial list boundary fails the performance gate (first complete-solve
samples regress 41% for scalar envelopes and 27% for vectors). Retain that
measurement and replace per-element Python object conversion before promotion.
Use rust-numpy 0.27.1 with the existing PyO3 0.27.2: a contiguous owned NumPy
input is borrowed only for the synchronous GIL-held call, and `PyArray::from_vec`
transfers ownership of the result allocation. The Python wrapper still makes
its own component-major input copy and returns an owned array. No borrowed
array or buffer pointer survives the call; constructor metadata lists are setup
only. Check the versioned [read-only array contract](https://docs.rs/numpy/0.27.1/numpy/borrow/struct.PyReadonlyArray.html)
and the crate source before wiring. Reject noncontiguous/wrong-dtype direct
private-binding inputs explicitly and test read-only input, retention and
output lifetime. These are boundary changes only; temporal physics and the
shared CPU extraction stay unchanged.

## Optional HDF5 and result processing — implementation design

Add `PropagationResult.save_hdf5(path)` using the optional `hdf5` extra
(`h5py>=3.11`). Import h5py only when this method is called, with an actionable
missing-extra error. Base installation, simulation and NPZ output remain
independent of HDF5. Preserve numeric field axes exactly: frequency/save for
mode averages, frequency/mode/save for modal results. Keep native complex128
datasets rather than splitting or converting complex fields.

Use the same named payload in NPZ and HDF5: existing `Eomega`, `z`, `t`,
`omega`, `to`, `omega_over`, `twin`, `towin`, `omega_win`, `sidx`, `parameters`
and `metadata`; add `format_version=1` and JSON `grid` describing grid type,
reference wavelength/frequency, wavelength limits, propagation/time ranges
and field-axis labels. Existing NPZ keys retain their meaning. Store strings
as UTF-8 scalar HDF5 datasets and plain Unicode NPZ arrays. Serialize nested
NumPy arrays/scalars to JSON without pickle, and reject unsupported objects
or nonfinite JSON numbers before opening either destination. Preserve existing
parameter/callback descriptions; no callback is serialized or invoked by output.
Use h5py's portable, lossless gzip compression for numeric arrays; scalar/string
datasets are uncompressed. These are completed-result files, not streaming
Julia `Output` objects, checkpoints or serialized simulation constructors.
Output methods follow existing NPZ overwrite semantics and propagate I/O errors.

Keep result serialization in a small Python-only helper shared by both methods.
Do not alter the stepper, physical field normalization, or introduce a new
energy/spectrum convention in this unit. Add a complete processing example
running both APIs, saving/reading both formats, checking exact data and metadata,
reconstructing the time field with the grid-appropriate inverse FFT, computing
pulse energy with the established `energy_t` convention and printing execution
diagnostics. Use a temporary output directory so offline example runs leave no
files in the installed package.

Validate independent NumPy/h5py readers, bitwise numeric equality, modal axes,
carrier/envelope reconstruction, nested JSON/Unicode metadata, repeated saves
and invalid metadata without truncating an existing file. Test absent h5py in
a fresh process while import/simulation/NPZ still succeed. Exercise real GNLSE,
scalar carrier and custom-response modal results. The I/O gate needs exact
round trips rather than a new trajectory tolerance; existing physics gates
remain applicable. Build fresh checkout and source-rebuilt wheels with the
optional extra, check packaging metadata, run focused installed output tests
and the processing example offline. Keep the already-running 901-test native
point artifacts frozen and record the new output artifact separately.

References checked before implementation: h5py's official
[dataset API](https://docs.h5py.org/en/stable/high/dataset.html) for complex
arrays/compression and [string API](https://docs.h5py.org/en/stable/strings.html)
for scalar UTF-8 datasets and `asstr()` reading.

## Linux wheel and Python-version acceptance — implementation design

The next installation unit builds the completed output/callback/point source
for actual CPython 3.11, 3.12, 3.13 and 3.14 on Linux x86_64. Keep the frozen
native-point suites running; do not replace their installed environments.
Use a fresh artifact directory and explicit portable `RUSTFLAGS=''`, CPU-only
build policy and `--manylinux 2_28 --auditwheel check`. Audit dependencies and
record interpreter/tool versions plus wheel/sdist hashes. Preserve the Rust
lockfile and build source-rebuilt wheels from a checked extracted sdist.

The local Docker CLI exists but the daemon requires unavailable group/sudo
access. Use maturin's documented `--zig` manylinux build option instead of
changing host permissions. Install build-only tools in a temporary environment.
Use existing 3.12/3.13/3.14 interpreters, and obtain a separate 3.11 development
interpreter through uv's python-build-standalone installer under a temporary
install/cache/bin directory. This download prepares validation infrastructure;
the installed package and offline examples never invoke uv or download Python.
Do not replace system Python or add executables to the user's normal PATH.
Record the exact downloaded interpreter/version before building.

For each interpreter, install binary dependencies and the actual wheel into
fresh environments outside the checkout. Verify the optional HDF5 extra and
`pip check`, inspect module paths and linked libraries, and run the complete
source-rebuilt suite with all twenty independent Julia fixtures supplied.
Exercise checkout wheels with the numerical solver/output smoke and all
seventeen examples; compare their packaged source to the frozen sdist.
Run complete examples with network access disabled and an empty toolchain PATH
after installation. Require no missing-oracle skips. No compiler/Julia/system
FFTW/libcubature is part of wheel installation or runtime requirements.
This establishes x86_64 Python-version evidence; Linux ARM64, Apple Silicon,
Windows and actual older-glibc host validation remain distinct platform gates.
Do not claim those platforms from cross-compilation or a successful wheel tag.
Ignore local in-place `_native` shared-library build products in git; source
distributions must continue excluding all compiled artifacts and build trees.

Build reference: [maturin distribution guidance](https://www.maturin.rs/distribution.html).
Development interpreter reference: [uv Python installation](https://docs.astral.sh/uv/guides/install-python/).

## Hosted wheel matrix — next delivery unit design

Prepare maintained tooling and CI from the locally exercised build/install
sequence while the frozen numerical artifacts finish. Hosted acceptance still
requires resolving those gates. Keep publication separate
from artifact preparation; workflow success alone does not authorize a release
or establish distribution-name availability.

Split standalone CI into one independent Julia-oracle producer and native
platform wheel consumers. The producer exports all twenty fixture families
from the checked-out repaired Julia source and uploads them with a provenance
manifest (commit, source/exporter hashes, Julia/CoolProp versions). Preserve
coarse/refined diagnostic fixtures and fail on incomplete export. Consumers
download the same fixture artifact and set every `AMALTHEA_*_ORACLE` path
explicitly. No Julia installation is needed in a wheel-test job.

Use native runners for Linux x86_64/ARM64, Apple Silicon and Windows x86_64,
each with CPython 3.11–3.14. Build checkout and extracted-sdist wheels using
portable CPU-only flags. Linux uses the verified maturin/Zig manylinux_2_28
command and independent auditwheel check. Apple Silicon sets an explicit
macOS 11.0 deployment baseline; Windows uses native MSVC x86_64. Do not treat
cross-build success as runtime evidence. Inspect and retain binary dependency
audits and package/source hashes. Source archives must include engine sources,
tests, data and licenses, and exclude build trees and compiled products.

The maintained artifact runner should share fixture-name mapping, log/provenance
format and example inventory rather than duplicating those in shell steps.
Fresh environments install only binary runtime dependencies and the actual
wheel, with the HDF5 extra exercised. Run checkout solver/output smoke, then
the complete source-rebuilt suite, accepting no missing-fixture skips. Retain
structured pytest results, complete stdout, versions, checksums and artifacts
on failure as well as success. Existing Julia CPU/FFI jobs remain independent.

Run all seventeen complete examples outside the checkout after installation,
with an empty toolchain PATH and enforced network denial. Linux uses a network
namespace as already validated locally. The macOS launcher uses a process
sandbox denying network access; the Windows launcher uses temporary firewall
rules scoped to that environment's Python executable, removing only those
rules in a finally block. If the required isolation mechanism is unavailable,
fail the platform's offline gate explicitly; do not label a normal networked
run offline. These controls apply only to ephemeral CI runners, not user hosts.
Record installed module paths and loaded-library checks appropriate to each OS.

Collect the prepared Apple Silicon diagnostic separately during platform
validation using `test/performance_audit/run_apple_quick_test.py`, retaining
its existing frozen-baseline rules. Hosted timings cannot substitute for the
controlled post-repair benchmark snapshot. Public Linux preview waits for
complete scientific and Linux platform evidence; stable waits for all four
platforms. No GPU Python support or standing GPU CI is added by this workflow.

The fixture producer writes its completion manifest only after all exporter
processes succeed, with hashes for every output file and normalized-LF source
hashes for the Julia source/data, Rust source and exporters. Consumers verify
the commit, source hashes and every fixture file before testing. LF normalization
is explicit so Windows git checkout conventions do not create false provenance
failures. Keep producer logs and environment/version records alongside fixtures.
The artifact runner treats any pytest failure, missing test collection or skip
as incomplete acceptance, using JUnit XML rather than only a successful exit or
a parsed console phrase. Prototype/local environments may reuse previously
built artifacts, but must run the same source/hash and installed-path checks.

Implement the manifest producer/consumer contract first in the existing Linux
job, before replacing that job with the platform matrix. Force every Rust
offload toggle off in the producer, clear partial-export selectors, and retain
all exporter logs when one fails. Test stale commits, changed source/data,
missing/extra fixtures, failed exports and JUnit collection/skip failures with
synthetic artifacts; synthetic data cannot count as physics evidence. Exercise
the process wrapper separately with a real Julia exporter. This intermediate
CI unit changes reference transport and acceptance, not the numerical formulas
or the already-running frozen wheel suites.

The wheel runner has explicit build and test stages with a reusable JSON
manifest. Build records exact checkout/sdist package hashes, native host/target,
interpreter, commands and binary audits. It rejects archive traversal, links,
compiled products and missing engine/test/data/license files. Test verifies the
recorded revision, host/interpreter, source and artifact hashes before creating
fresh environments outside the checkout; a build-only result is never labeled
tested. Each artifact runs all examples offline and checks installed paths and
loaded libraries. Checkout gets the solver/output smoke; the source-rebuilt
wheel gets the complete suite with no skips.

Linux isolates the example process in a network namespace. macOS uses an
explicit deny-network process profile and verifies that a connection attempt
is denied. Windows firewall rules cover both the venv redirector and the base
Python executable, because Windows venvs can launch the latter. Require an
explicit ephemeral-hosted-CI flag and GitHub-hosted runner environment before
making those Windows rules; verify firewall profiles/rules and remove exactly
the generated names in `finally`. Never change a host's global firewall policy.
Retain failure records if isolation cannot be established. Library inspection
uses Linux process maps, macOS dyld image names and Windows process-module
enumeration. These are development checks, without new runtime dependencies.

The maintained build audit found that maturin's checkout wheel can copy local
Python bytecode caches even though the sdist excludes them. Explicitly exclude
`__pycache__`, `.pyc` and `.pyo` from wheel and source packaging in pyproject;
keep exact package inventory validation so stale or interpreter-specific files
cannot be silently accepted. Preserve the failed artifact and rebuild from the
corrected packaging configuration; do not weaken the inventory comparison.

Invoke isolated test/example interpreters with explicit UTF-8 mode. Julia
fixture metadata and retained logs use UTF-8, including Unicode aliases; their
decoding must not depend on a Windows host's legacy text code page.

## Linux glibc 2.28 runtime — implementation design

Close the actual minimum-glibc runtime gap with a temporary rootless container.
The host's Docker daemon is unavailable; bubblewrap succeeds with normal host
namespace access and does not require system installation or policy changes.
Use an immutable official debuerreotype Debian buster slim root filesystem,
verify its Git blob identity and record SHA-256/package manifest provenance.
Extract only into a new temporary directory, converting absolute symlink targets
to equivalent in-tree relative targets before safe extraction. Bind that tree
read-only as the container root. Keep package installs, outputs and temporary
files in dedicated writable directories; expose no checkout, host toolchain,
Julia depot, host libraries or network to the test process.

Use development-only managed CPython 3.11–3.14 interpreters compatible with the
older userspace. Record their downloaded versions and executable hashes. Build
fresh CPU-only manylinux_2_28 checkout/source wheels through the maintained
runner, or use its existing final artifact where the exact package/source
hashes match. Inside the container verify `gnu_get_libc_version()=="2.28"` and
loaded libc paths, then create fresh environments and install the actual wheel
plus HDF5/tests from a predownloaded binary wheelhouse with `--no-index`.
After installation clear the toolchain PATH and run all seventeen examples,
installed-path/library checks and the complete source-rebuilt numerical suite
with the verified twenty-family reference artifact. Require no skipped tests.
Keep this evidence separate from host-version runs and from controlled timings;
the host kernel/CPU remain modern, so this proves the older glibc userspace
boundary rather than historical kernel/hardware support. The helper lives under
`test/standalone_wheels/`, leaving the already-running package/tool snapshots
unchanged. A rootless environment setup failure must remain a failed/incomplete
gate, never a substituted host run.

Run installation/offline smoke independently while the complete reference
artifact is being generated. Its result is explicitly `smoke_passed`; it does
not claim numerical-suite acceptance. The subsequent full test uses a new
output directory and fresh environments with the completed reference manifest.

## Post-repair Python performance snapshot — implementation design

Create a separate `test/python_performance/` harness; leave all frozen audit
files and results untouched. Compare installed CPU wheels (`auto`, `python`)
with the current Julia frontend using pure Julia and its existing resident Rust
engine, in separate processes. A Julia request for Rust must report the actual
stepper. Unsupported native configurations are explicitly unavailable rather
than relabeled Rust timings. Keep package/engine source unchanged while the
installed artifact gates run.

Use independently constructed, matched public GNLSE and capillary cases:
GNLSE Kerr and SDO Raman; carrier/envelope capillary Kerr; scalar ADK plasma;
molecular Raman; pressure gradient; multimode Kerr and vector Kerr. Add an
equivalent complete-array custom GNLSE Kerr callback to quantify callback cost.
Record exact physical inputs, solver tolerances, save cadence, mode counts,
temporal sizes, accepted/rejected steps and actual backend. Quantum noise is
disabled. Reuse production setup and solve/window/output routines; do not time
a hand-written replacement solver. Separate frontend setup from the complete
solve (including plan construction, accepted filtering and dense result output).
The matched complete-workload cost is setup plus solve. Also record process
startup/package import, the first public simulation, owned result copying,
optional HDF5 writing and process peak RSS. File-output formats/metadata differ
between frontends and must be labeled, not used as interchangeable cost claims.

Gate each timed fixture on independently prepared initial fields, grid axes,
linear operators and nonlinear feature controls, plus strict same-input RHS or
existing fixed-node evidence and full fields at 1e-6. Modal quadrature may use
different converged node sets; retain its established global-error criterion and
independent refinement evidence. Report adaptive step-count differences rather
than forcing or hiding them. The custom callback comparison must agree with
built-in Kerr at 1e-13 on the same input and change the linear oracle by more
than 1e-5. Use fresh models for each solve to avoid cached output/counters.

Capture source/artifact hashes, Python/Julia/Rust/FFT/dependency versions, CPU,
affinity, governor/turbo and thread environment. Fix BLAS/OMP/FFTW/Rayon/Julia to
one thread and select one physical CPU. Persistent backend processes perform
two unmeasured warmups followed by randomized round-robin samples. Require at
least ten samples, relative MAD <=3% and bootstrap 95% CI half-width <=5%; extend
to thirty or label unstable. Cold metrics are separately labeled single-process
observations until repeated. Do development smoke/correctness checks while
installation suites run, but collect accepted performance measurements only
after competing heavy jobs finish. Preserve raw records and failed attempts.
The first implementation unit establishes the harness and matched fixtures;
additions extend the same snapshot without altering its provenance. Only then
rank complete-workload bottlenecks and consider a >=5% optimization.

The first actual matched GNLSE SDO check exposes the existing resident Julia
path's ADE-versus-FFT discretization difference: about 9.87e-5 at the public
coarse temporal grid, while both standalone Python paths match Julia FFT
convolution near 1e-14. Preserve that workload and its failed Rust comparison.
Mark an explicitly identified old-Rust method mismatch inadmissible for timing
rankings; continue the Python/pure-Julia comparisons. Add paired finer temporal
grids, retain the measured convergence and require the same 1e-6 gate before
admitting the refined four-way comparison. Do not change production formulas,
relax tolerance, or describe excluded old-Rust timings as an accepted speedup.

Render a standalone Markdown report from the retained snapshot, recomputing
statistics from raw samples. Admit speedups only for completed, stable,
correctness-admitted backend pairs; report paired-ratio bootstrap intervals.
Smoke/failed/unstable snapshots may produce diagnostic tables but no accepted
speedups or bottleneck ranking. Include excluded comparisons, exact inputs,
source/artifact provenance, counts, cold metric boundaries and distinct HDF5
payloads. Keep report-generation tests separate from scientific acceptance.

## Apple diagnostic during platform validation — implementation design

Run the already prepared `test/performance_audit/run_apple_quick_test.py` on a
native `macos-15` runner in a separate CPU-only job. Preserve that runner and
the frozen audit unchanged. Supply a new runner-temp output path, retain its
console/JSON/Markdown on success or failure, and record revision/source/runtime
provenance in a small external wrapper under `test/standalone_wheels/`.
The wrapper rejects non-Apple execution and dry-run results. The existing quick
runner can return process success with false correctness flags, so explicitly
require all three levers, both 1/2/4-thread series, finite positive timings,
cross-build/thread errors <=1e-6 and exact modal/scan topology flags before
labeling the evidence passed. Runtime failures remain failed, without synthetic
hardware results. A shared hosted VM provides Apple execution evidence, not
controlled hardware-wide speedup claims or automatic LTO promotion.

Prepare this follow-up locally while the first hosted migration run proceeds;
batch its delivery with the next required platform correction or after that run
completes, so a bookkeeping push does not cancel the active matrix. Local tests
exercise acceptance/failure behavior only and cannot close the Apple gate.

## Consumer-local PPT fixture caches — portability correction

The actual glibc 2.28 full suites exposed seven tests replaying the Julia
exporter's absolute `PPT_options.cachedir`: 906 tests pass, seven fail because
that machine path is read-only in the isolated consumer. The host suite can
write there and passes 913 tests, but adds three Python cache files inside the
completed reference artifact, correctly causing a later manifest check to fail.
All originally hashed reference files remain unchanged. This is fixture/cache
transport behavior, not a plasma formula or tolerance defect.

Add a session-scoped pytest temporary cache directory and override only the
operational `cachedir` in the four consuming test functions. Preserve every
physical PPT parameter and all numerical assertions. The runtime package still
honors user-supplied cache locations. Add a post-test reference-manifest check
to both installed runners so a future consumer mutation cannot pass acceptance.
Direct pytest's own cache into the writable run directory for the read-only
glibc test mount.

Keep the original contaminated artifact and failed logs. Recover a new reference
directory by copying only files named in its original completion manifest after
verifying every original digest, plus the unchanged manifest/provenance/logs.
Do not rewrite fixture values or the manifest and do not treat extra caches as
reference data. The verified recovery avoids repeating unchanged Julia physics.
First rerun the seven previously failing cases in the actual glibc environment
with freshly generated consumer caches and read-only references. Then rebuild
the changed sdist/test artifacts and rerun complete final gates on each Python
version. Stop known-doomed old runs and leave controlled timing stopped until
the corrected installed gates pass. Deliver this necessary correction with the
prepared report/Apple follow-up; it may require restarting the active hosted
matrix because the old tests cannot satisfy the same portability requirement.

## Installed Python user guide — documentation delivery

Document the implemented public Python calls in `docs/src/python_native.md` and
add that page to the existing documentation navigation. Explain installation
from an actual matching wheel while publication is pending; do not imply the
provisional PyPI name or every target platform is already released. Keep live
acceptance status in BACKLOG. Include executable GNLSE and carrier capillary
examples, units and alias conventions, result axes/normalization, backend
diagnostics, complete-array custom responses, custom-mode requirements, profile
convergence and optional output. Reference the existing complete examples and
support matrix for detailed combinations, without inventing new API options.

Reconcile obsolete package README statements that still describe implemented
modal, carrier and custom-model support as future work. Prepare the README
revision separately while the exact package source is frozen for installed
acceptance and controlled timing. Promote that documentation-only change after
the source-dependent runs finish, preserving the previous artifact hashes. Do
not change runtime code, physical conventions or numerical tolerances for this
documentation unit. Execute the guide's Python blocks in order using the actual
installed source wheel outside the checkout, including custom/built-in
equivalence and pickle-free/HDF5 round trips. Link-check local documentation
paths and verify navigation without deploying documentation during validation.

## Apple scan diagnostic argument isolation — correction

The first actual Apple run at `fbb8b45` reaches the scan auxiliary and fails
because `Scan(name, QueueExec(...))` intentionally honors nonempty global
`ARGS`. The auxiliary's own `scan OUTPUT_JSON` arguments are therefore parsed
as scan-execution options. Preserve that failed hosted artifact. After copying
the auxiliary's two arguments to local variables, empty `ARGS` before invoking
the scan API. Keep production `Scans.jl` command-line precedence unchanged.

This corrects the prepared post-audit Apple helper introduced by `30d2eec`;
it does not alter the frozen audit workloads, upstream comparison, artifacts,
results or timing formulas. Retain the helper's original hash and verify all
other tracked audit-file hashes remain unchanged. Reproduce the failure with
the real local Julia process, then run the corrected scan and modal auxiliary
through their actual command-line interfaces. Local results prove the process
helper works, not Apple hardware performance. No shared library rebuild is
needed. Deliver the correction with the next required platform follow-up rather
than cancelling active wheel jobs for an isolated diagnostic fix.

## Downloaded wheel-matrix evidence — collection contract

Add a read-only collector under `test/standalone_wheels/` for artifacts downloaded
from one hosted run. Require the exact commit and all four CPython versions for
each requested platform (all four platforms by default). Reuse reference-manifest
and JUnit validation. Check the actual downloaded source archives and both
wheel variants against the build/test digests; verify package inventories,
native platform/interpreter tags, installed-package paths, seventeen completed
offline examples, isolation records, forbidden libraries and no-skip test counts.
Compare recorded build-source hashes to the matching checkout, accounting only
for Git's LF/CRLF text conversion on Windows. Do not follow producer absolute
paths on the consumer host; resolve the known artifact layout locally.

Record per-cell failures or missing evidence and continue inspecting the other
cells. Only a complete passing matrix produces a wheel-matrix pass. Keep the
parent workflow conclusion and any separate failures visible; a wheel report
does not establish Apple hardware diagnostics, controlled performance, release
publication or completion of the full roadmap. Export JSON and Markdown with
exact hashes and counts. Synthetic regression fixtures exercise changed hashes,
wrong commits/platforms, missing files, skipped tests and incomplete offline
runs; they never count as installed platform evidence. Exercise the collector
on real downloaded cells as soon as the producer/jobs finish.

Run the collector transport regressions alongside the existing validation-tool
checks in the reference producer and each standalone interpreter/platform job.
This wiring does not change scientific tolerances or installed test selection.

## Measured gradient workload — profiling and optimization gate

The completed controlled post-repair snapshot identifies `capillary-gradient`
as the largest Python-auto workload: 288.452 ms for setup plus solve, including
277.271 ms in the solve. The scalar and vector modal cases follow at 132.949
and 211.597 ms. Preserve that accepted snapshot, its installed source wheel and
the frozen historical audit. Profile the exact gradient workload with cProfile
using the installed baseline and the existing worker's setup/solver routines.
Retain call counts, self/cumulative cost, source hashes and field/count agreement.
Profiler timings are diagnostic only; they cannot establish an optimization.

Choose a targeted implementation only after the profile identifies its cost.
Preserve requested-position profile evaluation, serial callback ordering,
fresh density-dependent coefficients and the original numerical operations.
Do not infer constancy, sample replacement profiles, or cache arbitrary callback
results across evaluations. Document the selected change and its eligible
scope before changing package source. Compare independently installed baseline
and candidate complete workloads with the existing 10–30 sample stability and
numerical gates. Retain only a >=5% complete-solve/workload gain or removal of a
demonstrated regression; run affected setup/step/trajectory/profile/callback
checks and rebuild final artifacts for delivery. Record rejected prototypes.

The baseline profile attributes about 69% of cumulative workload time to
finite-difference group-velocity evaluation: every scalar frequency sample
repeats mode/material/density validation and interpolation. First prototype
batching the existing frequency samples for the known built-in, data-defined
gradient mode. Construct sample positions with the same scalar arithmetic,
retain the adaptive bound/step choice and left-fold coefficient reduction, and
require same-input linear and complete-field agreement. This prototype must not
enable array calls for arbitrary callbacks or change their invocation order.
Production eligibility and wiring require a separate recorded decision after
the prototype establishes numerical and complete-workload feasibility.

### Built-in scalar gradient batching — production design

The temporary experiment preserves the full trajectory and five linear
operators bit-for-bit, with diagnostic workload cost reduced from about 282 to
135 ms. Integrate a private batched evaluator into the existing derivative
algorithm, passing an evaluator explicitly so public `derivative` keeps its
serial scalar callback behavior. Generate stencil coordinates with the original
scalar operations, validate the complete returned shape/finiteness, convert
samples back to Python scalar values and retain the same adaptive bounds,
step choice and ordered reduction. Restrict the optimized mode dispatch to
first-order dispersion; other derivative orders retain their existing path.

Every public `MarcatiliMode` starts with batching disabled. Only `_Capillary`,
which constructs and owns a standard mode itself, may enable its private flag
when propagation is variable, the radius is numeric, and all gas pressure
profiles are data-defined (constant or explicit gradients, with no pressure
callbacks). Mixtures qualify only if every constituent satisfies that rule.
Constructed/custom modes, callable radius or pressure and the modal evaluator
retain scalar derivative calls. Keep `mode.dispersion` as the linear operator's
dispatch point, preserving explicit test/model overrides. No callback result
is cached and no density refresh or requested position is removed. Nonlinear
response callbacks retain their complete-field calls and order.

Test scalar/batched same-input derivatives/operators, public-mode scalar call
order, exclusion of callable profiles/radii and mixed callback profiles,
unchanged nonlinear callback traces, and full fields/counters on both grids.
Run the existing independent mode/profile/mixture/callback oracles without
changing tolerances. Build separate candidate wheels and compare them with the
immutable accepted baseline in independent processes; use two warmups and
10–30 randomized pairs, full-field/count checks and the established uncertainty
limits before accepting a >=5% complete-workload/solve gain. Preserve a clean
checkout of hosted `fbb8b45` for its pending artifact collection. Include the
prepared README correction with the new package build, preserving installer edits.

## Hosted platform validation corrections — Windows fixtures and sliced FFTs

The first complete reference producer succeeds at `fbb8b45`, then the actual
Windows tooling tests fail before building wheels. Their synthetic source is
written with native newlines, and the test blindly replaces LF with CRLF again,
creating CRCRLF on Windows. Its failing-export hook also assumes a `/gnlse`
suffix rather than a native path component. Correct the fixture to exercise
explicit LF and CRLF byte sequences on every host and use `Path(...).name`
for the exporter hook. Keep production source normalization and rejection of
changed data unchanged. Reproduce both defects without requiring a Windows
host; real Windows acceptance still requires rerunning its job.

All four actual Linux ARM64 artifacts build and install, then their seventeenth
example fails on bitwise comparison of the inverse FFT of a saved-position slice
against a slice of the full batched inverse FFT. The first sixteen examples
complete. The retained 3.11 log shows maximum absolute difference 7.731e-12 in
carrier modal fields; the spectral arrays are not the compared quantities.
Exact serialization checks must compare the saved spectrum to the same original
spectrum before either transform. Keep that bitwise check and compare the two
FFT reconstructions at the established 1e-13 reassociation norm tier, including
finite values and matching shapes. Do not use pointwise relative error in pulse
tails near zero. Record the achieved norm error, not only process success; rerun
all seventeen examples and independent output tests. No FFT implementation,
physical result, solver tolerance or independent Julia acceptance tier changes.
Retain the original failed artifacts and require actual ARM64 acceptance.

Prepare these package/example corrections after the gradient candidate's
source-dependent focused gate and controlled comparison finish, then rebuild
final artifacts. Include the already verified Apple auxiliary fix in the same
follow-up; collect useful current Linux/Apple job evidence before replacing CI.

### Delivery scheduling after confirmed platform failures

Once the current run has terminal Windows and ARM64 failures with reproduced
causes, deliver the verified corrections without waiting for its other eight
wheel cells. The user's instruction to continue implementation without waiting
for CI applies here: the new source needs another complete matrix regardless.
Branch concurrency will supersede the known-failed run. Preserve its completed
reference export, failed platform artifacts, job-state snapshot and exact-commit
checkout; do not treat cancelled cells as passes. Local source-dependent suites
remain on the original checkout and continue independently of delivery commits.
