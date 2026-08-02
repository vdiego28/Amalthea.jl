# GPU-Resident Propagation (Track S3) Design Document

> **Current status (2026-08-02, updated): the correctness block is FIXED.**
> `CudaNativeSim` computes real nonlinearity again and is verified on real
> hardware — stage derivatives match CPU native to ~1e-15 (previously
> `max|kᵢ| ≈ 3.5e-13` against CPU's 12225, i.e. pure linear propagation),
> fixed-step full-solve matches the Julia oracle to 3.5e-16, and `Luna.run`
> dense output to 1.25e-7. `set_mode_avg_params` now uploads `ωwin`, `sidx`,
> `pre`, `β`, `nlscale` and `sqrt_aeff`, and `compute_rhs_mode_avg` ports the
> CPU path's input scaling, oversampled crop/rescale, `norm_pre_beta` and
> frequency-window steps. **This also closes the `n_time`-vs-`n_time_over`
> sizing gap described in §8** — the two were not separable. A second bug was
> fixed alongside: `set_field` never seeded `ks_d[0]`, so the first `step()`
> read uninitialized device memory.
>
> **Remaining caveats:** landed scope is mode-averaged RealGrid/EnvGrid Kerr
> plus SDO Raman matching the grid (`RamanPolarField`/`RamanPolarEnv`), with
> at most 64 flattened oscillators (N₂ rotation has 49; rotation+vibration has
> 50), and with
> PPT or **thresholded ADK** plasma on RealGrid only. EnvGrid plasma is an
> explicit CPU fallback because its CUDA RHS has no plasma implementation.
> `:SiO2`, mixtures, shot noise,
> z-dependent Raman, and radial/modal/free-space GPU paths still return or
> route to CPU fallback. ADK with `threshold=false` remains CPU-only. There is still no GPU
> CI, so every GPU change needs a recorded manual hardware run.
> **2026-07-27:** adaptive acceptance now uses a real pre-acceptance trial and
> the same global weak norm as CPU/Julia; the three PPT cumtrapz operations
> are parallel prefix scans, and PPT `:auto` dispatch has a measured threshold.
> **2026-07-31:** mode-averaged RealGrid thresholded ADK is implemented and
> retained after a **2.147×** production-shaped benchmark at `n=8193`.
> `:auto` uses the exact `_GPU_ADK_N_THRESHOLD = 8193`; `threshold=false`
> remains CPU fallback. The strict CUDA baseline was green first.
> **2026-08-02:** mode-averaged SDO Raman landed for RealGrid carrier fields
> (`thg=true` and `thg=false`) and EnvGrid envelopes. The strict CUDA Raman
> item passed **53/53**, including direct stage checks, non-vacuity controls,
> fixed trajectories, rejected-step retries, and CPU fallback for `:SiO2`.
> Raman remains CPU-selected under `:auto` until a production benchmark sets a
> threshold; correctness tests use `AMALTHEA_NATIVE_GPU=on`.
> Sections below that describe the defect in the present tense are retained
> for provenance — BACKLOG S3 item 0 and
> `portlog-inbox/gpu-nonlinearity.md` are authoritative.

## 1. Goal

The original objective was to eliminate per-kernel PCIe round-trips by keeping
the simulation state resident on the GPU. The landed `CudaNativeSim` does own
the field, RK stages, error buffers, scratch, cuFFT plans, and the narrow
mode-averaged both-grid Kerr/Raman plus RealGrid PPT/thresholded-ADK state in
VRAM. The current objective is
no longer residency scaffolding; it is maintaining numerical parity with
`CpuNativeSim` while any later scope expansion is separately designed.

`CudaNativeSim` mirrors the CPU `NativeSim`: the **entire state vector and all
RK45 scratch buffers** reside in VRAM for the full duration of a `solve`.

## 2. Traffic Budget (Host ↔ Device)

- **Per RK45 sub-step (6 per step):** ZERO array transfers. Only scalars like `t` or `dt` (and maybe the reduced error scalar) are communicated.
- **Per accepted step:** The `NativeSim::native_resync_field` and `get_field`/`set_field` methods will be the *only* seams that trigger a `cudaMemcpy` from Device to Host. This happens once per accepted step (for dense output/saving to HDF5) and transfers exactly `n_t` elements (the current field). This is highly acceptable.

## 3. Data Residency

The following `NativeSim` fields will be completely migrated to device memory in `CudaNativeSim`:

- `field` (the current spectral field)
- `linop` (the linear operator)
- `ks[7]` (the 7 RK45 stage derivatives)
- `yerr` (the error estimate array)
- `ystage` (the scratch accumulation buffer)
- `eto`, `pto`, `eoo`, `poo` (the time and frequency domain interaction buffers)

## 4. Architectural Implementation (The `NativeBackend` Trait)
The implementation:

1. Renamed the original monolithic simulation to `CpuNativeSim`.
2. Defines a `NativeBackend` trait with the core interface:
   - `fn step(...) -> NativeStepResult`
   - `fn set_field(...)`
   - `fn get_field(...)`
   - `fn set_mode_avg_params(...)`
3. Stores `Box<dyn NativeBackend>` inside the FFI-facing `NativeSim`, rather
   than the originally sketched enum.
4. Delegates `native_step` and every `native_set_*` call through that trait,
   preserving one Julia FFI surface.

The one vtable call per accepted step is immaterial beside CUDA launch/sync
cost and is not a cleanup item.

## 5. cuFFT Lifecycle

- `CudaNativeSim` owns separate D2Z and Z2D `cufftHandle` plans.
- Plans are created during `native_set_mode_avg_params`, because
  `init_native_sim` knows only the spectral length.
- `free_native_sim` drops the backend and destroys the plans.
- Both `cufftPlan1d` return codes and the `cufftExecZ2D`/`D2Z` return codes
  are checked.

## 6. Kernel Requirements (`kernels.cu`)

The landed slice has CUDA kernels for:

1. **RK45 Fusion:** Fusing the stage accumulations (replicating the S1 optimization but in PTX).
2. **Error Estimation:** Computing the embedded error norm against a
   transactional fifth-order trial buffer, using the same global weak norm as
   `CpuNativeSim`.
3. **Exp-Linop:** The `exp(L * dt)` application.
4. **Kerr/Norm Broadcasts:** Applying the windowing and nonlinear scale.
5. **Cumtrapz:** PPT plasma, implemented as deterministic two-level
   256-sample Blelloch prefix scans plus parallel physics finalizers.
6. **ADK rate:** a pointwise `adk_ionization_kernel` for thresholded ADK,
   feeding the same fraction/current/polarization scans as PPT.

The 2026-07-25 correctness repair completed the surrounding pipeline: input
normalization, oversampled FFT sizing/cropping, spectral `pre/β`
normalization, and `ωwin`.

## 7. Scope of V1
The landed scope is **mode-averaged RealGrid or EnvGrid, constant linop, scalar
density, and exactly one plain Kerr response**. Both grids may add at most one
matching SDO Raman response with **1–64 flattened oscillators**; RealGrid alone may add at most one PPT or
thresholded ADK plasma response. EnvGrid plasma is explicitly ineligible.
`:auto` selects ADK only from the exact
`_GPU_ADK_N_THRESHOLD = 8193`; `threshold=false`, shot noise, mixtures,
z-dependence, `:SiO2`, and radial/modal/free-space return or route to
ineligibility and remain on `CpuNativeSim`. Plan 05's production-shaped
benchmark stayed below the 1.4x retention bar for every Raman class, so Raman
stays CPU-selected under `:auto`. Eligible GPU configurations are
not automatically rechecked because the project still lacks standing GPU CI.
Within this scope, the backend is numerically hardware-verified.

## 8. Status (updated 2026-08-02 — supersedes the historical reviews below)

The `Box<dyn NativeBackend>` decision in §4 is settled and not a TODO.

> **Historical correction, 2026-07-23; fixed 2026-07-25.** "Verified on real hardware" below
> means *ran to completion and matched the Julia oracle within the tolerance
> its test asserts*. That tolerance (`rel_solve < 1e-3`) turns out to be
> larger than the entire nonlinear effect of the config being tested
> (~4.5e-4), and direct measurement now shows the GPU-resident RHS
> contributes **no nonlinearity at all** (`max|kᵢ|` = 3.5e-13 vs the CPU
> backend's 12225; the accepted step is pure linear propagation to 15
> digits). The six bugs listed below were real and really fixed; the
> *numerical* verification claim was not the check it appeared to be. See
> `BACKLOG.md` S3 item 0 for the repair and non-vacuous re-verification.

**Verified on real CUDA hardware 2026-07-07** (RTX 5060 Ti, CUDA 13.3 —
the same machine, confirmed via `nvidia-smi`) and **wired into `RK45.jl`**,
opt-in via `AMALTHEA_USE_RUST_CUDA_NATIVE=1` (`RustNativeSimHandle`'s `use_gpu`
kwarg, dispatched from `_gpu_native_eligible`). This first real-hardware run
surfaced and fixed 6 independent bugs invisible to the (self-skipping, no
real GPU) CI-only unit tests — missing `init_gpu_context()`, a
backwards `resync_field` copy direction, temporary-lifetime UB in a kernel
launch that crashed inside `libcuda.so`, a missing `activate_context()`
before launch, a 7-argument kernel called with 6 (wrong argument, out of
order), and a cuFFT plan reused across both transform directions. Full list
with root causes: `BACKLOG.md`'s "GPU-resident stepper" entry under "Done
(recent)". The §5/§6 "Bug found and fixed (2026-07-05)" DP_B5-accumulation
fix below *did* hold up once actually run on hardware — it was correct by
inspection before verification and stayed correct after.

**Actual V1 scope, precisely** (§7's early "mode-averaged RealGrid Kerr
(+plasma)" wording was aspirational until PPT landed 2026-07-11). The current
`CudaNativeSim` `NativeBackend` impl (`cuda_native.rs`) implements
`set_mode_avg_params`, `set_plasma_params` (PPT),
`set_plasma_params_adk` for **thresholded** ADK, and resident SDO Raman via
`set_raman_params`. Other `set_*_params` (`set_radial_params`,
`set_raman_fft_params`, `set_modal_params`, `set_free_params`, every
`_zdep_*` variant, `set_mode_avg_noise[_cplx]`) still unconditionally return
`-1`; unthresholded ADK deliberately routes to CPU.
`RK45.jl`'s
`_gpu_native_eligible` docstring is the source of truth for exact scope.
Concretely, plasma-eligible configs are: `TransModeAvg`, `RealGrid`, a constant
(non-z-dependent) linop, scalar (non-mixture) density, no shot noise,
exactly one plain Kerr response, and at most one PPT plasma response or
**thresholded** `IonRateADK`, optionally with matching SDO Raman. EnvGrid
supports the same base Kerr and matching `RamanPolarEnv`, but never plasma.
Unthresholded ADK (`threshold=false`) and intermediate-broadening Raman are
intentionally unsupported by CUDA and fall back to CPU. RealGrid Raman
supports both THG flags; EnvGrid Raman uses envelope intensity.

**Plasma support added 2026-07-11** (BACKLOG.md S3 item 2; scan implementation
superseded 2026-07-27): PPT ionisation
rate lookup (reuses `ppt_ionization_kernel`, the same kernel and
`SplineSegment` upload format the standalone `AMALTHEA_USE_RUST_IONISATION`
path already uses) → a 3-stage cumtrapz sequence (ionisation fraction,
free-electron current, plasma polarisation — each fused with its adjacent
elementwise transform into one single-thread sequential kernel, since
cumtrapz is an inherently sequential prefix sum and `n_time` is small
enough at mode-averaged scale for one thread to be negligible next to this
step's FFT cost) → accumulated into `pto` before the shared time-window
kernel. Found and fixed a genuine pre-existing bug while wiring this in:
`rhs_mode_avg_real_kernel`'s call site passed its arguments in the wrong
order relative to the kernel's own declaration, so the Kerr kernel had
never actually written its result into the buffer that gets forward-FFT'd
— present since the original 2026-07-05/07 GPU work, never caught because
the existing Kerr-only test's energy was weak enough for the resulting
error to stay under tolerance regardless. See BACKLOG.md's S3 item 2 for
the full writeup, including why the new Kerr+plasma equivalence test uses
a looser (~5e-2) tolerance than the Kerr-only test's ~1e-3 (diagnosed, not
assumed — plasma's Keldysh-exponential field sensitivity amplifies the
existing `n_time`-vs-`n_time_over` gap below, confirmed via an energy sweep
showing linear scaling, and via the CPU-resident native path matching the
Julia oracle to `1.3e-16` on the identical config).

**Historical fidelity gap, fixed 2026-07-25:** the GPU Kerr/plasma FFT buffers/plans
are sized `n_time` (`grid.t`), not `n_time_over` (`grid.to`) — it skips the
oversampling/anti-aliasing padding both `CpuNativeSim` and Julia apply.
Earlier numbers attributed to this approximation are not trustworthy while
the nonlinear RHS is absent. Fix the sizing/crop path as part of S3 item 0,
then remeasure its residual effect; do not preserve it as an intentional
approximation without new evidence.

**Test coverage:** `test/test_native_cuda.jl` has two testitems (Kerr-only,
Kerr+plasma), each constructing a GPU-backed stepper via
`withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1")`; both self-skip cleanly on CI
(no GPU/toolkit) but on real hardware assert `_gpu_native_eligible`
actually returned `true` and check full-solve field agreement against
`PreconStepper`. The 2026-07-25 replacement tightens the Kerr/full-solve
tolerances to 1e-12, checks stage scale, and independently measures the
nonlinear control effect so a zero-nonlinearity backend cannot pass. The
2026-07-27 extension deliberately rejects and retries both Kerr and Kerr+PPT,
asserts rejection leaves the field bit-exact, compares `err`/`dtn` against
CPU native, and completes adaptive trajectories at `5.42e-15` / `2.24e-15`
relative agreement. Focused hardware result: 59/59.
`amalthea/src/lib.rs`
and `amalthea/tests/test_gpu_cuda.jl` also self-skip without a GPU —
**still true in CI today**: no CI runner has a GPU, so none of this
executes except when run by hand on hardware like this machine. This is
`BACKLOG.md`'s open "GPU CI coverage" item (Phase G.2) — not resolved by
either the 2026-07-07 or 2026-07-11 verification passes, both one-time
manual runs, not a standing CI job.

**What's still open, in order:**

1. Add scheduled/dedicated GPU CI.
2. The lead has kept standing GPU CI deferred. The local-hardware-gated
   mode-averaged RealGrid ADK and mode-averaged SDO Raman expansions are
   complete and retained; radial/modal/free-space remain unimplemented, and
   intermediate-broadening Raman remains CPU-only. PPT and ADK share the
   completed parallel scan pipeline.

The problem-size dispatch policy is:
`AMALTHEA_NATIVE_GPU=off/on/auto`, with `auto` selecting Kerr-only problems at
`length(y0) ≥ 16384` on RealGrid and **exactly** `length(y0) ≥ 32768` on
EnvGrid, supported PPT problems at `length(y0) ≥ 8192`, and supported
thresholded ADK problems at **exactly** `length(y0) ≥ 8193`.
The PPT threshold was remeasured after parallelizing the scans: GPU/CPU is
0.82× at n=2049, 1.08× at n=4097, and 2.94× at n=8193, so 8192 deliberately
skips the marginal crossover. Both policies remain behind the explicit
`AMALTHEA_USE_RUST_CUDA_NATIVE=1` master opt-in. EnvGrid retains its own
threshold because its c2c FFT timing differs from the RealGrid r2c/c2r path;
the RTX 5060 Ti sweep found 16,384 marginal in one batch and 32,768 stable at
3.31–3.98× GPU/CPU. Full evidence is in Luna feature plan 04.

## 9. First expansion completed — mode-averaged RealGrid ADK (2026-07-31)

The strict local hardware baseline completed first on the RTX 5060 Ti (driver
610.43.02): required-CUDA Rust tests, real-PTX release build, focused CUDA/
dense/dispatch tests, and the strict balanced Rust group. This did not register
the machine as a self-hosted runner; standing GPU CI remains deferred.

The landed pointwise CUDA rate kernel matches `AdkIonizationRate::rate`:
absolute field, exact zero for non-finite and below-threshold values, the
transferred power/exponential constants, and the existing optional
cycle-average multiplier. `set_plasma_params_adk` stores those constants and
selects the rate kind; the downstream parallel fraction/current/polarization
scans and finalizers are reused unchanged. The FFI signature does not change.

Eligibility stays narrow: constant linop, scalar density, mode-averaged
RealGrid, exactly one plain Kerr response, at most one **thresholded** ADK
plasma response, and no Raman, shot noise, mixture, z-dependence, or alternate
geometry.
Acceptance requires direct rate-boundary/CPU comparison, nonzero stage-scale
agreement, a Julia ADK control effect at least 100× the comparison tolerance,
fixed-step and adaptive trajectories, and a deliberate rejection whose field
is bit-exact before retry.

Direct strict-CUDA ADK tests and Julia integration (**17/17**) cover rate
boundaries, non-vacuity, stage scale, fixed/adaptive trajectories, and
bit-exact rejected-state retry; the focused CUDA suite passed **101/101**.
At `length(Eω)=8193`, `n_time_over=32768`, warmup plus the minimum of three
five-step batches measured CPU **[3.726, 3.707, 3.683]** ms/step and GPU
**[2.433, 1.965, 1.716]** ms/step: **2.147×**, passing the `>=1.4×` retention
gate. The source and eligibility are retained at the exact threshold **8193**;
do not round it to 8192. `threshold=false` remains CPU fallback. Full evidence
is in `PLANS.md` §11.4 and `PORT_LOG.md`.

---

## 10. Second expansion — mode-averaged SDO Raman (landed 2026-08-02)

This landed extension adds two ordered subphases to the explicitly opt-in CUDA
backend. The first supports `TransModeAvg` + `RealGrid` +
`RamanPolarField`; the second adds `TransModeAvg` + `EnvGrid` +
`RamanPolarEnv`. Both use the existing SDO/rotational oscillator flattening
and the resident `native_set_raman_params` FFI signature. The
intermediate-broadening `:SiO2` response remains on its existing CPU/native
FFT-convolution path and is not part of this item.

### 10.1 RealGrid carrier Raman

`CudaNativeSim` stores the precomputed `PrecomputedStepCoeffs` for each
oscillator and resident real intensity/polarisation buffers. The existing
`raman_ade_kernel` is launched with the mode-averaged oversampled length and a
generated **64-oscillator capacity contract** shared by Rust validation and
the PTX header. N₂ rotation (49) and rotation+vibration (50) therefore run in
CUDA; larger flattened responses remain CPU fallback. The kernel no longer
clamps excess oscillators, and Raman buffer byte counts use checked
multiplication before allocation. The launch has no per-stage allocation or
host copy.
there is no per-stage allocation or host copy. `thg=true` uses `E²` directly.
`thg=false` uses a resident c2c Hilbert plan and the same analytic-signal
convention as `RamanPolarField` before launching the ADE kernel. The result is
accumulated as `pto += ρ·E·P`, before the shared time-window and FFT stages.

### 10.2 EnvGrid envelope Raman

The CUDA setup gains a c2c mode-averaged plan and complex time/frequency
buffers. The time-domain intensity is `0.5·|E|²`; the ADE output remains real;
the contribution is accumulated as `pto += E·(ρ·P)`. The existing field/RK
buffers and EnvGrid Kerr path are reused. No `thg` branch exists for
`RamanPolarEnv`.

### 10.3 Eligibility, fallback, and verification

Eligibility remains constant-linop, scalar-density, mode-averaged, one plain
Kerr plus at most one `CombinedRamanResponse` made from 1–64 SDO or flattened
rotational oscillators. Mixtures, shot noise, z-dependent Raman, `:SiO2`, and
all radial/modal/free-space combinations remain `NativeIneligible` and use
the CPU resident/Julia fallback. The master CUDA opt-in and explicit
`AMALTHEA_NATIVE_GPU=on` correctness route do not change.

Verification passed with a direct CUDA-vs-CPU-ADE stage check, a non-vacuous
Raman-on/off oracle effect, fixed-step and adaptive reject/retry trajectories,
and explicit `:SiO2` CPU fallback coverage. The strict CUDA item passed 53/53;
RealGrid and EnvGrid GPU-vs-CPU fixed trajectories were at approximately
`5e-16` and `2e-16`, respectively. Plan 05 then measured the production-shaped
Raman classes with two warm-up steps and three five-step batches. The largest
speedup was only `1.141x` (EnvGrid, one vibrational oscillator, `Nω=32768`),
below the established `1.4x` retention bar; the rotational classes stayed at
parity. Therefore every Raman `:auto` policy threshold is deliberately unset:
supported Raman remains CPU-selected under `:auto`, while `:on` remains the
explicit CUDA correctness/experiment route. The complete table and the bounded
large-rotational benchmark gotcha are recorded in
`luna-feature-plans/LUNA_FEATURE_PLAN_05_GPU_RAMAN_AUTO_POLICY.md`.

---

## Historical: Status (2026-07-05 review, pre-hardware — superseded above)
Implemented as `Box<dyn NativeBackend>` rather than the `enum` described in §4 (functionally
equivalent, just not what was planned). **Not wired to Julia** — no `src/*.jl` file calls
`init_cuda_native_sim`; this is inert scaffolding with zero effect on the shipped CPU native
path. **Untested on real hardware**: this dev machine has an NVIDIA driver but no `nvcc`
toolkit, so `kernels.cu` never compiles to real PTX and `CudaNativeSim::new` fails to load
(the `lib.rs` unit test self-skips).

**Bug found and fixed (2026-07-05):** `CudaNativeSim::step` was never applying the final
5th-order solution weights (`DP_B5` in `native.rs`'s `CpuNativeSim::step`) before accepting a
step — it only ran the internal-stage accumulation (`DP_B`) and then re-propagated the
*unmodified* old field, silently dropping the entire nonlinear contribution. Fixed by adding
an extra `rk45_accumulate_stage_fn` launch (in-place on `field_d`, using `DP_B5` weights,
gated on `locextrap != 0` exactly like the CPU reference) right before the final
`apply_prop` call. Compiles and passes the existing (self-skipping) unit tests, but **has
still never been run on real CUDA hardware** — the fix is only checked for logical parity
against `CpuNativeSim::step`, not numerically verified.

**Opt-in gate added:** `init_cuda_native_sim` now refuses to initialize (returns null +
prints a warning to stderr) unless `AMALTHEA_USE_RUST_CUDA_NATIVE=1` is set in the environment,
and prints a second warning on successful opt-in reminding the caller this path is
unverified. This is deliberately stricter than a normal `AMALTHEA_USE_RUST_*` feature toggle —
those default-enable once verified; this one requires explicit, repeated opt-in until it has
been checked against the Julia oracle on real GPU hardware. See
`test_cuda_native_sim_ffi_gated_by_env_var` in `lib.rs`.

Still not wired to Julia/`RK45.jl`'s dispatch — do that only after real-hardware
verification. See `BACKLOG.md`'s "GPU-resident stepper" entry for the full status.
