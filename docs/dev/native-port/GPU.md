# GPU-Resident Propagation (Track S3) Design Document

> **Current status (2026-07-25, updated): the correctness block is FIXED.**
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
> **Remaining caveats:** scope is still only mode-averaged RealGrid Kerr +
> PPT plasma (everything else returns `-1` and falls back), and there is still
> no GPU CI, so every GPU change needs a recorded manual hardware run.
> **2026-07-27:** adaptive acceptance now uses a real pre-acceptance trial and
> the same global weak norm as CPU/Julia; the three PPT cumtrapz operations
> are parallel prefix scans, and PPT `:auto` dispatch has a measured threshold.
> Sections below that describe the defect in the present tense are retained
> for provenance — BACKLOG S3 item 0 and
> `portlog-inbox/gpu-nonlinearity.md` are authoritative.

## 1. Goal

The original objective was to eliminate per-kernel PCIe round-trips by keeping
the simulation state resident on the GPU. The landed `CudaNativeSim` does own
the field, RK stages, error buffers, scratch, cuFFT plans, and the narrow
mode-averaged Kerr/PPT state in VRAM. The current objective is no longer
residency scaffolding; it is numerical parity with `CpuNativeSim` before any
scope expansion.

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

The 2026-07-25 correctness repair completed the surrounding pipeline: input
normalization, oversampled FFT sizing/cropping, spectral `pre/β`
normalization, and `ωwin`.

## 7. Scope of V1
The intended landed scope is **mode-averaged RealGrid, constant linop, scalar
density, exactly one plain Kerr response, and at most one PPT plasma
response**. ADK, Raman, shot noise, z-dependence, and radial/modal/free-space
return or route to ineligibility and remain on `CpuNativeSim`. This table
describes intended eligibility only; eligible GPU configurations are not
automatically rechecked because the project still lacks standing GPU CI.
Within this scope, the backend is numerically hardware-verified.

## 8. Status (updated 2026-07-25 — supersedes the historical reviews below)

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

**Actual V1 scope, precisely** (§7 said "mode-averaged RealGrid Kerr
(+plasma)" — the "(+plasma)" was aspirational and was wrong until
2026-07-11, see below). `CudaNativeSim`'s `NativeBackend` impl
(`cuda_native.rs`) implements `set_mode_avg_params` and (as of 2026-07-11)
`set_plasma_params` (PPT only); every other `set_*_params`
(`set_plasma_params_adk`, `set_radial_params`, `set_raman_params[_fft]`,
`set_modal_params`, `set_free_params`, every `_zdep_*` variant,
`set_mode_avg_noise[_cplx]`) unconditionally returns `-1`. `RK45.jl`'s
`_gpu_native_eligible` docstring is the source of truth for exact scope.
Concretely, eligible configs are: `TransModeAvg`, `RealGrid`, a constant
(non-z-dependent) linop, scalar (non-mixture) density, no shot noise,
exactly one plain Kerr response, and at most one plasma response using PPT
ionisation (`IonRatePPTAccel` — ADK still returns `-1`).

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
2. Only after that, decide whether to expand beyond mode-averaged RealGrid
   Kerr/PPT. Raman, ADK, radial/modal/free-space, and parallel plasma scans
   remain unimplemented and should continue routing to CPU until individually
   designed and tested.

The problem-size dispatch policy is:
`AMALTHEA_NATIVE_GPU=off/on/auto`, with `auto` selecting Kerr-only problems at
`length(y0) ≥ 16384` and supported PPT problems at `length(y0) ≥ 8192`.
The PPT threshold was remeasured after parallelizing the scans: GPU/CPU is
0.82× at n=2049, 1.08× at n=4097, and 2.94× at n=8193, so 8192 deliberately
skips the marginal crossover. Both policies remain behind the explicit
`AMALTHEA_USE_RUST_CUDA_NATIVE=1` master opt-in.

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
