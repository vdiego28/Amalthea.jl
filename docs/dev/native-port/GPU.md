# GPU-Resident Propagation (Track S3) Design Document

## 1. Goal
The primary objective of Track S3 is to eliminate the severe PCIe bottleneck that currently throttles GPU acceleration in Amalthea. Currently, kernels (like Ionization or Raman) are individually accelerated, but the main simulation state (the spectral field `y`, the RK45 intermediate stages `k1..k7`, the step-error estimators `yerr`, and the scratch space `ystage`) lives on the CPU host. This requires copying arrays back and forth every RK45 sub-step.

We will introduce a `CudaNativeSim` that mirrors the CPU `NativeSim`. In this architecture, the **entire state vector and all RK45 scratch buffers** reside in VRAM for the full duration of a `solve`. 

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
To support this transparently to Julia, we will:
1. Rename the existing monolithic `NativeSim` struct in `native.rs` to `CpuNativeSim`.
2. Define a `NativeBackend` trait (or use an enum wrapper) that defines the core interface:
   - `fn step(...) -> NativeStepResult`
   - `fn set_field(...)`
   - `fn get_field(...)`
   - `fn set_mode_avg_params(...)`
3. Expose a new `NativeSim` which is just an `enum { Cpu(CpuNativeSim), Gpu(CudaNativeSim) }`.
4. The FFI `native_step` and `native_set_*` functions will just `match` on this enum and delegate the work. This avoids any dynamic dispatch overhead (no `Box<dyn>`) while perfectly preserving the FFI ABI for Julia.

## 5. cuFFT Lifecycle
- The `CudaNativeSim` will own `cufftHandle` plans.
- These plans will be created during the `native_set_mode_avg_params` configuration step (since `init_native_sim` only knows the spectral length `Nω`, not `Nt`).
- They will be destroyed when `free_native_sim` drops the handle.

## 6. Kernel Requirements (`kernels.cu`)
We will need CUDA kernels for:
1. **RK45 Fusion:** Fusing the stage accumulations (replicating the S1 optimization but in PTX).
2. **Error Estimation:** Computing the embedded error norm.
3. **Exp-Linop:** The `exp(L * dt)` application.
4. **Kerr/Norm Broadcasts:** Applying the windowing and nonlinear scale.
5. **Cumtrapz:** For plasma (using a work-efficient prefix scan).

## 7. Scope of V1
As per `SUGGESTIONS.md`, V1 of `CudaNativeSim` was planned as **mode-averaged RealGrid Kerr (+plasma)**. What actually shipped (see §8) is narrower: **mode-averaged RealGrid, scalar density, exactly one plain Kerr response, and nothing else** — plasma was never implemented (`set_plasma_params` and every other `set_*_params` beyond `set_mode_avg_params` returns `-1`). Free-space, Modal, and Radial geometries stay on `CpuNativeSim`; `RK45.jl`'s `_gpu_native_eligible` falls back to CPU for anything outside this scope.

## 8. Status (updated 2026-07-11 — supersedes the 2026-07-05 review below)

**Reconciled against §4:** implemented as `Box<dyn NativeBackend>`, not the
`enum { Cpu, Gpu }` described in §4 ("avoids dynamic dispatch overhead"). This
is a deliberate, permanent deviation, not a TODO — the dynamic-dispatch cost
is one vtable call per `native_step` (once per accepted step, not once per
RK45 sub-stage; see §2's traffic budget), immaterial next to a step's actual
GPU kernel-launch/sync cost, and `Box<dyn NativeBackend>` is what lets
`CpuNativeSim`/`CudaNativeSim` share the exact same FFI entry points
(`native_step`, `native_set_*`) as one polymorphic handle instead of every
call site matching on an enum. §4 above is left as historical design intent;
treat "`Box<dyn NativeBackend>`, not an `enum`" as current fact everywhere
else in this document.

**Verified on real CUDA hardware 2026-07-07** (RTX 5060 Ti, CUDA 13.3 —
the same machine, confirmed via `nvidia-smi`) and **wired into `RK45.jl`**,
opt-in via `LUNA_USE_RUST_CUDA_NATIVE=1` (`RustNativeSimHandle`'s `use_gpu`
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

**Plasma support added 2026-07-11** (BACKLOG.md S3 item 2): PPT ionisation
rate lookup (reuses `ppt_ionization_kernel`, the same kernel and
`SplineSegment` upload format the standalone `LUNA_USE_RUST_IONISATION`
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

**Known, documented, un-fixed fidelity gap** (not a correctness bug — a
bounded, intentional approximation): the GPU Kerr/plasma FFT buffers/plans
are sized `n_time` (`grid.t`), not `n_time_over` (`grid.to`) — it skips the
oversampling/anti-aliasing padding both `CpuNativeSim` and Julia apply.
Kerr-only full-solve agreement against `PreconStepper` is ~4.5e-4 (test
tolerance `<1e-3`); Kerr+plasma is ~2.0e-2 at the tested energy (test
tolerance `<5e-2`) — both versus the CPU-resident native path's ~1e-6
(Phase 1) / ~1e-16 (this exact plasma config). Fixing this is a real
buffer-sizing change in `cuda_native.rs`/`kernels.cu`, not attempted yet.

**Test coverage:** `test/test_native_cuda.jl` has two testitems (Kerr-only,
Kerr+plasma), each constructing a GPU-backed stepper via
`withenv("LUNA_USE_RUST_CUDA_NATIVE" => "1")`; both self-skip cleanly on CI
(no GPU/toolkit) but on real hardware assert `_gpu_native_eligible`
actually returned `true` (not a vacuous CPU-vs-CPU comparison) and check
full-solve field agreement against `PreconStepper`. `luna-rust/src/lib.rs`
and `luna-rust/tests/test_gpu_cuda.jl` also self-skip without a GPU —
**still true in CI today**: no CI runner has a GPU, so none of this
executes except when run by hand on hardware like this machine. This is
`BACKLOG.md`'s open "GPU CI coverage" item (Phase G.2) — not resolved by
either the 2026-07-07 or 2026-07-11 verification passes, both one-time
manual runs, not a standing CI job.

**What's still open, per `BACKLOG.md`'s S3 list:**
1. ~~Design doc reconciliation~~ — done 2026-07-11.
2. ~~PPT plasma~~ — done 2026-07-11 (mode-averaged only). Raman, radial/
   modal/free-space geometries, and ADK plasma remain unimplemented; every
   other `set_*_params` still returns `-1` as noted above.
3. Problem-size dispatch threshold (measured crossover, not guessed) so
   small grids stay on CPU; `LUNA_NATIVE_GPU=0/1/auto` env override — not
   started. Today it's all-or-nothing per `LUNA_USE_RUST_CUDA_NATIVE`.
4. Raman ADE / ADK plasma GPU kernels, radial/modal/free-space geometries —
   or an explicit `NativeIneligible`-style split keeping those configs on
   CPU, documented as such. A work-efficient parallel prefix scan for
   cumtrapz (superseding item 2's single-thread kernels) would matter for
   radial's much larger per-column plasma state, not at mode-averaged scale.
5. `test/test_native_gpu.jl`-style phase-test coverage of items 3-4, once
   they exist.
6. The `n_time_over` Kerr/plasma-buffer fidelity gap above.
7. Phase G.2 / "GPU CI coverage" — a scheduled/dedicated GPU CI runner so
   this doesn't silently bit-rot un-exercised again between manual runs.

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
prints a warning to stderr) unless `LUNA_USE_RUST_CUDA_NATIVE=1` is set in the environment,
and prints a second warning on successful opt-in reminding the caller this path is
unverified. This is deliberately stricter than a normal `LUNA_USE_RUST_*` feature toggle —
those default-enable once verified; this one requires explicit, repeated opt-in until it has
been checked against the Julia oracle on real GPU hardware. See
`test_cuda_native_sim_ffi_gated_by_env_var` in `lib.rs`.

Still not wired to Julia/`RK45.jl`'s dispatch — do that only after real-hardware
verification. See `BACKLOG.md`'s "GPU-resident stepper" entry for the full status.
