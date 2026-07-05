# GPU-Resident Propagation (Track S3) Design Document

## 1. Goal
The primary objective of Track S3 is to eliminate the severe PCIe bottleneck that currently throttles GPU acceleration in Luna-Rust. Currently, kernels (like Ionization or Raman) are individually accelerated, but the main simulation state (the spectral field `y`, the RK45 intermediate stages `k1..k7`, the step-error estimators `yerr`, and the scratch space `ystage`) lives on the CPU host. This requires copying arrays back and forth every RK45 sub-step.

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
As per `SUGGESTIONS.md`, V1 of `CudaNativeSim` is scoped strictly to **mode-averaged RealGrid Kerr (+plasma)**. Free-space, Modal, and Radial geometries will stay on `CpuNativeSim` initially. The runtime dispatcher will automatically fall back to CPU for unsupported configurations.

## 8. Status (2026-07-05 review)
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
