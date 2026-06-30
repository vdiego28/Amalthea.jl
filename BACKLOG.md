# Backlog

Deferred work and known issues for Luna-Rust.jl. Severity: 🔴 correctness · 🟡 robustness/CI · ⚪ informational.

## Open items

### 🟡 Wire remaining Rust kernels into the Julia pipeline
PPT ionization is now wired (opt-in via `LUNA_USE_RUST_IONISATION=1`).
The pattern: Rust exports an opaque handle lifecycle + a vector-eval FFI function;
Julia stores the handle in the struct and routes the in-place vector call through
`ccall`; a `@testitem tags=[:rust]` equivalence test guards the boundary.

Remaining kernels to wire (same pattern, in this order):
1. ✅ **PPT ionization** (`IonRatePPTAccel`) — `LUNA_USE_RUST_IONISATION` toggle —
   `test/test_ionisation_rust.jl`
2. ✅ **Time-domain Raman** (`raman.rs` `TimeDomainRamanSolver`) — toggle
   `LUNA_USE_RUST_RAMAN`, `init_raman_solver` / `free_raman_solver` / `raman_solve`
   exported, wired into `Nonlinear.jl` hot loop for carrier-field SDO responses
   (`CombinedRamanResponse` with all-SDO `Rs`, density-independent τ2) —
   `test/test_raman_rust.jl`. Follow-ups: rotational multi-oscillator (FFI already
   supports n_osc>1; needs Julia-side extraction of per-J Ω/K arrays);
   density-dependent τ2 (add `raman_update_coeffs` FFI entry); intermediate-broadening
   (Gaussian damping — stays Julia indefinitely); envelope (`RamanPolarEnv`) Rust path
   (needs real-buffer copy for complex→real conversion).
3. ✅ **Waveguide dispersion — Zeisberger** (`dispersion.rs` `ZeisbergerNeff`) — toggle
   `LUNA_USE_RUST_DISPERSION`, `init_zeisberger_neff` / `free_zeisberger_neff` /
   `zeisberger_neff_vector` exported, wired into `Antiresonant.jl` via a specialised
   `neff_β_grid(grid, ::ZeisbergerMode, λ0)` that batch-evaluates neff over the
   positive-frequency grid per propagation step — `test/test_dispersion_rust.jl`.
   Full Zeisberger eq.(15) parity: all four mode kinds (HE/EH/TE/TM), ϕ wall-thickness
   phase, σ⁴ real (C) and imaginary (D·loss_scale) terms. Equivalence at ~1e-12
   (same formula + Julia-supplied nco/ncl → only float-reassociation differences).
   Follow-ups: Rust-side multi-term Sellmeier (offload nco/ncl computation too);
   const-linop one-time setup path (negligible cost, left on Julia indefinitely).

3a. ✅ **Waveguide dispersion — MarcatiliMode** (`dispersion.rs` `MarcatiliNeff`) — same
    `LUNA_USE_RUST_DISPERSION` toggle; `init_marcatili_neff` / `free_marcatili_neff` /
    `marcatili_neff_vector` exported. Wired into the constant-radius specialisation
    `neff_β_grid(grid, ::MarcatiliMode{<:Number}, λ0)` in `Capillary.jl`. Nwg(ω)
    precomputed ONCE at setup (cladding-dependent, z-independent) and stored in the
    Rust handle; per step only nco(ω; z) is passed. Also adds z-level memoization
    even on the Julia-only fallback path (batching all sidcs before returning cached
    values). Equivalence is bitwise (0.0 rel error) — same IEEE 754 formula + same
    Float64 inputs. Model `:full` (`sqrt(εco-nwg)`) and `:reduced` (`1+(εco-1)/2-nwg`)
    both wired. Tests: `test/test_dispersion_rust.jl` (second `@testitem`).
4. ✅ **QDHT batch transform** — toggle `LUNA_USE_RUST_QDHT`, `init_qdht_ffi` /
   `free_qdht_ffi` / `qdht_ffi_mul_real` / `qdht_ffi_ldiv_real` / `qdht_ffi_mul_cplx` /
   `qdht_ffi_ldiv_cplx` exported. Wired into `TransRadial` in `NonlinearRHS.jl` via
   type-stable `_qdht_mul!` / `_qdht_ldiv!` dispatch. Stores Julia's T matrix
   (transposed to row-major at init); per-call transform uses Rayon parallel
   row-vector dot products with pre-allocated scratch (4×n_r×n_time), avoiding
   the two `permutedims` allocations that Julia's dim=2 QDHT path incurs.
   Handles both `Float64` (RealGrid) and `ComplexF64` interleaved (EnvGrid).
   Equivalence: ~1e-13 relative error vs Julia BLAS path (summation order differs).
   Tests: `test/test_qdht_rust.jl` (`@testitem tags=[:rust]`).
   Follow-ups: wire `TransFree` (2D Cartesian FFT, different transform type — stays Julia);
   consider cblas/openblas DGEMM binding for peak throughput.
5. ⬜ **RK45 stepper** (`stepper.rs` `Dopri5Stepper`) — largest scope; requires exporting
   a callback-based or fixed-array step loop through FFI.

- **Safety net:** `test/test_rust_ffi.jl`, `test/test_ionisation_rust.jl`,
  `test/test_raman_rust.jl`, `test/test_dispersion_rust.jl`, and
  `luna-rust/tests/*.jl` (`@testitem tags=[:rust]`, auto-discovered).

### 🔴 Real file locking for parameter scans on Windows
`luna-rust/src/scans.rs` `FlockLock` is a **no-op on non-Unix targets** (`#[cfg(not(unix))]`
returns an empty struct; `lock()`/`unlock()` do nothing). Concurrent scans on Windows are
therefore not process-safe and may race on the shared HDF5 queue file.
- **Fix:** implement locking via Windows `LockFileEx` / `UnlockFileEx`.
- **Blocker:** needs a Windows runner to validate; untestable from Linux/macOS.
- Currently latent — `ScanQueue` is only reachable via FFI, which `src/` doesn't call yet.

### 🟡 GPU CI coverage
`luna-rust/tests/test_gpu_cuda.jl` and the GPU numerical-equivalence tests in
`luna-rust/src/lib.rs` self-skip when no GPU/CUDA is present, so the GPU code paths
(CUDA/Vulkan dispatch, GPU QDHT/Raman/ionization) are never exercised in CI.
- **Fix:** add a GPU-equipped CI runner (or a scheduled job) that actually executes them.

## Informational / no action planned

- ⚪ `deps/build.jl` sets `RUSTFLAGS=""`, which overrides `.cargo/config.toml`'s
  `target-cpu=native` for the package-driven build. This is the intended portability
  safeguard (the runtime dispatcher in `dispatch.rs` selects the ISA at runtime); the
  native opt only applies to a manual `cargo build`.

## Done (recent)

- ✅ Skip-guard added to `test/test_rust_ffi.jl` and the four `luna-rust/tests/*.jl`
  testitems so a fresh-clone `]test Luna` skips (not fails) when the Rust lib is unbuilt.
- ✅ Windows scan-lock no-op documented in `scans.rs` (with `TODO`) and `luna-rust/README.md`.
- ✅ `CLAUDE.md` extended with a Math & Advancements section.
- ✅ **Removed root `Cargo.toml` workspace** so `cargo build` in `luna-rust/` writes to
  `luna-rust/target/release/` (where all tests look). Root `Cargo.lock` also removed.
  Previously, the workspace redirected output to `./target/release/`, silently breaking all
  `rust` CI jobs on a clean checkout. Also fixes `rust-cache` effectiveness (caches
  `workdir: luna-rust` but output was going to root `target/`).
- ✅ **Fixed `pointcalc!` data race in `src/NonlinearRHS.jl`**: `Threads.@threads` was
  added to the `TransModal` integration loop, but the loop body writes to shared struct
  fields (`t.Erω`, `t.Prω`, `t.Prmω`, etc.). Multiple threads clobbering each other's
  intermediate buffers produced corrupted modal overlap values → RK45 rejected every step →
  `"Reached limit for step repetition (10)"` on all `physics`/`sim-interface`/`sim-multimode`
  jobs. Fixed by reverting to a sequential loop.
- ✅ **Fixed `IonRatePPTAccel` to clamp instead of throw** (`src/Ionisation.jl:419`):
  the adaptive stepper evaluates the RHS at rejected trial points; a large-field trial step
  above `Emax` is recoverable and should not crash the run. Now returns `exp(spline(Emax))`
  (saturated rate) instead of calling `error()`.
