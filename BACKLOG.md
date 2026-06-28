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
2. ⬜ **Time-domain Raman** (`raman.rs` `TimeDomainRamanSolver`) — toggle
   `LUNA_USE_RUST_RAMAN`, export `init_raman_solver` / `free_raman_solver` /
   `raman_solve_step`, wire into `Raman.jl` or `NonlinearRHS.jl`.
3. ⬜ **Waveguide dispersion** (`dispersion.rs` `ChebyshevDispersion`) — toggle
   `LUNA_USE_RUST_DISPERSION`, export `init_chebyshev_dispersion` / `eval_dispersion`,
   wire into `LinearOps.jl` or `Capillary.jl` setup path.
4. ⬜ **QDHT** (`diffraction.rs` `Qdht`) — toggle `LUNA_USE_RUST_QDHT`, wire into
   `NonlinearRHS.jl` (`TransRadial`/`TransFree`).
5. ⬜ **RK45 stepper** (`stepper.rs` `Dopri5Stepper`) — largest scope; requires exporting
   a callback-based or fixed-array step loop through FFI.

- **Safety net:** `test/test_rust_ffi.jl`, `test/test_ionisation_rust.jl`, and
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
