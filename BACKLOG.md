# Backlog

Deferred work and known issues for Luna-Rust.jl. Severity: 🔴 correctness · 🟡 robustness/CI · ⚪ informational.

## Open items

### 🔴 Wire the Rust backend into the Julia pipeline
The `luna-rust` crate builds and its FFI surface is fully tested, but `src/*.jl`
contains **zero `ccall`s** — `prop_capillary` / `prop_gnlse` / `RK45` currently run in
pure Julia. The README and `CLAUDE.md` describe the Rust-accelerated architecture as the
design goal; the actual kernel routing is not connected yet.
- **Scope:** large. Decide per-kernel rollout order (e.g. dispersion → QDHT → Raman →
  stepper) and add `ccall` wrappers on the Julia side guarded by the existing FFI tests.
- **Safety net already in place:** `test/test_rust_ffi.jl` and `luna-rust/tests/*.jl`
  (`@testitem tags=[:rust]`, discovered recursively by `@run_package_tests`).

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
