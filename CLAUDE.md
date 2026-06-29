# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Luna-Rust.jl** is a performance-focused fork of [Luna.jl](https://github.com/LupoLab/Luna.jl) that simulates nonlinear optical pulse propagation (UPPE/GNLSE) in waveguides and free space. The Julia high-level interface is preserved and backwards-compatible, while compute-critical kernels are offloaded to a native Rust shared library (`luna-rust`) via Julia's `ccall` FFI.

## Repository Structure

```
Luna-Rust.jl/
├── src/           # Julia source (one file per module, all included in Luna.jl)
├── luna-rust/     # Rust crate (cdylib + rlib, built separately)
│   ├── src/       # Rust modules (dispatch, ffi, stepper, dispersion, diffraction, …)
│   ├── tests/     # Rust-side integration tests (run with `cargo test`)
│   └── benches/   # Criterion benchmarks
├── python/        # Python wrapper (juliacall-based, pip-installable)
├── test/          # Julia test suite (TestItems-tagged, run via runtests.jl)
├── deps/build.jl  # Build script: compiles Rust library and configures PyCall
└── examples/      # Simulation examples (simple_interface/, low_level_interface/)
```

## Build & Development Commands

### Rust backend

```bash
# Build the release shared library (required before Julia FFI tests)
cd luna-rust
cargo build --release        # → target/release/libluna_rust.{so,dylib,dll}

# Run Rust unit tests
cargo test

# Run benchmarks
cargo bench
```

> The `.cargo/config.toml` sets `target-cpu=native`. Override with `RUSTFLAGS=""` in CI or when cross-compiling.

### Julia package

```julia
# Install all Julia dependencies and trigger the build script (compiles Rust + PyCall)
]instantiate      # or: using Pkg; Pkg.instantiate()

# Run the full Julia test suite
]test Luna

# Run a single test group (env var controls TestItemRunner tag filter)
LUNA_TEST_GROUP=physics julia --project test/runtests.jl
# Valid groups: physics, rust, sim-interface, sim-multimode, sim-propagation, io, fields, All (default)
```

> Tests use `:estimate` FFTW mode to run faster; the package itself defaults to `:patient`.

### Python bindings

```bash
pip install -e ./python          # editable install
pytest python/tests/             # run Python API tests
```

## Architecture

### Propagation pipeline (Julia side)

1. **Grid** (`Grid.jl`) — defines the time/frequency sampling (`RealGrid` for field-resolved UPPE, `EnvGrid` for envelope/GNLSE).
2. **Modes** (`Modes.jl`, `Capillary.jl`, `Antiresonant.jl`, `StepIndexFibre.jl`, …) — waveguide mode geometry and dispersion.
3. **Fields** (`Fields.jl`) — pulse generation (Gaussian, sech, custom).
4. **LinearOps** (`LinearOps.jl`) — linear propagation operator (dispersion + loss).
5. **NonlinearRHS** (`NonlinearRHS.jl`) — nonlinear right-hand side; three `Trans*` types: `TransModeAvg`, `TransModal`, `TransRadial`/`TransFree` for different geometries.
6. **RK45** (`RK45.jl`) — adaptive Dormand-Prince integrator.
7. **Output** (`Output.jl`) — `MemoryOutput` (default) and `HDF5Output`.
8. **Interface** (`Interface.jl`) — high-level `prop_capillary` and `prop_gnlse` entry points.

The main `Luna.run(Eω, grid, linop, transform, FT, output)` ties everything together (defined in `src/Luna.jl`).

### Rust backend (`luna-rust/src/`)

| Module | Purpose |
|---|---|
| `dispatch.rs` | Runtime hardware dispatcher: auto-selects CUDA → Vulkan → AVX-512 → AVX2/NEON → portable |
| `ffi.rs` | `#[no_mangle] extern "C"` symbols exposed to Julia via `ccall` |
| `stepper.rs` | Dormand-Prince 5(4) adaptive stepper with FSAL and Lund PI controller |
| `dispersion.rs` | Chebyshev-fit dispersion (`ChebyshevDispersion`) and Sellmeier gas model |
| `diffraction.rs` | Quasi-Discrete Hankel Transform (`Qdht`) for radial free-space propagation |
| `ionization.rs` | PPT ionization rate spline LUT with strict lower-bound cutoff |
| `raman.rs` | Time-domain ADE Raman solver using matrix-exponential integrator |
| `integrator.rs` | 4-point Gauss-Legendre integrating factor quadrature |
| `io.rs` | Dynamic HDF5 I/O (`dlopen` at runtime — no link-time dependency) |
| `scans.rs` | `ScanQueue` for process-safe parameter scans via `flock` |

### Julia ↔ Rust FFI boundary

Julia calls Rust kernels via `ccall` to the `libluna_rust` shared library. The library is built by `deps/build.jl` (via `cargo build --release`) and loaded at Julia package init time. The canonical zero-copy FFI smoke test is `test/test_rust_ffi.jl` (CI group `rust`); additional cross-language integration scripts (stepper/dispatch, scans/HDF5 I/O, GPU) live in `luna-rust/tests/*.jl`. Keep these green — they are the safety net guarding the FFI surface.

#### Kernel wiring pattern (established with PPT ionization)

Each Rust kernel follows this pattern for progressive migration:

1. **Rust side (`ffi.rs`):** export three `#[unsafe(no_mangle)] pub unsafe extern "C"` functions — `init_<kernel>`, `free_<kernel>`, and `<kernel>_evaluate` — using the opaque-handle lifecycle pattern from `init_scan_queue` / `init_simulation_engine`.
2. **Julia side (`src/*.jl`):** add a `mutable struct Rust<Kernel>Handle` with a GC finalizer that calls `free_<kernel>`. The parent struct stores this as a type parameter `RH` (either `Nothing` for Julia path or `Rust<Kernel>Handle` for Rust path). An env-var toggle (`LUNA_USE_RUST_<KERNEL>`) gates construction at runtime — Julia path is always the default.
3. **Test (`test/test_<kernel>_rust.jl`):** `@testitem tags=[:rust]` with a skip-guard (mirrors `test_rust_ffi.jl`). Asserts relative agreement within spline-interpolation precision (~1e-8), not machine epsilon, and asserts exact boundary conditions (below-cutoff → 0.0).

**Current wiring status:**
- PPT ionization (`LUNA_USE_RUST_IONISATION=1`) — `src/Ionisation.jl`, `test/test_ionisation_rust.jl`.
- Time-domain Raman (`LUNA_USE_RUST_RAMAN=1`) — `src/Nonlinear.jl` + `src/Interface.jl`,
  `test/test_raman_rust.jl`. Two wrinkles vs ionization: (a) eligibility check lives in
  `Interface.jl` (included after `Raman.jl`; `Nonlinear.jl` sees no Raman types); (b)
  equivalence tolerance is ~1e-4 (method difference: FFT convolution vs exponential integrator),
  not 1e-8. Only `RamanPolarField` (carrier field) is wired; `RamanPolarEnv` (envelope)
  always uses Julia in this slice. Eligibility: `CombinedRamanResponse` with all-SDO `Rs`
  and density-independent τ2 — `Interface._make_rust_raman_handle_from_response`.
- Zeisberger anti-resonant dispersion (`LUNA_USE_RUST_DISPERSION=1`) — `src/Antiresonant.jl`,
  `test/test_dispersion_rust.jl`. New wrinkles vs Raman: (a) the acceleration seam is
  `neff_β_grid(grid, ::ZeisbergerMode, λ0)` — a per-step batch over all positive-frequency
  grid points (not a per-call transform); (b) Julia still evaluates `nco(ω)`/`ncl(ω)` via
  its own multi-term Sellmeier, then passes the arrays to Rust which runs only the Zeisberger
  geometry (eq. 15) — so equivalence is near machine epsilon (~1e-12), not a method-difference
  tolerance; (c) the const-linop setup path (`make_const_linop`, one-time at init) is
  intentionally left on Julia (negligible cost); (d) `_make_rust_zeisberger_handle` must be
  defined AFTER `struct ZeisbergerMode` in the source file (Julia evaluates top-to-bottom).
  Full parity: HE/EH/TE/TM branches, ϕ wall-thickness phase, σ⁴ real + imaginary loss terms.

See `BACKLOG.md` for remaining kernels and follow-ups.

## Math & Advancements

The `luna-rust` crate replaces several numerical approximations from the original Julia codebase with exact analytical forms and allocation-free solvers. **`luna-rust/README.md` is the authoritative math reference** (with full equations); the summary below is a map:

1. **Waveguide dispersion** (`dispersion.rs`) — instead of evaluating the mode solver 7–11×/step to finite-difference β(ω), the **Zeisberger** model applies the analytical chain rule to the Sellmeier equations, and the **Marcatili** model precomputes β(ω) on a Chebyshev grid (Nc≈64) and differentiates via Chebyshev recurrence (`c'_{n-1} = c'_{n+1} + 2·n·c_n`) to ~1e-16 accuracy with zero runtime finite differences.
2. **Spatial diffraction / QDHT** (`diffraction.rs`) — dense matrix-vector transform optimised as cache-resident Level-2 BLAS (`dgemv`/`zgemv`); at Nr≈128–256 the matrix fits in L1/L2, beating fast-Hankel prefactor overheads.
3. **PPT ionization** (`ionization.rs`) — 1D cubic B-spline LUT over |E| with a strict lower-bound cutoff E_min (returns 0 below, errors above E_max), avoiding ill-conditioned low-field extrapolation.
4. **Raman response** (`raman.rs`) — solves the single-damped-oscillator ADE directly in the time domain via an explicit exponential integrator (`x_{n+1} = A·x_n + B0·I_n + B1·I_{n+1}`), turning O(Nt·log Nt) FFT convolution into O(Nt) in-place, allocation-free, phase-exact stepping.
5. **Integrating-factor propagator** (`integrator.rs`, `stepper.rs`) — integrates the diagonal linear operator L(ω,z) over a step with 4-point Gauss-Legendre quadrature (order-8 local accuracy); since L commutes with itself in frequency space the Magnus expansion truncates to the integral.
6. **Dynamic HDF5 I/O & scan queue** (`io.rs`, `scans.rs`) — locates and `dlopen`s `libhdf5` at runtime (no link-time dependency, `LUNA_HDF5_LIB` override) and serializes parallel parameter sweeps with `libc::flock` (Unix only — see caveat below).

The adaptive **Dormand-Prince 5(4)** stepper (`stepper.rs`) runs in the interaction picture with a Lund PI step controller, FSAL reuse, and pre-allocated stage buffers (zero hot-loop allocation). The **runtime dispatcher** (`dispatch.rs`) cascades GPU (CUDA → Vulkan) → vectorized CPU (AVX-512 / Apple AMX) → standard SIMD (AVX2 / NEON) → portable scalar.

> **Caveat (Windows scans):** the `flock`-based queue lock in `scans.rs` is a no-op on non-Unix targets, so parallel parameter scans are not yet process-safe on Windows. See the `TODO` in `scans.rs`.

## Key Design Points

- **Unicode variable names**: the codebase uses `ω`, `λ`, `β`, `τ`, `ε`, etc. throughout Julia files. Enter them with `\omega<tab>`, `\lambda<tab>`, etc.
- **Dual grid types**: `RealGrid` (carrier-resolved, uses `rfft`) vs. `EnvGrid` (envelope, uses `fft`). The `setup` and `run` functions are overloaded for each grid type.
- **Mode-averaged vs. modal propagation**: `modes=:HE11` (mode-averaged) and `modes=1` (single-mode modal) are **not** equivalent when ionisation is active — mode-averaged treats spatial dependence of the nonlinear polarisation like Kerr, while modal projects it onto each mode.
- **FFTW wisdom**: automatically loaded/saved by `Utils.loadFFTwisdom`/`saveFFTwisdom`. Tests disable this with `:estimate` planning mode.
- **Precompilation run**: `src/Luna.jl` runs two `prop_capillary` calls and one `prop_gnlse` call at module load time to trigger precompilation.

## CI Test Groups

| Group | What it covers |
|---|---|
| `physics` | Unit tests: grid, maths, physdata, modes, fields, linops, ionisation, Raman, processing, scans, … |
| `rust` | Rust `cargo test` + Julia FFI linking (`test_rust_ffi.jl`) |
| `sim-interface` | `~56 prop_capillary` integration calls |
| `sim-multimode` | Multimode + polarisation propagation tests |
| `sim-propagation` | Radial, free-space, GNLSE, Kerr, Raman propagation tests |
| `io` | HDF5 output tests |
| `fields` | Field generation tests |
