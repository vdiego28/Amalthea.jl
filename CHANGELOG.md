# Changelog

All notable changes to Amalthea.jl are documented here. This project is a
fork of [Luna.jl](https://github.com/LupoLab/Luna.jl); versions below are
this fork's own, starting from the point the Rust backend was introduced.

## [Unreleased]

### Fixed
- Repair `locextrap=false` across Julia, legacy Rust, resident CPU, and CUDA:
  evaluate the starting derivative at the accepted fourth-order state, and
  make fourth-order dense output reach that state continuously. Default
  fifth-order propagation is unchanged.
- Reconcile release and GPU-support documentation; maintain current work and
  release status in the backlog's resume queue.

## [1.0.4]

CPU performance, Apple Silicon, and concurrency update for the native backend.

### Added
- Added AVX2 and AArch64 NEON Raman ADE recurrence kernels with scalar-tail and
  scalar-oracle parity tests.
- Added scratch-isolated Julia `TransModal` callback batching for recognized
  standard Kerr, Julia plasma, and Julia Raman responses, with conservative
  sequential fallback for arbitrary/stateful closures and legacy Rust handles.
- Added a 5–10 minute Apple Silicon JSON/Markdown diagnostic covering Raman,
  radial QDHT, modal threading, two-worker scans, and a safely restored
  host-native thin-LTO comparison build.

### Changed
- Resident radial QDHT now initializes Julia's configured BLAS provider
  directly and selects BLAS automatically above the measured workload
  threshold. `AMALTHEA_QDHT_BLAS` accepts `off`/`auto`/`on` (`0`/`1` aliases);
  deterministic mode still forces Rayon.
- `QueueExec` now uses stable cache-directory queue names, local queue state,
  guaranteed worker cleanup, and `threads_per_worker=1` by default to bound
  Julia, FFTW, and resident Rayon concurrency.
- Native step attempts reuse the dense-output left-endpoint buffer and avoid
  redundant field synchronization for the built-in no-op callback.
- Corrected public documentation, authorship, compatibility, registry, and
  historical hardware-dispatch claims after the v1.0.3 release audit.
- Added a reproducible, equivalence-checked Julia-oracle versus resident-native
  CPU benchmark and published its non-speedup result without extrapolating to
  other workloads or hardware.
- Updated `CITATION.cff` to CFF 1.2.0 with complete authorship, release,
  repository, documentation, and Zenodo metadata; corrected README citation
  links that previously pointed to the v0.7.0 record instead of the Amalthea
  concept DOI or current v1.0.3 record.
- Published the historical registry, compatibility, and hardware-dispatch
  correction on the Zenodo v1.0.0 metadata record without changing its DOI or
  archived release files.

### Fixed
- Corrected the Dormand--Prince propagated solution across Julia, legacy Rust,
  resident CPU, and CUDA steppers: the default is now the true fifth-order
  rule and `locextrap=false` explicitly selects the embedded fourth-order
  rule instead of retaining a post-RHS internal stage.
- Removed Rust stage-field clones while preserving rejected-step,
  `locextrap=false`, dense-output, callback-window, and panic-restoration
  semantics.
- Corrected the Cartesian Julia `TransModal` upper-bound check to test the
  second coordinate.

## [1.0.3]

GPU geometry expansion and portable-installation update for the native backend.

### Added
- Explicit CUDA-resident support for radial `RealGrid`/`EnvGrid` SDO Raman,
  modal `RealGrid`/`EnvGrid` Kerr, modal `RealGrid` SDO Raman, and free-space
  `RealGrid`/`EnvGrid` Kerr.
- Free-space `RealGrid` CUDA support for PPT plasma, thresholded ADK, and SDO
  Raman, with independent resident state for every transverse time series.
- CPU-only Linux ARM64 release binaries and a native ARM64 package-build/FFI
  smoke job in the standing test workflow.
- A cross-platform installation and configuration guide covering Linux,
  macOS, Windows, ARM, source fallbacks, and optional CUDA builds.

### Changed
- Package and release builds now default to CPU-only operation and never
  require CUDA or probe `nvcc`; CUDA is enabled explicitly with
  `AMALTHEA_CUDA_BUILD=required`.
- Prebuilt selection now matches the exact OS/architecture pair. Unsupported
  platforms compile from source instead of receiving a mismatched binary.
- CUDA radial, modal, and free-space additions remain explicit-on while their
  production-shaped automatic-dispatch thresholds are unmeasured.

### Fixed
- Preserved both retained spectral halves in oversampled modal and free-space
  `EnvGrid` CUDA transforms.
- Released the free-space c2c cuFFT plan during final CUDA teardown.
- Serialized cold-depot Julia worker precompilation before parallel CI bucket
  startup, preventing shared-cache bootstrap races.
- Added an actionable diagnostic when CUDA is requested from a CPU-only native
  library.

## [1.0.2]

Correctness, safety, and GPU-physics update for the native backend.

### Added
- Thresholded ADK ionisation on the GPU-resident mode-averaged `RealGrid`
  path, with automatic dispatch from the measured `n = 8193` threshold.
- Strict required-CUDA build policy that rejects missing, dummy, or invalid
  PTX, together with transactional CUDA setup and rollback coverage.
- Stronger native-stepper regressions for non-default error norms, local
  extrapolation, rejected steps, and ADK non-vacuity.
- Item-level CI scheduling with balanced worker assignments, one-minute live
  heartbeats, and complete console-safe worker logs on failure.

### Changed
- Unsupported custom RK45 norms now route to the Julia oracle instead of
  silently using the Rust backend's `weaknorm` implementation.
- GitHub Actions jobs use read-only permissions by default; write access is
  isolated to documentation, benchmark, release, and upstream-issue jobs.
- CUDA ADK automatic dispatch requires the exact first measured passing size,
  `8193`; manually constructed `threshold=false` ADK remains on the CPU path.
- CUDA PPT fraction/current/polarisation integration now uses parallel prefix
  scans instead of the former serial device bottleneck.

### Fixed
- Corrected `locextrap=false` on legacy, CPU-resident, and CUDA-resident RK45
  paths so accepted and rejected steps use the actual fourth-order trial.
- Corrected CUDA adaptive error control to use the same global weak norm and
  pre-acceptance trial state as the CPU/Julia implementations.
- Hardened `native_step` pointer and panic handling, mode-averaged buffer
  contracts, and CUDA reconfiguration against partial setup failures.
- Removed project-owned CI and documentation warnings and made the CUDA kernel
  symbol failure paths release their partially initialized resources.
- Made the parallel test scheduler robust to Windows UTF-8, CRLF, and
  console-encoding behavior while preserving actionable failure diagnostics.

## [1.0.1]

Correctness, compatibility, and performance update for the native backend.

### Added
- Order-5 dense output shared by the Julia and resident-Rust steppers.
- Native support for multi-mode Zeisberger/Vincetti fibres, radial `EnvGrid`
  Raman, modal/free-space gas mixtures, additional high-level mode entry
  points, and opt-in native HDF5 scan-point writes.
- Problem-size-aware GPU dispatch and substantially stronger CUDA
  nonlinearity regression coverage on real hardware.
- Python integration tests and expanded Python examples/documentation.

### Changed
- Release assets now use the canonical `libamalthea-<triple>` names for Linux
  x86_64, Apple Silicon, and Windows x86_64.
- Native modal and free-space workloads use the validated threaded paths;
  FFT-based Raman convolution uses real-to-complex transforms.
- Source checkouts always compile their matching Rust library instead of
  downloading a binary from the last tagged release.

### Fixed
- Restored the CUDA-resident nonlinear contribution and seeded its first RK
  stage correctly.
- Corrected the FSAL carry and dense-output implementation, modal
  two-polarisation plasma example construction, prebuilt-asset compatibility,
  and several low-level examples.
- Stabilized macOS CI by using one FFTW thread in its test harness while
  retaining Julia thread coverage.

## [1.0.0]

> **Historical accuracy note (added 2026-08-10):** the original v1.0.0 text
> overstated both compatibility and hardware dispatch. Amalthea intends to
> retain Luna-compatible APIs and tests its Julia/native paths for numerical
> equivalence, but a moving hard fork cannot promise unconditional backwards
> compatibility. The `dispatch.rs` CUDA/Vulkan/CPU cascade described below
> was detection-only and was not wired into propagation; no Vulkan backend
> existed. The v1.0.0 Zenodo/GitHub prose also incorrectly said Amalthea was
> registered in Julia General. It was not and remains installed from GitHub.

First stable release. Amalthea.jl retained Luna.jl's high-level Julia
interface while replacing performance-critical numerical kernels with a
native Rust backend (`luna-rust`), called transparently via `ccall` — no Rust
knowledge is required to use the package.

### Added
- **Native-Rust resident stepper** (`RustNativeStepper` / `NativeSim`): the
  entire RK45 hot loop — field, RK scratch buffers, and FFTW plans — lives in
  Rust for the duration of a `solve`, eliminating the per-stage Julia
  callback round-trip. Covers mode-averaged, radial, modal, and free-space
  geometries; `RealGrid` and `EnvGrid`; Kerr, plasma (PPT/ADK), and Raman
  (including `:SiO2` intermediate-broadening) nonlinearities; gas mixtures;
  z-dependent (graded-core, tapered, multi-point gradient) linear operators;
  and shot noise. Falls back to the Julia stepper automatically for any
  configuration outside this scope (`NativeIneligible`).
- **Runtime hardware detection (historical correction)**: `dispatch.rs`
  detected CUDA, Vulkan, AVX-512/Apple AMX, AVX2/NEON, and portable-scalar
  capabilities, but this cascade did not dispatch propagation and there was
  no Vulkan implementation. CUDA-resident propagation was a separate,
  explicitly opt-in path (`AMALTHEA_USE_RUST_CUDA_NATIVE=1`).
- **Per-kernel Rust acceleration** (opt-in via `AMALTHEA_USE_RUST_*` toggles,
  used independently of the resident stepper): PPT ionisation rate,
  time-domain Raman (ADE exponential integrator), Zeisberger/Marcatili
  dispersion, and QDHT batch transforms.
- **Python bindings** (`python/`, `juliacall`-based, pip-installable) with a
  `numpy`-backed output wrapper and ASCII/Unicode keyword translation
  (`lambda0` ↔ `λ0`, etc.).
- **Prebuilt binary releases**: tagged releases publish `libluna_rust` for
  Linux, macOS, and Windows; `deps/build.jl` downloads the matching prebuilt
  library for the installed version before falling back to a local
  `cargo build`.
- Cross-platform scan-queue locking (`flock` on Unix, `LockFileEx` on
  Windows), validated by CI on all three platforms.

### Changed
- Package renamed to Amalthea.jl (previously Luna-Rust.jl, itself a fork of
  Luna.jl); citation metadata updated (Zenodo DOI), new package UUID minted.

### Fixed
- `Polarisation.ellipse` angle calculation (was always 0; now
  `angle(Q + 1im*U)/2`), and other fork-vs-upstream parity fixes — see
  `docs/dev/REVIEW.md` for the full audit and `docs/dev/BACKLOG.md`'s
  "Phase A/B" entries for what was ported back from upstream.
- Shell/command-injection hardening in `Scans.jl`'s SSH/Slurm/Condor
  submission paths (`Cmd` arrays instead of interpolated shell strings).

See [`docs/dev/BACKLOG.md`](docs/dev/BACKLOG.md) and
[`docs/dev/native-port/PORT_LOG.md`](docs/dev/native-port/PORT_LOG.md) for
the full phase-by-phase development history.
