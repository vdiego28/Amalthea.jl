# Changelog

All notable changes to Amalthea.jl are documented here. This project is a
fork of [Luna.jl](https://github.com/LupoLab/Luna.jl); versions below are
this fork's own, starting from the point the Rust backend was introduced.

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

First stable release. Amalthea.jl keeps Luna.jl's Julia interface fully
backwards-compatible while replacing performance-critical numerical kernels
with a native Rust backend (`luna-rust`), called transparently via `ccall` —
no Rust knowledge is required to use the package.

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
- **Runtime hardware dispatch**: automatic selection of CUDA → Vulkan →
  AVX-512/Apple AMX → AVX2/NEON → portable scalar, including an opt-in
  GPU-resident backend (`AMALTHEA_USE_RUST_CUDA_NATIVE=1`).
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
