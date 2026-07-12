# Changelog

All notable changes to Luna-Rust.jl are documented here. This project is a
fork of [Luna.jl](https://github.com/LupoLab/Luna.jl); versions below are
this fork's own, starting from the point the Rust backend was introduced.

## [1.0.0]

First stable release. Luna-Rust.jl keeps Luna.jl's Julia interface fully
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
  GPU-resident backend (`LUNA_USE_RUST_CUDA_NATIVE=1`).
- **Per-kernel Rust acceleration** (opt-in via `LUNA_USE_RUST_*` toggles,
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
- Package renamed to Luna-Rust.jl; citation metadata updated (Zenodo DOI).

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
