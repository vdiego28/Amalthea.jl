# Luna Rust (`amalthea`)

`amalthea` is a high-performance, stateless physical propagation engine rewritten in Rust. It serves as the optimized backend for `Amalthea.jl`, migrating critical physics kernels, solvers, and spatial/spectral grids to Rust 2024. CPU throughput comes from `target-cpu=native` plus LLVM auto-vectorization of the branchless RHS kernels (confirmed via disassembly — see §3 below), not hand-written intrinsics; GPU acceleration (CUDA) is real and opt-in via the native-port's `CudaNativeSim`.

---

## 1. Mathematical Enhancements & Optimizations

The Rust migration replaces several numerical approximations from the original Julia codebase with exact analytical forms and highly stable algorithms:

### 1. Waveguide Dispersion ($\beta_k$ & GVD)
*   **Original (Julia)**: Evaluated the mode solver 7–11 times per propagation step to approximate derivatives of the propagation constant $\beta(\omega, z)$ using finite differences. This introduced grid discretization noise and catastrophic cancellation.
*   **Zeisberger Model (Rust)**: Applies the analytical chain rule directly to the Sellmeier equations. Group velocity dispersion (GVD) is computed via exact derivatives, eliminating numerical grid noise.
*   **Marcatili Model (Rust)**: Precomputes $\beta(\omega)$ on a Chebyshev grid at startup and differentiates using Chebyshev recurrence relations:
    $$c'_{n-1} = c'_{n+1} + 2 n c_n$$
    This achieves machine precision ($\approx 10^{-16}$) using a small grid size ($N_c \approx 64$) with zero runtime finite-difference evaluation.

### 2. Spatial Diffraction (Hankel / QDHT)
*   **Original (Julia)**: Planned and executed Quasi-Discrete Hankel Transforms (QDHT) as a dense matrix-vector multiplication.
*   **Proposed (Rust)**: Retains the dense matrix-vector multiplication but optimizes it via cash-aligned Level-2 BLAS (`dgemv`/`zgemv`). At typical radial grids ($N_r \approx 128 - 256$), the transform matrix fits entirely within L1/L2 caches. This bypasses the prefactor and grid interpolation overheads of Fast Hankel Transforms or Fast Multipole Methods, executing near peak CPU/GPU FLOP/s.

### 3. PPT Ionization Rate Model
*   **Original (Julia)**: Precomputed the rate on a dense grid of field values, converting to log-scale, and interpolated at runtime using cubic splines.
*   **Proposed (Rust)**: Retains the 1D cubic B-spline lookup table (LUT) over the electric field amplitude $|E|$, but enforces a strict lower-bound cutoff $E_{\text{min}}$:
    $$w(|E|) = \begin{cases} 0 & |E| < E_{\text{min}} \\ \exp(S_{\text{spline}}(|E|)) & E_{\text{min}} \le |E| \le E_{\text{max}} \\ \text{Error} & |E| > E_{\text{max}} \end{cases}$$
    Enforcing $E_{\text{min}}$ prevents the spline from extrapolating into ill-conditioned low-field regions where $\gamma \to \infty$, avoiding divisions by zero and floating-point underflows.

### 4. Raman Response (ADE Time-Domain Solver)
*   **Original (Julia)**: Computes the Raman polarization convolution in the frequency domain using a double-grid FFT scheme with zero-padding. This requires multiple FFTs per step and high garbage collection pressure.
*   **Proposed (Rust)**: Solves the Single Damped Oscillator (SDO) Auxiliary Differential Equation (ADE) directly in the time domain using an explicit exponential integrator:
    $$\mathbf{x}_{n+1} = \mathbf{A} \mathbf{x}_n + \mathbf{B}_0 I_n + \mathbf{B}_1 I_{n+1}$$
    This reduces the computational scaling from $O(N_t \log N_t)$ to $O(N_t)$, executes entirely in-place with zero memory allocations, and solves the dissipative damping exactly to avoid phase error.

### 5. Integrating Factor (IF) Propagator
*   **Original (Julia)**: Integrates the $z$-dependent linear operator $L(z)$ using a trapezoidal approximation or assumes constant coefficients.
*   **Proposed (Rust)**: Integrates $L(\omega, z)$ over step $h$ using 4-point Gauss-Legendre quadrature. Since the linear operator is diagonal in frequency space, elements commute $[L(z_1), L(z_2)] = 0$, meaning the Magnus expansion truncates exactly to the first term (the integral). A 4-point quadrature provides order-8 local accuracy, matching the Dormand-Prince step without Magnus commutator overhead.

### 6. Dynamic HDF5 I/O & Queue Serialization (`io.rs`, `scans.rs`)
*   **Original (Julia)**: Depended on `HDF5.jl` package configurations to write formatted datasets and handle file locking via Julia's standard library pidlocks.
*   **Proposed (Rust)**: Avoids compile-time linking requirements by dynamically locating and loading `libhdf5.so` at runtime from the system or Julia's artifact cache. File locking for queued sweeps is implemented via `libc::flock` on Unix and Win32 `LockFileEx`/`UnlockFileEx` on Windows, enabling process-safe parameter scans without external dependencies.
    > The Windows `LockFileEx` path is validated on real Windows CI (`test_scan_queue_flock`, `rust` group's `windows-2025-vs2026` matrix entry) — see `BACKLOG.md`'s "Windows scan-lock validation" entry.

### 7. Group-Velocity β1(z) for Graded-Core (Pressure-Gradient) Waveguides
*   **Original (Julia)**: For a z-dependent (pressure-gradient) capillary, `β1(z)` — the group-velocity term rebuilt at every propagation step — is computed with an *adaptive* finite difference (`FiniteDifferences.central_fdm`), which has a real, repeatable ~1e-12 relative error against the true derivative (the truncation/rounding step-size tradeoff never fully closes at `Float64` precision).
*   **Rust**: `εco(ω;z)-1` is separable into a fixed per-ω gas term times a scalar `dens(z)`, and the waveguide term `nwg(ω)` is z-independent, so the chain rule collapses `β1(z)` to a closed form in 4 constants (`γ0, dγ0, nwg0, dnwg0`) computed once — via `Maths.derivative` fed a `BigFloat` argument, not hand-derived per-material symbolics, so it works for any gas/glass Sellmeier closure unmodified. This makes Rust's `β1(z)` *more* accurate than Julia's own value, at the cost of a small, deliberate, and fully characterized divergence from the Julia oracle (see `docs/dev/native-port/BETA1_ANALYTIC.md` for the derivation, verification, and the resulting tolerance tradeoff).

---

## 2. Solver Architecture (`stepper.rs`)

`amalthea` implements an adaptive **Dormand-Prince 5(4) Stepper** operating in the interaction picture:
*   **Vectorized Integrating Factor (IF) Application**: Linear Integrating Factor propagation is applied to entire vectors outside the stage loops rather than element-by-element, reducing $3 \times \text{size}$ operations to single cache-aligned vector evaluations.
*   **Lund PI Controller**: Controls adaptive stepping using error history to prevent step size oscillations and minimize rejected steps.
*   **FSAL (First Same As Last)**: Reuses the final derivative evaluation of an accepted step as the first evaluation of the next step, reducing the number of RHS evaluations per step.
*   **Zero-Allocation Design**: All intermediate stage buffers are pre-allocated in the `Dopri5Stepper` struct, ensuring zero runtime allocations in the hot propagation loop.

---

## 3. Hardware Paths: What's Real vs. `dispatch.rs`

`dispatch.rs` defines a `HardwarePath`/`SimulationEngine` enum (`Auto`, `CpuX86AVX512`, `CpuX86AVX2`, `CpuArmNeon`, `CpuArmAMX`, `GpuCuda`, `GpuVulkan`, `CpuPortable`) that *detects* available hardware features (`std::is_x86_feature_detected!("avx512f")`, Vulkan `dlopen` probing, etc.) and records which one was picked. **It is not wired into any RHS kernel or the real GPU path** — grep the crate and the only callers are its own unit tests (`lib.rs`). Investigated directly (`docs/dev/BACKLOG.md` S5.2, S1 item 4) before relying on it for anything:

- **CPU vectorization is real, but compiler-driven, not `dispatch.rs`-driven.** `.cargo/config.toml` sets `target-cpu=native`; LLVM auto-vectorizes the branchless elementwise RHS loops (Kerr, FFT scaling, windowing, noise) into AVX2/AVX-512 `%ymm`/`%zmm` instructions for whatever CPU the binary is actually compiled on — confirmed via `objdump`/`nm` on the release `.so`, not assumed. No runtime branch selects between hand-written AVX2/AVX-512/NEON kernel variants; there aren't any, except one:
- **The one genuine hand-written SIMD lane is `raman.rs::solve_avx2`**, because the Raman ADE update is a sequential recurrence (`x_{n+1}` depends on `x_n`) that LLVM's auto-vectorizer structurally cannot parallelize — the opposite situation from the flat Kerr/window broadcasts above.
- **GPU acceleration is real and does run**, but through the native-port's own `CudaNativeSim`/`cuda.rs` (`init_gpu_context`, opt-in via `AMALTHEA_USE_RUST_CUDA_NATIVE=1`), which never touches `dispatch.rs` either. Currently scoped to mode-averaged Kerr(+PPT plasma); measured on real hardware (RTX 5060 Ti) to be **~20-30x slower than the CPU path when plasma is active** (single-thread sequential cumtrapz kernels, a documented V1 tradeoff — BACKLOG S3 item 4/6) but **~5-27x faster for Kerr-only above n≈16k grid points** (FFT-dominated, embarrassingly parallel). `AMALTHEA_NATIVE_GPU=off/on/auto` (Julia `Config.jl`) dispatches on exactly this measured split: `auto` (default) uses GPU only for a plasma-free config above the measured threshold, `on` forces GPU unconditionally (the old behavior), `off` forces CPU — see `RK45._GPU_KERR_ONLY_N_THRESHOLD`'s docstring for the full measured table (BACKLOG S3 item 3).

If you're looking for "the SIMD dispatcher," there isn't a real one — the compiler is already doing that job for the parallel case, and `raman.rs::solve_avx2` is the one place a human had to.

---

## 4. Compilation & Verification

### Build the Shared Library
To compile the optimized C-compatible library for FFI:
```bash
cargo build --release
```
This produces `target/release/libamalthea.so` (Linux), `libamalthea.dylib` (macOS), or `amalthea.dll` (Windows).

### Running Unit Tests
To run the internal Rust test suite (covering Chebyshev fit, QDHT orthogonality, PPT ionization rate lower cutoff, Raman ADE, Integrating Factor quadrature, Dormand-Prince stepper, and hardware dispatcher):
```bash
cargo test
```

### Running Julia FFI Integration Tests
To run the cross-language FFI verification test ensuring zero-copy memory transfers and solver initialization:
```julia
# Run FFI grid modification test
julia tests/test_julia_ffi.jl

# Run Stepper and Hardware dispatcher FFI test
julia tests/test_stepper_dispatch.jl

# Run Scans and HDF5 IO Integration FFI test
julia --project tests/test_scans_io.jl
```
