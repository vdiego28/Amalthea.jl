# Luna Rust (`luna-rust`)

`luna-rust` is a high-performance, stateless physical propagation engine rewritten in Rust. It serves as the optimized backend for `Luna.jl`, migrating critical physics kernels, solvers, and spatial/spectral grids to Rust 2024 to exploit hardware vectorization (AVX-512, Apple AMX) and GPU acceleration (CUDA, Vulkan).

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
*   **Proposed (Rust)**: Avoids compile-time linking requirements by dynamically locating and loading `libhdf5.so` at runtime from the system or Julia's artifact cache. File locking for queued sweeps is implemented via lightweight `libc::flock` system calls, enabling process-safe parameter scans without external dependencies.

---

## 2. Solver Architecture (`stepper.rs`)

`luna-rust` implements an adaptive **Dormand-Prince 5(4) Stepper** operating in the interaction picture:
*   **Vectorized Integrating Factor (IF) Application**: Linear Integrating Factor propagation is applied to entire vectors outside the stage loops rather than element-by-element, reducing $3 \times \text{size}$ operations to single cache-aligned vector evaluations.
*   **Lund PI Controller**: Controls adaptive stepping using error history to prevent step size oscillations and minimize rejected steps.
*   **FSAL (First Same As Last)**: Reuses the final derivative evaluation of an accepted step as the first evaluation of the next step, reducing the number of RHS evaluations per step.
*   **Zero-Allocation Design**: All intermediate stage buffers are pre-allocated in the `Dopri5Stepper` struct, ensuring zero runtime allocations in the hot propagation loop.

---

## 3. Multi-Branch Hardware Dispatcher (`dispatch.rs`)

To achieve maximum performance across platforms, the engine implements a runtime dispatcher that automatically routes calculations to the most efficient branch available, with seamless fallbacks:

### Fallback Priority Cascade:
1.  **GPU Branch (CUDA/Vulkan)**: Checks for CUDA drivers (`libcuda.so` / `/dev/nvidia0` on Linux). If CUDA is unavailable, queries Vulkan driver libraries. Offloads FFTs, Hankel transforms, and time-domain convolutions to GPU VRAM.
2.  **Vectorized CPU Branch (AVX-512 / Apple AMX)**: If a CPU architecture is detected, uses `std::is_x86_feature_detected!("avx512f")` (on x86) or Apple Silicon vendor checks to leverage 512-bit registers or AMX matrix units.
3.  **Standard CPU Vectorization (AVX2 / NEON)**: Targets standard vector pipelines (AVX2 on x86, NEON on ARM).
4.  **CpuPortable**: A generic, highly-optimized scalar fallback loop.

Users can manually force a specific path using the FFI initialization code or let the engine auto-detect.

---

## 4. Compilation & Verification

### Build the Shared Library
To compile the optimized C-compatible library for FFI:
```bash
cargo build --release
```
This produces `target/release/libluna_rust.so` (Linux), `libluna_rust.dylib` (macOS), or `luna_rust.dll` (Windows).

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
