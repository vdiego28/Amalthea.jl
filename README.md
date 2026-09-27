# Amalthea.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20359892.svg)](https://doi.org/10.5281/zenodo.20359892)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://vdiego28.github.io/Amalthea.jl/stable/)
[![CI](https://github.com/vdiego28/Amalthea.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/vdiego28/Amalthea.jl/actions/workflows/run_tests.yml)

> [!IMPORTANT]
> **Amalthea.jl** is a performance-engineering fork of [Luna.jl](https://github.com/LupoLab/Luna.jl) that provides a native Rust backend (`amalthea`) for numerical kernels. Its retained Julia interfaces are intended to remain API-compatible with Luna.jl, with compatibility checked by Julia-oracle/native equivalence tests—not promised unconditionally as the hard fork evolves.

Amalthea.jl is a flexible platform for the simulation of nonlinear optical dynamics—both in waveguides (such as optical fibres) and free-space geometries—using the unidirectional pulse propagation equation (UPPE) and its approximate forms, such as the commonly used generalised nonlinear Schrödinger equation (GNLSE). Some of the key features of Amalthea.jl:

- A variety of propagation geometries treated in a unified way:
    - Single-mode (mode-averaged) propagation in waveguides
    - Multi-mode propagation in waveguides with arbitrary (including non-symmetric) mode-shapes, full polarisation resolution, and intermodal coupling for arbitrary nonlinear polarisation terms
    - Waveguides with arbitrarily varying material properties and cross-sections (e.g. tapered fibres)
    - Free-space propagation with radial symmetry
    - Full (3+1)-dimensional free-space propagation
- Both field-resolved and envelope propagation equations
- A range of linear and nonlinear optical effects:
    - Modal dispersion and loss in waveguides
    - Optical Kerr effect (third-order nonlinearity)
    - Raman scattering in molecular gases or glasses
    - Strong-field photoionisation and plasma dynamics
- A built-in interface for the running and processing of multi-dimensional [parameter scans](#running-parameter-scans) in serial or parallel
- A standard library of [plotting](#plotting-results) and [processing](#output-processing) functions, including calculation of spectrograms and beam properties
- **Rust-resident propagation backend** with LLVM auto-vectorized CPU kernels
  and an experimental, explicitly opt-in CUDA backend. The legacy
  `dispatch.rs` AVX/CUDA/Vulkan selector is detection-only and is not wired
  into propagation.

Amalthea.jl is designed to be extensible: adding e.g. a new type of waveguide or a new nonlinear effect is straightforward, even without editing the main source code.

Amalthea.jl was originally derived from a codebase developed for modelling ultrafast pulse propagation in gas-filled hollow capillary fibres and hollow-core photonic crystal fibres. It is also excellent for modelling propagation in solid-core fibres.

Amalthea.jl is written in the [Julia programming language](https://julialang.org/), chosen for its unique combination of readability, ease of use, and speed, and accelerated by [Rust](https://www.rust-lang.org/) for maximum performance. If you want to use Amalthea.jl but are new to Julia, see [the relevant section of this README](#new-to-julia).

There are two ways of using Amalthea.jl:
1. A very simple high-level interface for the most heavily optimised applications: propagation in gas-filled hollow capillary fibres and hollow-core photonic crystal fibres (consisting of the function [`prop_capillary`](#quickstart) and some helper functions to create input pulses); or propagation of simple GNLSE simulations (consisting of the function [`prop_gnlse`](#gnlse)).
2. A low-level interface which allows for full control and customisation of the simulation parameters, the use of custom waveguide modes and gas fills (including gas mixtures), and free-space propagation simulations.

For a short introduction on how to use the simple interface, see the [Quickstart](#quickstart) or [GNLSE](#gnlse) sections below. More information, including on the internals of Amalthea.jl, can be found in the [Documentation](https://vdiego28.github.io/Amalthea.jl/stable/).

## Relationship to Luna.jl

Amalthea.jl is an independent hard fork of [Luna.jl](https://github.com/LupoLab/Luna.jl), not a set of changes intended to land upstream. The Julia-level API, physics models, and much of the original interface layer come directly from that project; what Amalthea.jl adds is a from-scratch Rust numerical backend (`amalthea/`) that the compute-critical kernels are offloaded to, plus a resident native-Rust stepper that removes the per-step Julia↔Rust callback round-trip entirely (see [`docs/dev/native-port/ARCHITECTURE.md`](docs/dev/native-port/ARCHITECTURE.md)).

Compatibility is tested against the fork's retained Julia implementation,
which serves as the Luna-compatible numerical oracle. The hosted matrix runs
on Linux, macOS, and Windows with Julia LTS/current/pre-release and covers
mode-averaged, radial, modal, and free-space propagation on `RealGrid` and
`EnvGrid`, including single-step and full-solve Julia/native comparisons.
Amalthea does not continuously test against the tip of upstream Luna.jl, so
this is a tested compatibility intention rather than a claim that every
current or future Luna API behaves identically.

Two things worth being explicit about:

- **Why a hard fork and not a PR series.** Luna.jl is still maintained by the Lupo Lab, but it isn't set up to take in a change of this scope (a parallel native backend plus a new build/FFI toolchain) through its normal contribution path. Diverging as a fork was the practical way to pursue this direction without blocking on upstream review bandwidth for a change this large.
- **What stays intact.** The MIT license and copyright notice from the original authors (Chris Brahms and John Travers) are preserved in [`LICENSE`](LICENSE), as required by that license, and they're credited in [`CITATION.cff`](CITATION.cff) and the [Credits & Acknowledgements](#credits--acknowledgements) section below. If you're citing this software, please cite both the original Luna.jl work and this fork — see [Citing](#citing).

If you're deciding which one to use: Luna.jl is the actively-maintained, PR-accepting original; Amalthea.jl trades that contribution model for raw throughput on the numerical hot path. They're expected to diverge further over time rather than reconverge.

## The Rust Backend (`amalthea`)

The `amalthea` crate provides the high-performance numerical engine that powers the most compute-intensive parts of the simulation. Key features include:

- **Resident CPU propagation**: eligible simulations use the native CPU
  backend by default. LLVM performs target-appropriate auto-vectorization;
  the CUDA-resident backend is opt-in, with hardware-verified slices across
  mode-averaged, radial, modal, and free-space geometries. Supported grids,
  responses, and dispatch restrictions are listed in the
  [support matrix](docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md#cross-cutting-notes).
  Standing GPU CI remains deferred. `dispatch.rs` detects hardware for its own tests but is not a
  propagation dispatcher, and there is no Vulkan implementation.
- **Parallelised transforms**: resident radial QDHT automatically uses Julia's
  configured BLAS provider for production-sized batches, with a Rayon fallback
  for small or deterministic work.
- **Raman solver**: the time-domain matrix-exponential recurrence has real
  AVX2 (`x86_64`) and NEON (Apple/Linux AArch64) kernels plus a scalar portable
  fallback; oscillator accumulation order is preserved.
- **Controlled concurrency**: standard Julia `TransModal` responses use
  scratch-isolated callback batching, while process-parallel scans default to
  one Julia/Rayon/FFTW thread per worker to avoid oversubscription.
- **Cross-platform**: builds and runs on Linux, macOS, and Windows, including
  Linux ARM64 and Apple Silicon. CUDA is optional; ordinary installations use
  the resident CPU backend and do not need an NVIDIA GPU or CUDA toolkit. The
  `Scans.jl`/`QueueExec` file-locking path also supports Windows through
  `LockFileEx`/`UnlockFileEx`, exercised by CI on every push
  (`windows-2025-vs2026` runner). See `docs/dev/BACKLOG.md`'s "Windows
  scan-lock validation" entry.

The Rust backend is called transparently via Julia's `ccall` interface; no Rust knowledge is needed to use Amalthea.jl.

## Performance and reproducible comparison

The native backend is a performance-engineering project, not a guarantee that
every workload is faster on every machine. The public comparison uses the
same Amalthea checkout twice: once through the retained Luna-compatible Julia
oracle and once through the resident native CPU backend. This isolates backend
cost without conflating it with dependency or API differences from a separate
upstream Luna.jl installation.

For the README quickstart workload on 2026-08-10, the native path was slightly
slower on the measured host:

| Version and host | Julia oracle | Native CPU | Julia/native ratio | Final-field relative error |
|---|---:|---:|---:|---:|
| `v1.0.3` (`65489dd`), Linux `x86_64`, AMD Zen 3 (`znver3`), Julia 1.12.6, one thread | 0.986 s | 1.113 s | 0.885× | 1.66e-9 |

These are medians of five warmed complete solves for a 125 μm × 3 m helium
capillary at 1 bar, 800 nm, 10 fs, and 120 μJ, with Kerr and PPT plasma on and
Raman/shot noise off. The result is reported precisely because it does not
support a blanket speedup claim. Hardware and simulation shape matter; run the
comparison for your own target:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
AMALTHEA_BENCHMARK_TRIALS=5 \
julia --startup-file=no --project test/benchmark_julia_vs_native.jl
```

The script prints every sample and refuses to report a ratio unless the two
final fields agree within the repository's `1e-6` full-solve tier. The
[GitHub Pages root](https://vdiego28.github.io/Amalthea.jl/) is a separate
native-step regression dashboard; it tracks changes over commits and is not a
Luna-versus-Amalthea comparison.

The focused 2026-08-24 optimization audit on the same Linux Zen 3 host found
30–48% lower native fixed-step medians and 31–50% lower adaptive-solve medians
across five affected radial/rotational-Raman/shot-noise fixtures. Per-step
Julia-visible allocations fell from 16 KB–787 KB to 96 bytes. These are
pinned-core, ten-sample component results—not a replacement for the complete
workload benchmark above. Exact commands and JSON interpretation are in
[`test/performance_audit/README.md`](test/performance_audit/README.md).

## Installation

Amalthea.jl requires Julia 1.10 or newer. It is not yet registered in Julia's
General registry, so install the latest tagged release directly from GitHub
(currently `v1.0.4`):

```julia
pkg> add https://github.com/vdiego28/Amalthea.jl#v1.0.4
```

Check the [Releases page](https://github.com/vdiego28/Amalthea.jl/releases)
and substitute a newer stable tag when available.

On Linux or macOS, the equivalent one-command installer provisions Julia via
Juliaup when needed and can be rerun to update Amalthea:

```bash
curl -fsSL https://raw.githubusercontent.com/vdiego28/Amalthea.jl/main/install.sh | bash
```

The normal installation is CPU-only and requires neither CUDA nor an NVIDIA
GPU. The installer downloads a checksum-verified native library when the
release contains one, otherwise it compiles from source with Rust 1.85 or
newer:

| Platform | Release binary |
|---|---|
| Linux `x86_64` | Yes |
| Linux ARM64/AArch64 | Yes |
| macOS Apple Silicon | Yes |
| Windows `x86_64` | Yes |
| Intel macOS, Windows ARM64, ARM32, musl Linux, other systems | Source build; not release-tested |

Platforms without a release binary build from source and therefore require
[rustup](https://rustup.rs/). A git checkout or the `main` development branch
also always builds from source:

```julia
using Pkg
Pkg.add(url="https://github.com/vdiego28/Amalthea.jl", rev="main")
```

To compile and enable the experimental CUDA backend, build in a fresh Julia
process and restart Julia afterward:

```julia
ENV["AMALTHEA_CUDA_BUILD"] = "required"
using Pkg
Pkg.build("Amalthea")
```

```julia
ENV["AMALTHEA_USE_RUST_CUDA_NATIVE"] = "1"
ENV["AMALTHEA_NATIVE_GPU"] = "on"
using Amalthea
```

See the complete [installation and configuration guide](docs/src/installation.md)
for Linux, macOS, Windows, ARM, source prerequisites, CPU/CUDA switching,
environment-variable syntax, updating, verification, and troubleshooting.

On Apple Silicon, leave `AMALTHEA_QDHT_BLAS=auto` so radial simulations use
Julia's configured provider (including Accelerate when Julia is configured for
it). Use process-parallel scans with the default `threads_per_worker=1`, then
run the 5–10 minute Apple diagnostic before tuning process/thread topology:

```bash
python3 test/performance_audit/run_apple_quick_test.py
```

It records the M-chip topology, Julia/Rust/FFTW/BLAS providers, 1/2/4-thread
results, modal and scan correctness, and a portable-versus-host-native thin-LTO
diagnostic. The normal portable library is restored even if the diagnostic
fails. Thin LTO is not enabled in production by this runner.

## Quickstart

To run a simple simulation of ultrafast pulse propagation in a gas-filled hollow capillary fibre, you can use `prop_capillary`. As an example, take a 3-metre length of HCF with 125 μm core radius, filled with 1 bar of helium gas, and driving pulses centred at 800 nm wavelength with 120 μJ of energy and 10 fs duration. We consider a frequency grid which spans from 120 nm to 4 μm and a time window of 1 ps.
```julia
julia> using Amalthea
julia> output = prop_capillary(125e-6, 3, :He, 1; λ0=800e-9, energy=120e-6, τfwhm=10e-15, λlims=(150e-9, 4e-6), trange=1e-12)
```
The first time you run this code, you will see the precompilation message:
```julia
julia> using Amalthea
[ Info: Precompiling Amalthea [2a0a82e6-4dc7-4219-a2c1-d2369ab6895d]
```
This will take some time to complete (and you may see additional precompilation messages for the packages Amalthea depends on), but is only necessary once, unless you update Amalthea.jl or edit the package source code. Since this is using the default options including FFT planning and caching of the PPT ionisation rate, you will also have to wait for those processes to finish. After the simulation finally runs (which for this example should take between 10 seconds and one minute), you will have the results stored in `output`:
```julia
julia> output = prop_capillary(125e-6, 3, :He, 1; λ0=800e-9, energy=120e-6, τfwhm=10e-15, λlims=(150e-9, 4e-6), trange=1e-12)
[...]
MemoryOutput["simulation_type", "dumps", "meta", "Eω", "grid", "stats", "z"]
```
You can access the results by indexing into `output` like a `Dict`, for example for the frequency-domain field `Eω`:
```julia
julia> output["Eω"]
8193×201 Array{Complex{Float64},2}:
[...]
```
The shape of this array is `(Nω x Nz)` where `Nω` is the number of frequency samples and `Nz` is the number of steps that were saved during the propagation. By default, `prop_capillary` will solve the full-field (carrier-resolved) UPPE. In this case, the numerical Fourier transforms are done using `rfft`, so the number of frequency samples is `(Nt/2 + 1)` with `Nt` the number of samples in the time domain.

### Multi-mode propagation
`prop_capillary` accepts many keyword arguments (for a full list see the [documentation](https://vdiego28.github.io/Amalthea.jl/stable/interface.html)) to customise the simulation parameters and input pulse. One of the most important is `modes`, which defines whether mode-averaged or multi-mode propagation is used, and which modes are included. By default, `prop_capillary` considers mode-averaged propagation in the fundamental (HE₁₁) mode of the capillary, which is fast and simple but less accurate, especially at high intensity when self-focusing and photoionisation play important roles in the propagation dynamics.

Mode-averaged propagation is activated using `modes=:HE11` (the default) or replacing the `:HE11` with a different mode designation (for mode-averaged propagation in a different mode). To run the same simulation as above with the first four modes (HE₁₁ to HE₁₄) of the capillary, set `modes` to `4` (this example also uses smaller time and frequency windows to make the simulation run a little faster):
```julia
julia> prop_capillary(125e-6, 3, :He, 1; λ0=800e-9, modes=4, energy=120e-6, τfwhm=10e-15, trange=400e-15, λlims=(150e-9, 4e-6))
```
The propagation will take much longer, and the output field `Eω` now has shape `(Nω x Nm x Nz)` with `Nm` the number of modes:
```julia
julia> output_multimode["Eω"]
2049×4×201 Array{Complex{Float64},3}:
[...]
```
**NOTE:** Setting `modes=:HE11` and `modes=1` are **not** equivalent, except if only the Kerr effect is included in the simulation. The former uses mode-averaged propagation (treating all spatial dependence of the nonlinear polarisation the same as the Kerr effect) whereas the latter projects the spatially dependent nonlinear polarisation onto a single mode. This difference is especially important when photoionisation plays a major role.

### Plotting results
More usefully, you can directly plot the propagation results using `Plotting.prop_2D()` (`Plotting` is imported at the same time as `prop_capillary` by the `using Amalthea` statement):
```julia
julia> Plotting.prop_2D(output)
PyPlot.Figure(PyObject <Figure size 2400x800 with 4 Axes>)
```
This should show a plot like this:
![Propagation example 1](assets/readme_modeAvgProp.png)
You can also display the power spectrum at the input and output (and anywhere in between):
```julia
julia> Plotting.spec_1D(output, [0, 1.5, 3]; log10=true)
PyPlot.Figure(PyObject <Figure size 1700x1000 with 1 Axes>)
```
which will show this:
![Propagation example 2](assets/readme_modeAvgSpec.png)
`Plotting` functions accept many additional keyword arguments to quickly display relevant information. For example, you can show the bandpass-filtered UV pulse from the simulation using the `bandpass` argument:
```julia
julia> Plotting.time_1D(output, [2, 2.5, 3]; trange=(-10e-15, 30e-15), bandpass=(180e-9, 220e-9))
PyPlot.Figure(PyObject <Figure size 1700x1000 with 1 Axes>)
```
![Propagation example 3](assets/readme_modeAvgTime.png)

For multi-mode simulations, the plotting functions will display all modes individually by default. You can display the sum over modes instead using `modes=:sum`:
```julia
julia> Plotting.spec_1D(output_multimode; log10=true, modes=:sum)
PyPlot.Figure(PyObject <Figure size 1700x1000 with 1 Axes>)
```
![Propagation example 4](assets/readme_multiModeSpec.png)
(Compare this to the mode-averaged case above and note the important differences, e.g. the appearance of additional ultraviolet dispersive waves in higher-order modes.)

More plotting functions are available in the [`Plotting`](https://vdiego28.github.io/Amalthea.jl/stable/modules/Plotting.html) module, including for propagation statistics (`Plotting.stats(output)`) and spectrograms (`Plotting.spectrogram()`)

### Output processing
The `Processing` module contains many useful functions for more detailed processing and manual plotting, including:
- Spectral energy density on frequency or wavelength axis with optional spectral resolution setting (`Processing.getEω` and `Processing.getIω`)
- Time-domain fields and pulse envelopes with flexible frequency bandpass and linear (dispersive) propagation operators (`Processing.getEt`)
- Energy (`Processing.energy`) and peak power (`Processing.peakpower`) including after frequency bandpass
- FWHM widths in frequency (`Processing.fwhm_f`) and time (`Processing.fwhm_t`) as well as time-bandwidth product (`Processing.time_bandwidth`)
- g₁₂ coherence between multiple fields (`Processing.coherence`)

## GNLSE propagation
To run a simple simulation of nonlinear pulse propagation in an optical fibre using the generalised nonlinear Schrödinger equation (GNLSE), you can use `prop_gnlse`. As an example, we can model supercontinuum generation in a solid-core photonic crystal fibre for parameters corresponding to the simulations in Fig. 3 of Dudley et. al, RMP 78 1135 (2006).
```julia
julia> using Amalthea
julia> γ = 0.11
julia> flength = 15e-2
julia> βs = [0.0, 0.0, -1.1830e-26, 8.1038e-41, -9.5205e-56,  2.0737e-70, -5.3943e-85,  1.3486e-99, -2.5495e-114,  3.0524e-129, -1.7140e-144]
julia> output = prop_gnlse(γ, flength, βs; λ0=835e-9, τfwhm=50e-15, power=10e3, pulseshape=:sech, λlims=(400e-9, 2400e-9), trange=12.5e-12)
```
After this has run, you can visualise the output, with e.g.
```julia
julia> Plotting.prop_2D(output, :λ, dBmin=-40.0,  λrange=(400e-9, 1300e-9), trange=(-1e-12, 5e-12))
PyPlot.Figure(PyObject <Figure size 2400x800 with 4 Axes>)
```
This should show a plot like this:
![GNLSE propagation example](assets/readme_gnlse_scg.png)


## Examples
The [examples folder](examples/) contains complete simulation examples for a variety of scenarios, both for the [simple interface](examples/simple_interface/) and the [low-level interface](examples/low_level_interface). Some of the simple interface examples require the `PyPlot` package to be present, and many of the low-level examples require other packages as well—you can install these by simply typing `] add PyPlot` at the Julia REPL or the equivalent for other packages.

## The low-level interface
At its core, Amalthea.jl is extremely flexible, and the simple interface using `prop_capillary` only exposes part of what it can do. There are lots of examples in the [low-level interface examples folder](examples/low_level_interface). A representative subset of these (covering mode-averaged, modal, GNLSE, Raman, mixture, and step-index propagation) is smoke-tested in CI at a shrunk fibre length — see `test/test_examples_smoke.jl` — but most of the folder is not actively maintained and not guaranteed to run. As a side effect of its flexibility, it is quite easy to make mistakes when using the low-level interface. If you have trouble with this interface, [open an issue](https://github.com/vdiego28/Amalthea.jl/issues/new) with as much detail as possible.

## Running parameter scans
Amalthea.jl comes with a built-in interface which allows for the running of single- and multi-dimensional parameter scans with very little additional code. An example can be found in the [examples folder](examples/simple_interface/scan.jl) and more information is available in the [documentation](https://vdiego28.github.io/Amalthea.jl/stable/scans.html).

## New to Julia?
There are many resources to help you learn Julia. A good place to start is [Julia Academy](https://juliaacademy.com/) which has several courses for learning Julia depending on your current experience. There are additional resources linked from the [Julia website](https://julialang.org/learning/).

To edit and run Julia code, a very good option is the [Julia extension](https://www.julia-vscode.org/) for [Visual Studio Code](https://code.visualstudio.com/).

Julia fully supports [Unicode symbols in code](https://docs.julialang.org/en/v1/manual/variables/), including Greek letters. Amalthea.jl makes heavy use of this to name variables `ω` instead of `omega`, `π` instead of `pi`, etc. In any Julia console you can enter many Unicode characters using [a backslash and the tab key](https://docs.julialang.org/en/v1/manual/unicode-input/), for example `\omega<tab>` will result in `ω`, and `\ne<tab>` will result in `≠` (and the latter is equivalent to `!=`). For even faster entry of Greek letters specifically, you can use [this AutoHotkey script](https://github.com/q2apro/ahk_greekletters) or a number of other solutions.

## Getting help & contributing
If something does not work as expected, you have found a bug, or you simply want some advice, please [open a new issue](https://github.com/vdiego28/Amalthea.jl/issues/new) on this GitHub repository.

Amalthea.jl is being actively developed on this GitHub repository. To contribute a bugfix or a new feature, please create a pull request here. If you are new to GitHub, follow any one of the [many](https://github.com/firstcontributions/first-contributions) [useful](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/) [guides](https://codeburst.io/a-step-by-step-guide-to-making-your-first-github-contribution-5302260a2940) around to learn the (very simple!) GitHub workflow.

## Credits & Acknowledgements

**Amalthea.jl** is developed by Diego Andrés Valenzuela Berríos (Pontificia Universidad Católica de Chile).

This project is a fork of [Luna.jl](https://github.com/LupoLab/Luna.jl), originally developed by Chris Brahms ([@chrisbrahms](https://github.com/chrisbrahms)) and John Travers ([@jtravs](https://github.com/jtravs)) at the [Lupo Lab](https://lupo-lab.com/). We gratefully acknowledge their foundational work.

## Citing

If you use Amalthea.jl in your research, please cite it using the following DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20359892.svg)](https://doi.org/10.5281/zenodo.20359892)

```bibtex
@software{valenzuela_berrios_2026_amalthea_1_0_3,
  author    = {Valenzuela Berríos, Diego Andrés and Brahms, Christian and Travers, John C.},
  title     = {Amalthea.jl},
  year      = {2026},
  version   = {1.0.3},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21872422},
  url       = {https://doi.org/10.5281/zenodo.21872422}
}
```

For a version-independent reference, use the Amalthea.jl concept DOI
[`10.5281/zenodo.20359892`](https://doi.org/10.5281/zenodo.20359892).

## References
1. Kolesik, M., Moloney, J.V., 2004. Nonlinear optical pulse propagation simulation: From Maxwell's to unidirectional equations. Physical Review E - Statistical, Nonlinear, and Soft Matter Physics 70. https://doi.org/10.1103/PhysRevE.70.036604
2. Dormand, J.R., Prince, P.J., 1986. Runge-Kutta triples. Computers & Mathematics with Applications 12, 1007–1017. https://doi.org/10.1016/0898-1221(86)90025-8
3. Dormand, J.R., Prince, P.J., 1980. A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics 6, 19–26. https://doi.org/10.1016/0771-050X(80)90013-3
4. Hult, J., 2007. A Fourth-Order Runge–Kutta in the Interaction Picture Method for Simulating Supercontinuum Generation in Optical Fibers. Journal of Lightwave Technology 25, 3770–3775. https://doi.org/10.1109/JLT.2007.909373
5. Geissler, M., Tempea, G., Scrinzi, A., Schnürer, M., Krausz, F., Brabec, T., 1999. Light Propagation in Field-Ionizing Media: Extreme Nonlinear Optics. Physical Review Letters 83, 2930–2933. https://doi.org/10.1103/PhysRevLett.83.2930
6. Perelomov, A.M., Popov, V.S., Terent 'ev, M.V., 1966. Ionization of atoms in an alternating electric field. Soviet Physics JETP 23, 1393–1409.
7. Ammosov, M.V., Delone, N.B., Krainov, V.P., 1986. Tunnel Ionization Of Complex Atoms And Atomic Ions In Electromagnetic Field. Soviet Physics JETP 64, 1191–1194. https://doi.org/10.1117/12.938695
8. Börzsönyi, A., Heiner, Z., Kalashnikov, M.P., Kovács, A.P., Osvay, K., 2008. Dispersion measurement of inert gases and gas mixtures at 800 nm. Applied Optics 47, 4856. https://doi.org/10.1364/AO.47.004856
9. Ermolov, A., Mak, K.F., Frosz, M.H., Travers, J.C., Russell, P.S.J., 2015. Supercontinuum generation in the vacuum ultraviolet through dispersive-wave and soliton-plasma interaction in a noble-gas-filled hollow-core photonic crystal fiber. Physical Review A 92, 033821. https://doi.org/10.1103/PhysRevA.92.033821
10. Lehmeier, H.J., Leupacher, W., Penzkofer, A., 1985. Nonresonant third order hyperpolarizability of rare gases and N2 determined by third harmonic generation. Optics Communications 56, 67–72. https://doi.org/10.1016/0030-4018(85)90069-0
