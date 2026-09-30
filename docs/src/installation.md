# Installation and configuration

Amalthea requires Julia 1.10 or newer. The normal installation uses the
resident CPU backend and does **not** require CUDA, an NVIDIA GPU, `nvcc`, or a
Rust toolchain when a matching release binary is available.

## Platform support

The installer selects binaries using both the operating system and CPU
architecture. It never installs a binary for a different architecture.

| Operating system | Architecture | Release binary | Installation path |
|---|---|---|---|
| Linux (glibc) | `x86_64` | Yes | Download verified binary; build from source if unavailable |
| Linux (glibc) | ARM64/AArch64 | Yes | Download verified binary; build from source if unavailable |
| macOS | Apple Silicon/ARM64 | Yes | Download verified binary; build from source if unavailable |
| Windows | `x86_64` | Yes | Download verified binary; build from source if unavailable |
| macOS | Intel `x86_64` | No | Build from source |
| Windows | ARM64 | No | Build from source |
| Linux | ARMv6/ARMv7, musl, or another architecture/libc | No | Build from source; not release-tested |
| Other Julia-supported systems | Varies | No | Source-build fallback; not release-tested |

The Linux ARM64 release job uses Ubuntu 22.04 to avoid imposing the newer
glibc baseline of Ubuntu 24.04. A machine with an older or incompatible libc
can still compile the library locally.

WSL follows the Linux row, not the Windows row. Check the platform Julia sees
with:

```julia
julia> (Sys.KERNEL, Sys.ARCH)
(:Linux, :x86_64)
```

## Tagged installation on current release platforms

### One-command installer (Linux and macOS)

For a new CPU installation or an update to the latest release, run:

```text
curl -fsSL https://raw.githubusercontent.com/vdiego28/Amalthea.jl/main/install.sh | bash
```

The script installs Julia through Juliaup when `julia` is not already
available, resolves the newest GitHub release tag, and installs that exact tag
into Julia's default environment. It does not configure CUDA. If Juliaup must
be installed, its own installer manages the Julia installation and PATH setup.
Rerun the same command to update Amalthea; it changes only the Amalthea package
revision, not every package in the environment.

To install a particular release instead, pass its tag after `--`:

```text
curl -fsSL https://raw.githubusercontent.com/vdiego28/Amalthea.jl/main/install.sh | bash -s -- --version v1.0.4
```

The installer needs `curl` and an internet connection. On Windows, use the
Julia package command below (or run the script from WSL); native PowerShell
support is not provided by this POSIX shell script.

After either installation method, start Julia normally and run:

```julia
using Amalthea
Amalthea.backend_report()
```

### Manual tagged installation

Install Julia from [julialang.org/downloads](https://julialang.org/downloads/),
start Julia, and enter package mode with `]`. Amalthea is not yet registered in
Julia's General registry, so install a tagged release directly from GitHub.
The current release is `v1.0.4`:

```julia
pkg> add https://github.com/vdiego28/Amalthea.jl#v1.0.4
```

The equivalent programmatic command is the same on every operating system:

```julia
using Pkg
Pkg.add(url="https://github.com/vdiego28/Amalthea.jl", rev="v1.0.4")
```

Use the newest tag shown on the project's
[Releases page](https://github.com/vdiego28/Amalthea.jl/releases). Do not omit
the tag unless a source build of the development branch is intended.

The `v1.0.4` binaries cover Linux `x86_64`, Linux ARM64, macOS Apple Silicon,
and Windows `x86_64`. Other OS/architecture combinations use the
architecture-safe source fallback.

During installation, `deps/build.jl`:

1. identifies the exact OS/architecture pair;
2. downloads the package version's release library when one exists;
3. verifies it against `SHA256SUMS.txt`;
4. otherwise runs `cargo build --release` from source.

Binaries produced by the current release workflow are CPU-only. CUDA libraries
are loaded dynamically only if the experimental CUDA backend is explicitly
enabled later.

Verify the installation:

```julia
using Amalthea
Amalthea.backend_report()
```

`last_stepper_type` is `nothing` until a propagation has run. The normal
configuration reports `native = true`, `cuda_native = false`, and
`gpu_dispatch = :auto`.

## When Rust is required

Install Rust 1.85 or newer when any of these apply:

- no prebuilt release binary exists for the platform;
- the prebuilt download is unavailable or deliberately disabled;
- installing the development branch or working from a git checkout;
- building CUDA kernels.

Install Rust through [rustup](https://rustup.rs/) and open a new terminal so
`cargo` is on `PATH`. Confirm it with:

```text
cargo --version
```

The operating system must also provide its normal native linker:

- **Linux, including ARM64:** install the distribution's C build tools (for
  example, the package commonly named `build-essential` on Debian/Ubuntu).
- **macOS:** install the Xcode Command Line Tools with
  `xcode-select --install`.
- **Windows:** use rustup's default MSVC toolchain and install the Visual Studio
  C++ Build Tools when rustup requests them.

CUDA is not part of these source-build prerequisites.

## Development checkout

A checkout always builds from source because its Julia and Rust FFI code may
be newer than the latest tagged binary:

```text
git clone https://github.com/vdiego28/Amalthea.jl.git
cd Amalthea.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Amalthea")'
```

On Windows PowerShell, the last command is also valid as written. To build the
Rust library directly:

```text
cd amalthea
cargo build --release
```

Direct Cargo builds default to `AMALTHEA_CUDA_BUILD=auto`; package builds
default to `off`. Use `RUSTFLAGS=""` for portable or cross-compiled binaries,
because the repository's developer Cargo configuration otherwise selects the
build machine's native CPU features.

## CPU-only configuration

No environment variables are needed for ordinary CPU operation. The effective
defaults are:

```text
AMALTHEA_CUDA_BUILD=off
AMALTHEA_USE_RUST_NATIVE=1
AMALTHEA_USE_RUST_CUDA_NATIVE=0
AMALTHEA_NATIVE_GPU=auto
```

To force a CPU-only source rebuild without probing `nvcc`, use a fresh Julia
process that has not loaded Amalthea:

```julia
ENV["AMALTHEA_CUDA_BUILD"] = "off"
using Pkg
Pkg.build("Amalthea")
```

To use the Julia implementation instead of the resident Rust CPU backend:

```julia
ENV["AMALTHEA_USE_RUST_NATIVE"] = "0"
using Amalthea
```

This is a correct but generally slower fallback and is also the numerical
oracle used by the native-backend tests.

## CUDA installation

CUDA is an explicit source-build option. It requires an NVIDIA driver and a
CUDA toolkit containing `nvcc`. Modern macOS systems do not have a supported
NVIDIA CUDA path and should use CPU-only mode.

Build in a fresh Julia process:

```julia
ENV["AMALTHEA_CUDA_BUILD"] = "required"
using Pkg
Pkg.build("Amalthea")
```

`required` skips CPU-only release binaries and fails if real PTX cannot be
compiled. If `nvcc` is installed outside the conventional location, set one of
these before building:

```julia
ENV["NVCC"] = "/absolute/path/to/nvcc"
# or
ENV["CUDA_HOME"] = "/path/to/cuda"
# CUDA_PATH is also recognized, especially on Windows.
```

Close that Julia process after the build. In a new process, enable CUDA:

```julia
ENV["AMALTHEA_USE_RUST_CUDA_NATIVE"] = "1"
ENV["AMALTHEA_NATIVE_GPU"] = "on"  # force CUDA for supported configurations
using Amalthea
```

Use `AMALTHEA_NATIVE_GPU=auto` instead of `on` to apply the measured automatic
dispatch thresholds. `off` always selects CPU. Both `auto` and `on` still
require the master `AMALTHEA_USE_RUST_CUDA_NATIVE=1` opt-in. Unsupported
geometry/physics combinations fall back to CPU; see the
[native support matrix](https://github.com/vdiego28/Amalthea.jl/blob/main/docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md)
for the current CUDA scope.

An NVIDIA Jetson or another Linux ARM64 CUDA system must build from source with
its native toolkit. That combination is not covered by the CPU-only ARM64
release artifact.

If a CPU-only library is accidentally used with CUDA enabled, initialization
stops before loading the driver and reports that the package must be rebuilt
with `AMALTHEA_CUDA_BUILD=required`.

## Environment-variable syntax by operating system

Setting variables through Julia's `ENV` dictionary, as shown above, is the
most portable method. For shell-level configuration:

### Linux and macOS shells

For one command:

```text
AMALTHEA_CUDA_BUILD=off julia -e 'using Pkg; Pkg.build("Amalthea")'
```

For the current shell and its child processes:

```text
export AMALTHEA_USE_RUST_CUDA_NATIVE=1
export AMALTHEA_NATIVE_GPU=on
julia
```

Put the `export` lines in the appropriate shell profile only if the setting
should be persistent.

### Windows PowerShell

For the current PowerShell session:

```powershell
$env:AMALTHEA_CUDA_BUILD = "off"
julia -e 'using Pkg; Pkg.build("Amalthea")'
```

For CUDA runtime selection:

```powershell
$env:AMALTHEA_USE_RUST_CUDA_NATIVE = "1"
$env:AMALTHEA_NATIVE_GPU = "on"
julia
```

To persist a setting for future terminals:

```powershell
[Environment]::SetEnvironmentVariable("AMALTHEA_NATIVE_GPU", "on", "User")
```

Open a new terminal after changing persistent variables.

### Windows Command Prompt

For one `cmd.exe` session:

```text
set AMALTHEA_CUDA_BUILD=off
julia
```

Then run `using Pkg; Pkg.build("Amalthea")` at the Julia prompt. This avoids
shell-specific nested-quote rules.

Use `set AMALTHEA_USE_RUST_CUDA_NATIVE=1` and
`set AMALTHEA_NATIVE_GPU=on` before starting Julia for forced CUDA dispatch.

## Configuration reference

Build-time settings are consumed by `Pkg.build` or Cargo:

| Variable | Values and default | Effect |
|---|---|---|
| `AMALTHEA_CUDA_BUILD` | `off`, `auto`, `required`; package default `off`, direct Cargo default `auto` | Disable CUDA compilation, try it with CPU fallback, or require real PTX |
| `AMALTHEA_RUST_SKIP_DOWNLOAD` | `0`/unset or `1`; default unset | Force the source-build path instead of trying a release binary |
| `NVCC` | executable path | Override the CUDA compiler location |
| `CUDA_HOME`, `CUDA_PATH` | toolkit root | Search `<root>/bin/nvcc` |
| `RUSTFLAGS` | Rust compiler flags | Set to an empty string for portable release/cross builds |

Primary runtime settings are re-read when backend configuration is queried:

| Variable | Values and default | Effect |
|---|---|---|
| `AMALTHEA_USE_RUST_NATIVE` | `0` or `1`; default `1` | Enable the resident Rust CPU backend |
| `AMALTHEA_USE_RUST_CUDA_NATIVE` | `0` or `1`; default `0` | Master opt-in for resident CUDA |
| `AMALTHEA_NATIVE_GPU` | `off`, `auto`, `on`; default `auto` | Force CPU, use measured dispatch, or force supported CUDA configurations |
| `AMALTHEA_QDHT_BLAS` | `off`, `auto`, `on` (`0`, `1` aliases); default `auto` | Force Rayon, automatically use configured BLAS above the measured QDHT threshold, or force configured BLAS |
| `AMALTHEA_NATIVE_DETERMINISTIC` | `0` or `1`; default `0` | Force the fixed-order Rayon QDHT kernel regardless of `AMALTHEA_QDHT_BLAS` |
| `AMALTHEA_NATIVE_FFTW_WISDOM` | `0` or `1`; default `0` | Opt into native FFTW wisdom import/export |

Boolean switches recognize exactly `"1"` as enabled. GPU mode recognizes
exactly `off` and `on`; other values resolve to `auto`, so use the documented
spellings.

Older per-kernel experimental toggles remain available for developers:
`AMALTHEA_USE_RUST_STEPPER`, `AMALTHEA_USE_RUST_IONISATION`,
`AMALTHEA_USE_RUST_RAMAN`, `AMALTHEA_USE_RUST_DISPERSION`,
and `AMALTHEA_USE_RUST_QDHT`. They default to `0` and are not needed for
normal resident-backend use. `AMALTHEA_QDHT_BLAS` is shared by the legacy and
resident radial paths and defaults to `auto`.

Resident QDHT calls Julia's configured `libblastrampoline` provider; it does
not assume OpenBLAS. On macOS this means Accelerate is used when Julia has been
configured for Accelerate. For Apple Silicon correctness and topology tuning,
run `python3 test/performance_audit/run_apple_quick_test.py` from a checkout;
the runner restores the normal portable library after its diagnostic build.

## Updating, rebuilding, and switching modes

Because Amalthea is installed from GitHub rather than General, upgrade by
selecting the newer release tag. The current tagged release is:

```julia
using Pkg
Pkg.add(url="https://github.com/vdiego28/Amalthea.jl", rev="v1.0.4")
```

Replace `v1.0.4` with the newer tag shown on the Releases page when one is
published. The package's build step runs automatically after the revision is
changed; `Pkg.build("Amalthea")` can be used to repeat it manually.

Use a fresh Julia process for rebuilds, especially on Windows where a loaded
DLL cannot safely be replaced. After changing between CPU-only and CUDA builds,
restart Julia before running a simulation.

To force a fresh source build of the installed tagged release:

```julia
ENV["AMALTHEA_RUST_SKIP_DOWNLOAD"] = "1"
ENV["AMALTHEA_CUDA_BUILD"] = "off"  # or "required"
using Pkg
Pkg.build("Amalthea")
```

## Troubleshooting

### `cargo` cannot be found

No matching prebuilt binary was usable, so the installer selected its source
fallback. Install Rust through rustup, open a new terminal, verify
`cargo --version`, and rerun `Pkg.build("Amalthea")`.

### A prebuilt binary was expected but source compilation started

Check `(Sys.KERNEL, Sys.ARCH)`, package version, access to `github.com`, and
whether `AMALTHEA_RUST_SKIP_DOWNLOAD=1` is set. A source checkout always builds
locally. CUDA `auto` or `required` also deliberately skips CPU-only binaries.

### `nvcc` is missing or PTX compilation fails

CPU users should set `AMALTHEA_CUDA_BUILD=off`. CUDA users should verify
`nvcc --version`, then set `NVCC`, `CUDA_HOME`, or `CUDA_PATH` and rebuild with
`required`.

### CUDA says the library was built without kernels

The installed binary is CPU-only. Close Julia, rebuild from source with
`AMALTHEA_CUDA_BUILD=required`, and start a new Julia process before enabling
the CUDA runtime variables.

### A downloaded Linux library requires a newer libc

Force a local source build with `AMALTHEA_RUST_SKIP_DOWNLOAD=1`. The resulting
library links against the local system instead of the release runner's glibc.

### The package loads but a configuration uses Julia

Run `Amalthea.backend_report()` after a propagation. Unsupported native or CUDA
configurations deliberately use a correct fallback. Consult the native support
matrix before treating a fallback as an installation failure.
