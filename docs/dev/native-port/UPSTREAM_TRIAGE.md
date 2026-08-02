# Upstream Luna.jl triage — 2026-07-31

This document records the useful and potentially backlog-worthy items found in
the [Luna.jl open pull requests](https://github.com/LupoLab/Luna.jl/pulls) and
[open issues](https://github.com/LupoLab/Luna.jl/issues), cross-checked against
the Amalthea native-port tree. It is a proposal list, not an authorization to
implement every upstream change. A candidate becomes live work only after its
design is recorded in [`PLANS.md`](PLANS.md) and promoted in
[`BACKLOG.md`](../BACKLOG.md).

## Highest-value candidates

### 1. Make `Scan` safe in IJulia — issue [#317](https://github.com/LupoLab/Luna.jl/issues/317)

IJulia puts the kernel connection JSON in `ARGS`. The current
[`src/Scans.jl`](../../../src/Scans.jl) path treats `ARGS` as Luna execution
arguments by default, so scans launched from notebooks can fail in argument
parsing.

Proposed work:

- isolate notebook/kernel arguments from Luna scan arguments;
- preserve the existing explicit `Scan(name, args)` and `Scan(name, exec)` APIs;
- add a regression test using representative IJulia kernel arguments;
- document the intended notebook invocation.

This is the clearest short, independent fix and is unrelated to the Rust
backend.

### 2. Barrier-suppression ionisation corrections — PR [#426](https://github.com/LupoLab/Luna.jl/pull/426)

The PR adds `bsi=:auto|:none|:tonglin|:zhang` corrections to PPT ionisation.
Local [`src/Ionisation.jl`](../../../src/Ionisation.jl) currently uses barrier
suppression only when selecting the field range for its rate cache; the rate
correction itself is not present.

Design requirements:

- decide whether `:auto` should become the default and for which species;
- define cache-key/version behavior and preserve reproducibility for
  `bsi=:none`;
- keep Julia, resident CPU-native, and CUDA behavior equivalent;
- include a single-step test and a full-solve test that prove the correction
  changes the oracle by more than the asserted tolerance.

### 3. VUV refractive indices and full RDW phase matching — PR [#425](https://github.com/LupoLab/Luna.jl/pull/425)

The PR adds VUV models for Ne, Ar, and Kr and phase-matching terms for linear,
nonlinear, and ionisation contributions. These models and the proposed
`λRDWfull`/`PhaseMatching` functionality are absent from local
[`src/PhysData.jl`](../../../src/PhysData.jl) and [`src/Tools.jl`](../../../src/Tools.jl).

Split this into two reviewable units:

1. add named VUV models without silently changing existing model names or
   defaults;
2. add full phase-matching calculations with bracket/root and range tests.

### 4. Step-index discontinuity filtering — issue [#320](https://github.com/LupoLab/Luna.jl/issues/320), PR [#321](https://github.com/LupoLab/Luna.jl/pull/321)

The local step-index solver appears to contain the TE/TM correction, but not
the upstream filter that rejects roots caused by characteristic-function
discontinuities. Add a focused regression test first, then decide whether the
filter belongs in the retained Julia solver. This should remain separate from
the parked native multi-mode port, which still requires a numerical design.

### 5. Spatially varying `preionfrac` — PR [#417](https://github.com/LupoLab/Luna.jl/pull/417)

The PR allows preionisation to vary with propagation distance. Local
[`src/Nonlinear.jl`](../../../src/Nonlinear.jl) and the native FFI currently bake a
scalar value into the simulation.

The design must choose between an explicit Julia fallback and a native setter
or per-step update. It must also preserve old callback signatures, validate
callability, and enforce the physical range `[0, 1]`.

## Useful, but dependent on upstream/API decisions

### TPA — PR [#412](https://github.com/LupoLab/Luna.jl/pull/412)

Two-photon absorption would be useful, but the upstream review identified
disabled-flag behavior, unsafe `@inbounds` length assumptions, documentation
errors, and a potentially problematic initial step-size change. Revisit after
those issues settle; start with a Julia implementation and define native/GPU
scope explicitly.

### Slurm execution — PR [#420](https://github.com/LupoLab/Luna.jl/pull/420)

The local `SlurmExec` implementation is still minimal. The upstream work adds
memory, project, thread/process, working-directory, array, time, partition,
and completion-marker support. Wait for the remaining SSHExec/release-note
coverage, then port the script-generation behavior with tests for quoting,
memory units, and missing project paths.

### MKL and alternative FFT providers — PR [#411](https://github.com/LupoLab/Luna.jl/pull/411)

This is relevant for users who cannot use FFTW, but it needs a native-backend
design: Rust currently binds FFTW directly. The Julia-only partial-dimension
workarounds should not be copied unconditionally into the normal FFTW path.
Provider detection, wisdom behavior, and an actual MKL test environment are
required before implementation.

## Strategic watch list

- **Free-space/χ²/birefringent geometries — PR [#416](https://github.com/LupoLab/Luna.jl/pull/416).** Relevant to the existing broader GPU physics/geometries item, but the PR is a large WIP and its review identifies scalar-`nfun`, non-square-grid, array-dimensionality, indexing, thread-safety, and FFT-plan issues. Use it as design input only after it stabilizes.
- **External companion workbench — issue [#435](https://github.com/LupoLab/Luna.jl/issues/435).** Worth monitoring for file-format and job-runner interoperability; it currently proposes an external application rather than a Luna integration.
- **Public API cleanup — PR [#434](https://github.com/LupoLab/Luna.jl/pull/434).** Renaming `getφ` to `spectral_phase` is reasonable, but keep a compatibility alias for one release. The related `DataField(...; λ0=...)` forwarding bug is already fixed locally.
- **Grating compression — PR [#364](https://github.com/LupoLab/Luna.jl/pull/364).** Potentially useful for large optical systems, but lower priority than scan reliability and physics correctness.

## Already covered or not immediate

Do not duplicate these as new backlog items without new evidence:

- optional plotting support — PR [#313](https://github.com/LupoLab/Luna.jl/pull/313);
- vector-only capillary gradients — PR [#330](https://github.com/LupoLab/Luna.jl/pull/330);
- end-to-end example smoke testing — issue [#225](https://github.com/LupoLab/Luna.jl/issues/225);
- Roots compatibility — PR [#424](https://github.com/LupoLab/Luna.jl/pull/424);
- short-kernel Raman optimization — issue [#120](https://github.com/LupoLab/Luna.jl/issues/120), measured and rejected locally;
- Raman in hollow capillaries — issue [#311](https://github.com/LupoLab/Luna.jl/issues/311), whose upstream discussion indicates a material-model mismatch rather than a missing generic kernel.

## Suggested order

1. IJulia `ARGS` isolation.
2. Step-index discontinuity regression.
3. BSI PPT correction.
4. VUV models, then full RDW phase matching.
5. z-dependent `preionfrac`.
6. TPA, Slurm, and MKL according to user demand and upstream maturity.

Each selected item should first receive a design entry in `PLANS.md`, then a
live status entry in `BACKLOG.md`, followed by the normal Julia-oracle and
native-equivalence tests.
