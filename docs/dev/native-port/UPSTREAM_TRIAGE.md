# Upstream Luna.jl triage — 2026-07-31

This document records the useful and potentially backlog-worthy items found in
the [Luna.jl open pull requests](https://github.com/LupoLab/Luna.jl/pulls) and
[open issues](https://github.com/LupoLab/Luna.jl/issues), cross-checked against
the Amalthea native-port tree. It is a proposal list, not an authorization to
implement every upstream change. A candidate becomes live work only after its
design is recorded in [`PLANS.md`](PLANS.md) and promoted in
[`BACKLOG.md`](../BACKLOG.md).

## 2026-09-20 low-load review

Read-only GitHub inspection confirms `master` is still
[`08a53b32cbb4d811b7df0a65fa56ea6f79957256`](https://github.com/LupoLab/Luna.jl/commit/08a53b32cbb4d811b7df0a65fa56ea6f79957256).
There are no additional merged commits after the August review. The local
implementation and validation scope for this maintenance unit are defined in
[PLANS §24](PLANS.md#24-low-load-scan-and-upstream-maintenance-2026-09-20);
current pending gates live in [BACKLOG](../BACKLOG.md).

Disposition of every commit after the fork/performance baseline `0a52ffb`:

| Upstream commit | Local disposition |
|---|---|
| `7747c93` | Deferred FSAL carry is already implemented and tested locally; retain the stronger resident/legacy coverage. |
| `1d7e4c3` | DOPRI propagated weights are repaired in `30d26fd`; the additional fourth-order lifecycle correction is in `7f70784` (PLANS §17–18). |
| `4f1d1a7`, `8d9ee3d` | Merge commits integrating the preceding solver changes; no separate code port. |
| `32a6701` | Reviewed and deferred: `src/tsit.jl` is inactive. Correct and validate its tableau before ever enabling it; active `RK45` includes `dopri.jl`. |
| `08a53b3` | Ported `spectral_phase` naming and deprecated `getφ` forwarding in the working tree. `DataField` wavelength forwarding was already fixed locally. Numerical phase behavior is unchanged. |

The sole open fork issue at inspection,
[#67](https://github.com/vdiego28/Amalthea.jl/issues/67), lists the already
reviewed solver commits. The scheduled check now uses
`.github/upstream-reviewed.txt` so old reviewed commits do not trigger new
drift. This checkpoint denotes review, including explicit deferrals, rather
than blanket integration. Its update leaves the frozen performance baseline
unchanged. No issue comment or closure was submitted.

Notebook scan construction is repaired locally from the independently
confirmed `ARGS` ownership defect described in
[#317](https://github.com/LupoLab/Luna.jl/issues/317#issuecomment-1492459791).
The constructor no longer consumes global arguments; initialized IJulia uses
empty implicit arguments, while explicit argument vectors remain authoritative.

Open upstream proposals and their September 20 dispositions:

- [#442](https://github.com/LupoLab/Luna.jl/pull/442), head `132b3d3`, removes
  the spectral centering ramp before unwrapping. Adopted in the later
  September 20 analytic work under [PLANS §25](PLANS.md#25-analytic-audit-of-propagation-frames-and-spectral-diagnostics-2026-09-20),
  separately from the merged API rename; current validation status is in BACKLOG.
- [#441](https://github.com/LupoLab/Luna.jl/pull/441) recovers scan points
  abandoned by dead queue workers; it needs process-failure/ownership tests.
- [#440](https://github.com/LupoLab/Luna.jl/pull/440) changes modified shot-noise
  propagation; it needs a coordinated Julia/native mathematical review.
- [#439](https://github.com/LupoLab/Luna.jl/pull/439) extends loss to real scaling
  and frequency-dependent functions; review against existing eligibility and
  callback contracts before porting.

## 2026-08-24 upstream refresh and v1.0.4 gate (historical)

Upstream `master` was fetched through commit `08a53b3` after the CPU work was
implemented. The frozen performance-audit comparison remains pinned to
`0a52ffb`; changing that baseline would invalidate its recorded measurements.

- `7747c93` fixes deferred FSAL and dense-output state. Amalthea already has an
  independently implemented version with stronger resident/legacy coverage.
- `1d7e4c3` proves that the Dormand–Prince weight vectors were named backwards
  and that default `locextrap=true` propagates the embedded fourth-order state.
  Amalthea's Julia, legacy Rust, resident CPU, and CUDA steppers all share that
  historical convention, so mutual equivalence cannot detect it. This is a
  correctness gate for `v1.0.4`: it requires one coordinated, design-first
  change plus exact order, FSAL, endpoint, rejection, dense-output, and
  cross-backend tests. It is not folded into the CPU optimization commit.
- `32a6701` corrects the analogous Tsitouras tableau, but Amalthea's active
  RK45 module includes `dopri.jl`, not `tsit.jl`; it is recorded rather than
  treated as a release blocker.
- `08a53b3` adds `spectral_phase` with a deprecated `getφ` alias. The naming
  improvement is suitable for a separate compatibility change; its associated
  `DataField(...; λ0=...)` fix was already present locally.
- Open PR #438 adds rate-based absorbing boundaries. It remains a candidate
  until its upstream documentation job is green and a resident-native design
  defines callback/state ownership.

The release-candidate branch may run hosted tests while the DOPRI item is
resolved, but it must not be tagged or published merely because the shared-bug
Julia/native equivalence suite is green.

## Highest-value candidates

### 1. Make `Scan` safe in IJulia — issue [#317](https://github.com/LupoLab/Luna.jl/issues/317)

Selected and implemented in the 2026-09-20 working tree; see the review above
and the backlog for validation status. The following is the original rationale.

IJulia puts the kernel connection JSON in `ARGS`. Before the repair, the
[`src/Scans.jl`](../../../src/Scans.jl) path treated `ARGS` as Luna execution
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
- **Public API cleanup — PR [#434](https://github.com/LupoLab/Luna.jl/pull/434).** The merged rename and compatibility alias are ported in the September 20 working tree; see the review above. The related `DataField(...; λ0=...)` forwarding bug was already fixed locally.
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

1. Complete validation of the IJulia `ARGS` repair when compute is available.
2. Step-index discontinuity regression.
3. BSI PPT correction.
4. VUV models, then full RDW phase matching.
5. z-dependent `preionfrac`.
6. TPA, Slurm, and MKL according to user demand and upstream maturity.

Each selected item should first receive a design entry in `PLANS.md`, then a
live status entry in `BACKLOG.md`, followed by the normal Julia-oracle and
native-equivalence tests.
