# Backlog

Deferred work and known issues for Amalthea.jl. Severity: 🔴 correctness · 🟡 robustness/CI · ⚪ informational.

> **Where things live.** This file is the *live* backlog. Completed work —
> Phases A-J, tracks S1 and S4, and the rolling "Done (recent)" log — moved to
> [`ARCHIVE.md`](ARCHIVE.md) with its section names unchanged. Cross-references
> below to a phase, to S1/S4, or to "Done (recent)" resolve there.

## Start here — current resume queue (2026-07-28)

This is the authoritative short queue. The long sections below retain design
history and measured evidence, but older words such as "next", "not started",
or "verified" inside a superseded narrative do not outrank this list.

> **2026-07-25 agent wave — four of the five queue items below were worked
> and are resolved or closed.** Four agent branches merged with no conflicts.
> Per-item records: `native-port/portlog-inbox/{gpu-nonlinearity,
> examples-repair,prebuilt-asset-compat,raman-short-kernel}.md`, each also
> summarized in `PORT_LOG.md`.
>
> **Gate, re-run by the lead on the integrated tree (not agent-reported):**
> full 7-group `run_full_gate.py` **exit 0, 774.3s** — physics 1657/1657,
> rust 42247/42248, sim_multimode 33/33, sim_interface 314/314,
> sim_propagation 18/18, io 2302/2302, fields 334/334. Supplemental
> `examples` group **20/20**. GPU verified separately by running the `rust`
> group under `AMALTHEA_USE_RUST_CUDA_NATIVE=1 AMALTHEA_NATIVE_GPU=on`:
> `test_native_cuda.jl` **31/31** — see item 9 for why that same env makes
> 18 *unrelated* CPU-native tests fail.

1. 🟢 **Restore nonlinear physics on `CudaNativeSim` (S3 item 0) — DONE
   2026-07-25, verified on real hardware.** `set_mode_avg_params` now uploads
   `pre`/`β`/`sidx`/`ωwin`/`nlscale`/`sqrt_aeff`, and a new
   `compute_rhs_mode_avg` helper ports CPU Steps 1/2/5/6/7 alongside the
   existing Kerr cubic; all Kerr/plasma buffers and cuFFT plans resized to
   `n_time_over` (**this also closes S3 item 6**, which is not separable —
   Steps 1 and 5 are crop/pad operations); both `cufftPlan1d` return codes
   checked. A **second, undiagnosed bug** was found in review:
   `CudaNativeSim::set_field` never seeded `ks_d[0]` (CPU's does), so the
   first `step()` read uninitialized `cuMemAlloc` memory — latent, and the
   Kerr fix would have activated it. Measured on the RTX 5060 Ti: stage
   derivatives `3.5e-13` → `~1230` (CPU-native agreement ~1e-15), fixed-step
   full-solve vs the Julia oracle **3.5e-16**, `Luna.run` dense output
   **1.25e-7**. Test tolerances tightened `1e-3`/`5e-2` → `1e-12`, with the
   config's nonlinear share (`rel_nl` ≈ 4.5e-4) now *measured in-test* and
   asserted to exceed the tolerance by >100× — the AGENTS.md §3 step 4 rule
   whose absence let this ship. Still open: the `err` weak-norm placeholder
   is demoted to a diagnostic rather than fixed (a real pre-acceptance trial
   solution in `step()` is the honest fix), and GPU scope beyond
   mode-averaged RealGrid Kerr(+PPT) is untouched.
2. 🟡 **Add standing GPU CI — deliberately deferred 2026-07-25 (lead's
   call).** A CUDA-equipped scheduled/dedicated runner must run the CUDA Rust
   tests and `test/test_native_cuda.jl`; until then every GPU change also
   requires a recorded manual hardware run. **Item 1 is the argument for
   this:** a zero-nonlinearity GPU backend survived over two weeks because
   nothing re-measured it automatically.
3. 🟢 **Repair the known-broken low-level examples — 7 of 7 DONE
   2026-07-27.** Both documented classes fixed (`linop` before assignment ×6;
   `norm_modal(grid.ω)` ×3) and re-audited across all 44 example files — the
   file list was exactly right. Four *further* bugs surfaced that the
   original audit missed because its harness stopped at the first error per
   file (a scalar `ϕ` where `Fields.PulseField.ϕ::Vector{Float64}` is
   required; and in `elliptical_env.jl` an undefined `τ`, a missing broadcast
   dot, a missing `import FFTW`, and a positional `normfun` where `setup`
   takes `norm!` as a keyword). Regression cases for both classes added to
   `test/test_examples_smoke.jl` and verified to fail against the unfixed
   originals. The seventh file's apparent library defect was corrected
   2026-07-27 after tracing the Cubature callback: its
   `PlasmaCumtrapz` was constructed with a vector example field even though
   `components=:xy` supplies an N×2 field. The example now allocates N×2
   plasma buffers, `PlasmaCumtrapz` reports a focused shape error for this
   misuse, and `test/test_transmodal_vector_plasma.jl` covers the actual
   `full=true` + npol=2 + plasma path. `examples` 20/20,
   `sim-multimode` 41/41; the corrected example also completed end-to-end
   at a shortened 5 mm length.
4. 🟢 **Prebuilt release installation — local fix DONE 2026-07-25; release
   handling decided.** `deps/build.jl` now tries the canonical
   `libamalthea-<triple>` asset first and falls back to the legacy
   `libluna_rust-<triple>` name *only* for versions ≤ `v1.0.0`, so a future
   genuinely-broken release cannot be masked. Verified against the real
   published `v1.0.0` (download + checksum + install) and by a 4-scenario
   local-HTTP fixture suite (20/20). `release.yml` already stages canonical
   names for future tags — unchanged. **Lead's decision: leave `v1.0.0`'s
   published assets untouched and prepare a `v1.0.1` instead** (see the new
   item below); no release asset was mutated.
   🔴→🟢 **Regression on the very next push, fixed 2026-07-26.** With the
   fallback in place, CI itself downloaded the stale `v1.0.0` binary over the
   library it had just built and every job in both workflows died with
   `undefined symbol: native_compute_extra_stages` (the S5.3 dense-output FFI
   symbol, added after the tag). Root cause is the version keying, not the
   name matching: `Project.toml` still reads `1.0.0` while `main`'s sources
   are far ahead of it. `deps/build.jl` now refuses the download entirely for
   a source checkout (`_is_source_checkout()`, `.git` present — registered
   `Pkg.add` installs keep the fast path), and both workflows set
   `AMALTHEA_RUST_SKIP_DOWNLOAD=1` at workflow level. **Closed
   2026-07-28:** immediately after tagging `v1.0.1`, `main` moved to
   `Project.toml` version `1.0.2-DEV` (and Python `1.0.2.dev0`), so source
   tarballs no longer identify themselves as the preceding release. See
   `docs/dev/native-port/portlog-inbox/prebuilt-asset-compat.md` §7.
5. ⚪ **Short-kernel Raman convolution — measured 2026-07-25, RECOMMEND
   AGAINST, closed.** This **reverses Phase J.6(c)'s prior "recommend"**,
   whose premise was an unmeasured guess. `PLANS.md` §6.3 assumed the SiO2
   response decays within ~100fs / 5-10% of the padded grid; the real
   Hollenbeck & Cantrell support at an f64-noise cutoff is **~4.15ps ≈ 76-86%
   of `n_time_over`** — about 40× longer. Isolated speedup is **0.98× (i.e.
   slower)** at the repo's real config (n_time_over=4096: the natural
   `n_time_over+M` length isn't a power of two and FFTW's mixed-radix penalty
   erases the gain), reaching 2.1× only at sizes no config here uses.
   Projected end-to-end ~0.99-1.05× against the >1.4× S5.1 gate. Correctness
   was never the blocker (truncation error 1.7e-16-3.4e-14, inside the
   existing tier). Bench and profiling reverted per S1.6/S5.1 discipline.

### New items raised by the 2026-07-26 CI repair

11. 🟢 **macOS CI `Bus error: 10` — FIXED and repeatedly verified
    2026-07-27.** `physics -
    macos-latest` died with `signal 10 (1): Bus error: 10`, "in expression
    starting at `test/test_rk45.jl:64`", in **2 of 3** runs on 2026-07-26
    (fail `30209977981`, pass `30210333905`, fail `30212265976`). Same file,
    same line, no Julia backtrace printed, always mid-solve (~11-13 % of the
    way in, after ~1650-1900 steps).

    **The most important fact: line 64 is `RK45.solve(f!, ...)` — the plain,
    *non-preconditioned* Julia stepper.** `AMALTHEA_USE_RUST_NATIVE` is only
    wired at `solve_precon`, and this testitem's `f!`/`fnl!` are pure Julia
    closures over `FFTW.plan_fft!`/`plan_ifft!`. So no `RustNativeStepper`,
    no FFI, no `libamalthea` code runs in the crashing call. Do not start
    this investigation in the native port.

    Not introduced by the 2026-07-26 CI repair: that changed only which
    library CI builds (source vs stale release asset) and a Python skip
    guard, neither of which this code path touches. It was *masked* before —
    on 2026-07-25 the whole suite died earlier at `undefined symbol`, and the
    last macOS-green run was 2026-07-23. Which leaves the change in the
    environment (macos-latest image, Julia '1' point release) or a
    long-standing rare race that only now got sampled.

    Leads, in order: (a) the restored `julia-actions/cache` carried
    `lunacache/FFTW*wisdom` between runs and its key does not distinguish
    Apple Silicon generations — wisdom recorded on one CPU replayed on
    another is a classic SIGBUS source, and the crash logs do show "Found
    FFTW wisdom at …" immediately before; (b) in-place `FFT*out`/`IFT*out`
    plans applied to a buffer whose alignment differs from the planned one;
    (c) genuine memory corruption elsewhere in the physics group that only
    lands here. The bounded first mitigation in
    `.github/workflows/run_tests.yml`: retain package caching everywhere,
    but set `cache-scratchspaces: false` only for the macOS physics matrix
    entry so CPU-specific FFTW wisdom cannot be restored across runners —
    was **not sufficient**. Branch run `30291822719`, job `90063141471`
    created fresh host-local wisdom and still crashed at 94.68% / 20,541
    steps. The next bounded change extends `test/runtests.jl`'s existing
    Windows-only `set_fftw_threads(1)` guard to macOS: the job uses
    `JULIA_NUM_THREADS=auto`, which otherwise makes Amalthea request 12 FFTW
    threads for this repeatedly-executed 1024-point in-place transform.
    Julia threads remain enabled and production defaults are unchanged.
    Branch run `30293434654` passed the full 16-job matrix; its macOS physics
    job then passed three consecutive executions on commit `3c3eadf`:
    attempt 1 job `90068647392` (6m07s), attempt 2 job `90074181421`
    (6m06s), and attempt 3 job `90075895290` (6m25s). This closes the
    reproducing 2-of-3 failure. If it ever recurs, test explicit
    `FFTW.UNALIGNED` plans next, then prior memory corruption.

### New items raised by the 2026-07-25 wave

6. 🟢 **`TransModal` `DimensionMismatch` for `full=true` + npol=2 + plasma —
   FIXED 2026-07-27.** The initial diagnosis was wrong: Cubature merely
   rethrew the nonlinear callback's exception. The example passed a vector
   example field to `PlasmaCumtrapz`, which therefore allocated vector
   `P`/`J`/phase buffers, while the `components=:xy` transform passed an N×2
   `Et`. The example now constructs the response with
   `zeros(length(grid.to), 2)`; `PlasmaCumtrapz` checks its stored buffer
   shape and emits a focused constructor diagnostic; and
   `test/test_transmodal_vector_plasma.jl` proves both the diagnostic and a
   finite, nonzero full=true/npol=2/plasma transform whose plasma control
   effect exceeds 1e-8. Focused test 8/8, `sim-multimode` 41/41, and the
   corrected shortened end-to-end example completed in 39 accepted steps.
7. 🟢 **Prepare and publish `v1.0.1` — RELEASED 2026-07-28.** Release commit
   `b991d7c` passed the 16/16-job test matrix (`30360587278`) and
   documentation (`30360585023`) before tagging. Release workflow
   `30379620216` built and published canonical Linux x86_64, Apple Silicon,
   and Windows x86_64 `libamalthea-*` assets. The initially assembled
   checksum manifest exposed a Windows formatting defect (single separator
   space plus CRLF): the Julia installer's whitespace parser accepted it,
   but GNU `sha256sum -c` skipped that line. The published manifest was
   replaced with a two-space/LF form verified against all three downloaded
   binaries, and `release.yml` now generates that portable form directly.
   Release: <https://github.com/vdiego28/Amalthea.jl/releases/tag/v1.0.1>.
8. ⚪ **GPU `err` estimate uses a placeholder.** `weaknorm_elem_kernel` passes
   `field_d` as both the "old" and "trial new" field because `step()` has no
   pre-acceptance trial solution. Harmless under fixed-step (`stepcontrol_pi`
   forces acceptance), and now printed as a diagnostic rather than asserted,
   but unproven for adaptive stepping in general.
9. 🟢 **`AMALTHEA_NATIVE_GPU=on` process-wide silently reroutes CPU-native
   equivalence tests onto the GPU — FIXED and verified on hardware
   2026-07-26.** The five vulnerable files (`test_native_phase1.jl`,
   `test_native_phase2.jl`, `test_native_phase8.jl`,
   `test_native_fftw_wisdom.jl`, `test_native_dense_order5.jl`) now pin
   `withenv("AMALTHEA_NATIVE_GPU" => "off")` around every `RustNativeStepper`
   construction and assert the choice with a counted
   `@test !RK45._gpu_native_eligible(...)`. Measured on the RTX 5060 Ti: the
   full `rust` group under `AMALTHEA_USE_RUST_CUDA_NATIVE=1
   AMALTHEA_NATIVE_GPU=on` is **42269 pass / 1 broken / 0 failures** (the 18
   failures it replaces are agent-measured, matching the count recorded
   independently on 2026-07-25; only this post-fix state was lead-verified),
   and the default-env run has **identical totals** — proving no
   test was disabled — while the GPU tests still execute on the GPU. Two extra
   instances the original count missed: `test_native_phase8.jl` was passing
   only by tolerance luck (1.7e-9 vs an expected ~1.6e-11, under a loose 1e-8
   bound) and `test_native_dense_order5.jl`'s GPU testitem was comparing GPU
   against GPU. Record:
   `docs/dev/native-port/portlog-inbox/gpu-env-pinning.md`. Original report
   below.
   Found 2026-07-25 while
   verifying item 1: running the whole `rust` group with
   `AMALTHEA_USE_RUST_CUDA_NATIVE=1 AMALTHEA_NATIVE_GPU=on` produces **18
   failures** (`test_native_phase1.jl` 6, `test_native_fftw_wisdom.jl` 3,
   `test_native_phase2.jl` 1, and others), while the same tree under the
   default env is fully green (7-group gate exit 0, `rust` 42247/42248).
   Cause: those tests construct a `RustNativeStepper` and assert CPU-native
   tolerance tiers, but have no `withenv` guard pinning the backend — `on`
   bypasses the `:auto` size/shape dispatch that normally protects them.
   Only `test_native_cuda.jl` sets its backend explicitly. **Not a
   regression from the 2026-07-25 wave and not a GPU-correctness problem**
   (`test_native_cuda.jl` passes 31/31 under exactly that env), but it makes
   "run the suite on the GPU" quietly unusable as a verification technique.
   Fix by pinning `AMALTHEA_NATIVE_GPU=off` (or asserting the chosen
   backend) in the CPU-native phase tests.
10. 🟢 **Test discovery recurses into `.claude/worktrees/` — FIXED
    2026-07-26.** `test/runtests.jl` now filters out any test item whose file
    lives under a `.claude` path component (matched via
    `splitpath(relpath(...))`, not a bare substring, so a checkout path that
    merely contains the string ".claude" can't misfire), in both the `All`
    branch and the tagged branch — the `All` branch previously ran
    `@run_package_tests` with no filter at all. `amalthea/tests/*.jl` (the
    auto-discovered half of the `rust` safety net, `CLAUDE.md`/this file
    line ~1182) has no `.claude` path component and stays discovered.
    **No leftover worktrees existed to reproduce with, so the confounder was
    constructed by hand:** a throwaway `.claude/worktrees/fake/test/`
    holding one copy of `test_noise.jl` (`:fields`). Measured on the
    `fields` group: **432/432 with the fix reverted** (334 baseline + 98 from
    the duplicated file) vs. **334/334 with the fix applied**, confounder
    still present — matching the documented `fields` baseline exactly. The
    throwaway directory was then deleted (`git status` clean of it, never
    `git add`ed) and a final clean-tree run reconfirmed **334/334**.
    **Provenance (per the item-9 convention):** the two confounder-present
    numbers (432/432, 334/334) are **agent-measured**; the lead independently
    re-ran the clean tree and confirms **334/334 in 2m37.6s** — which is the
    number the before/after pair is meaningful relative to. Checked
    the parallel/gate path too (`test/parallel_group_tests.py`,
    `test/run_full_gate.py`): `discover_group_files()` uses a **non-recursive**
    `TEST_DIR.glob("*.jl")` scoped to `test/`, so it can never see
    `.claude/worktrees/.../test/*.jl` — not affected. Its worker subprocess
    (`test/run_group_bucket.jl`) does call `@run_package_tests` and re-walks
    the repo, but already carries its own `in_this_checkout` guard (commit
    `fe08fa9`, predating this fix) — so only the plain serial
    `test/runtests.jl` entry point (the one `AGENTS.md` §3 step 5 tells
    every agent to run) had the gap. No changes needed to either Python
    script. Record: `docs/dev/native-port/portlog-inbox/test-discovery-worktree-exclusion.md`.
    **Separate discrepancy surfaced while verifying this, NOT fixed here and
    not the one §5 documents:** the two guards are not the same predicate.
    `run_group_bucket.jl`'s is `dirname(abspath(f)) == THIS_TEST_DIR` (exact
    directory), while `runtests.jl`'s new one only excludes `.claude`
    components — so the serial path discovers the four `tags=[:rust]`
    `@testitem`s in `amalthea/tests/` (`test_julia_ffi.jl`,
    `test_stepper_dispatch.jl`, `test_scans_io.jl`, `test_gpu_cuda.jl`) and
    the parallel/gate path does not. `VANILLA_LUNA_ISSUES.md` §5's
    "one-shared-process vs many-processes" caveat is about the **FFTW
    wisdom pool-channel** delta (42104 vs 42087), a different cause — it
    does not cover this file-set divergence. So serial and parallel `rust`
    counts still legitimately differ, for two independent reasons rather
    than one. Worth reconciling if `rust` counts are ever compared across
    the two entry points; harmless otherwise, and pre-existing.
    Original report below.
    Running a test
    group from the repo root while agent worktrees exist inflates counts —
    `@run_package_tests` finds each nested worktree's copy of every test
    file. Measured 2026-07-25: the `examples` group reported **120/120** from
    the root versus the true **20/20** in a clean worktree. Harmless to
    correctness (everything passes either way) but it makes assertion counts
    meaningless as a regression signal, which this backlog relies on
    throughout. Either prune stale worktrees before gating, run the gate from
    inside a worktree, or exclude `.claude/` from discovery.

Explicitly parked, and therefore **not** resume points without a new user need:
multi-mode `StepIndexMode` (no consumer), the full SoA conversion (~1% ceiling),
the cold-start standalone CLI (porting all Julia setup has negative ROI), and
direct PPT/direct-error-coefficient rewrites (premises did not survive study).

## Completed work — status index

Full narrative for everything below lives in
[`ARCHIVE.md`](ARCHIVE.md); section names there are unchanged, so a source
comment citing "Phase E.3" or "S1 item 6" still resolves. The remainder list
below preserves each item's disposition; only entries explicitly marked open
are live.

### Improvement plan (2026-07-02 review) — Phases A-J

Phased plan from the fork-vs-upstream review ([`REVIEW.md`](REVIEW.md) —
fully executed, kept as provenance). Gate for every phase: full
`LUNA_TEST_GROUP=All` suite green.

| Phase | Scope | Status |
|---|---|---|
| A | Upstream sync (#428 ellipse angle, #427 SSHExec `files`, `upstream_sync.yml`) | ✅ |
| B | Correctness & parity (`_safe_n`, Rust `Emax` clamp, density z-independence guard) | ✅ |
| C | Make the default workload actually native (decouple the ionisation LUT from its opt-in toggle) | ✅ |
| D | Native scope: EnvGrid + plasma/Raman in more geometries | ✅ |
| E | Native scope: modal generality (`full=true`, TE/TM/`n>1`, npol=2, tapered radius) | ✅ |
| F | Native scope: Raman completions (`thg=false`, `RamanPolarEnv`) + z-dependence | ✅ |
| G | Platform & CI robustness (Windows scan lock, CI benchmark job) | ✅ except GPU CI — see "GPU CI coverage" under Open items |
| H | Upstream contributions | 3 of 4 sent; `pointcalc!` race fix not actionable (upstream doesn't thread it) |
| I | Close remaining native-port gaps (incl. the 🔴🔴 missing plasma density factor) | ✅ except deliberately parked I.5b (`StepIndexMode`) |
| J | Post-completion audit (2026-07-08) | ✅ all items closed; J.6(c) measured and rejected 2026-07-25 |

### Suggestions backlog — closed tracks

| Track | Scope | Status |
|---|---|---|
| S1 | Hot-loop CPU performance (FFTW wisdom, fused RK45 accumulation, de-branched Kerr, BLAS-3 QDHT, SoA spike) | ✅ all 6 items resolved or deliberately parked |
| S2 | Threading the native RHS (radial, modal, free-space) | ✅ closed 2026-07-22 |
| S4 | Architecture cleanups (`BackendConfig`, `RK45.check_ffi`, explicit accessor seams) | ✅ gate closed 2026-07-11 |
| S5 | Numerics options (mixed precision, deterministic mode, order-5 dense output) | ✅ all 3 items resolved, closed 2026-07-23 |

S3 and the release/example remainders of S6 stay live below; S2 closed
2026-07-22 and S5 on 2026-07-23.

### Remainders lifted out of the archived phases

1. 🟢 **Phase I.5a — `ZeisbergerMode`/`VincettiMode` multi-mode: native Rust
   port done (2026-07-22).** `RK45.jl`'s native modal guard now accepts both
   wrapper types — it unwraps to the inner `Capillary.MarcatiliMode` for the
   accessor fields the guard/setup read as raw struct fields (`kind`/`a`/
   `unm`/`ϕ`/`n`); `field`/`N` already delegate through generic dispatch. No
   `native.rs` change was needed: the native modal RHS never reads
   `neff`/dispersion, only the pre-baked `linop` (built by Julia before the
   RHS runs) and Marcatili field-synthesis parameters. `test/test_native_modal_zv.jl`
   (single-step 6e-18/exact, full-solve 3.5e-16/2.6e-15). See ARCHIVE.md
   Phase I item 5.
1b. 🟡 **Phase I.5b — `StepIndexMode` multi-mode: still native-ineligible.**
   No closed-form `neff` (numerical root-finding only), so it can't cheaply
   join the "bake dispersion into `linop`, unwrap for the field-synthesis
   accessors" pattern I.5a uses. Feasibility studied 2026-07-22 and found
   bounded but not currently worth building (no consumer constructs this
   config) — full design record and the exact narrow scope for a future
   implementer in [`native-port/PLANS.md`](native-port/PLANS.md) §5.
2. 🟢 **Phase J.3 — r2c/c2r halving for both FFT-conv Raman convolutions:
   done 2026-07-22, measured, bar cleared, kept.** Criterion spike
   (`amalthea/benches/raman_fft_r2c_bench.rs`) measured 1.8–2.8× across
   n_time_over=1024..65536 (~2.2× at the real `:SiO2` config's grid size).
   Implemented in both the native `:SiO2` kernel and Julia's `RamanPolarEnv`
   together, keeping the equivalence tier r2c-vs-r2c. `test/test_native_raman_sio2.jl`
   40/40 (native-vs-Julia 1.8e-13–3.6e-13).
3. 🟢 **Phase J.5 — consolidate the two resident Raman kernels' plumbing:
   done 2026-07-22, alongside J.3.** Extracted the duplicated `0.5·|E|²`
   intensity and `pto += E·(ρ·P)` accumulation loops (Steps 3b/3c in
   `rhs_mode_avg_env`) into shared free functions — pure code motion, no
   numerical change.
4. 🟢 **Phase J.6 — beyond-Luna math options, closed.** Feasibility studied
   2026-07-22 (full write-up in [`native-port/PLANS.md`](native-port/PLANS.md)
   §6): (a) direct DP5(4) error coefficients — **recommend against**, both
   backends already precompute `errest = b5.-b4` at load, so the premise's
   runtime cancellation doesn't exist; (b) direct PPT evaluation — **recommend
   against**, the true series has a BigFloat-quadrature tail that can't live
   in a hot loop and the LUT error is already below physical significance;
   (c) short-kernel Raman pad-shortening — **recommend against after
   measurement**: the response support is 76-86% of `n_time_over`, the real
   n=4096 configuration measured 0.98×, and projected end-to-end improvement
   was ~0.99-1.05×, below the >1.4× gate. The prototype was reverted.
5. 🟢 **Phase S5.3 — order-5 dense-output continuous extension: done
   2026-07-23.** The Calvo-Montijano-Rández (1990) order-5 tableau, wired
   into both steppers (extra-stage FFI + shared `interpC5`/
   `_dp5_extra_stages!` helpers). The 2026-07-22 attempt's blocker — order-4
   *and* order-5 interpolants both degrading as O(h²) — was **not** a test
   artifact: `step!` performed the FSAL carry k7→k1 at accept time, so
   `interpolate` was handed k7 in place of the finished interval's k1 and
   the continuous extension collapsed to first order. Inherited from
   upstream Luna and re-ported into all three Rust steppers; fixed in all
   four by deferring the carry to the top of the next step. The WIP's test
   additionally ran at h=2e-3, where the order-5 defect is already at the FP
   floor and no ratio means anything. Full postmortem, tableau provenance
   and measured orders: `native-port/portlog-inbox/dense-order5.md`.

---

## Suggestions backlog (imported from SUGGESTIONS.md, 2026-07-07)

Full detail (equations, rationale, per-item code sketches) stays in
`SUGGESTIONS.md` — this is the tracking summary, synced so status lives in
one place. S1, S2, S4 and S5 are now closed (S2 on 2026-07-22, S5 on
2026-07-23). S6's HDF5 writer and release machinery are implemented; its
cold-start CLI was studied and parked, while the v1.0.0 asset-name repair and
example repairs are complete; the v1.0.1 release remains in the queue above.
S1.5's BLAS-3 correctness
bug was fixed; the path remains opt-in because its ≥1.5× default-flip
benchmark was never demonstrated. **S3 is partially implemented and its
landed narrow slice is hardware-verified, but still lacks standing GPU CI**
(items 0 and 2 below):
the GPU-resident stepper work landed 2026-07-05/07 (see Phase G's "Open
items" entry and ARCHIVE.md's "Done (recent)") implements a narrow slice of S3
(mode-averaged RealGrid Kerr-only, no threading/dispatch-threshold/design
doc) — and did so *before* the GPU CI dependency S3 itself declares
("needs a CUDA machine... do NOT start before [GPU CI] exists, or it will
rot"), which is exactly what happened once already (uncommitted 2 days,
found broken until manually re-verified). GPU CI is still open (see "GPU
CI coverage" below) — treat S3's remaining scope (design doc, full
`NativeBackend` parity, threading, dispatch threshold, `test_native_gpu.jl`)
as still gated on it. **Update 2026-07-25:** the zero-nonlinearity defect
found on 2026-07-23 is fixed and non-vacuously hardware-verified; the episode
is why every future GPU change requires either standing CI or a recorded
manual hardware run.

**ISA / hardware dispatch — synced to actual code state (2026-07-07):**
`dispatch.rs`'s hardware cascade (CUDA → Vulkan → AVX-512 → AVX2 → NEON →
Apple AMX → portable) is real for *detection and selection*
(`is_x86_feature_detected!`, `dlopen`-based Vulkan/CUDA probes all
genuinely check the running machine). But grepping `amalthea/src/` for
`target_feature`/`_mm256_`/`_mm512_`/`std::arch` finds vectorized code in
exactly **one** file: `raman.rs`'s `solve_avx2`. Every other kernel (Kerr,
RK45 stage accumulation, window/norm broadcasts, QDHT) runs the same
portable-scalar code regardless of which `HardwarePath` `dispatch.rs`
selects — so today the dispatcher's choice of AVX-512/AVX2/NEON is
**cosmetic** for everything except Raman. This is suggestion 3 below,
tracked as S1.4; until it lands, "AVX-512 path selected" does not mean
"AVX-512 code ran."

### 🟢 S2 — Threading the native RHS (suggestion 2) — COMPLETE (all 4 items, closed 2026-07-22)
*Started 2026-07-10, reverted same day, root-caused and re-landed
2026-07-11 (radial, items 1-2). Modal (item 3) landed 2026-07-20;
free-space (item 4) landed 2026-07-22 — the whole track is now done. See
`docs/dev/native-port/PLANS.md §3` for the full phased plan.*

**Phase 3 REVERTED 2026-07-10, RE-LANDED 2026-07-11 — root cause was a
missing GC root, not a Rust-side memory-safety bug in the parallel code
itself.** After Phase 3 was committed/pushed (`d15a25c`), a post-hoc
wall-clock benchmark (`bench_threads.jl`-style: `n_threads=1`→4 at N=32
then N=128, plasma enabled throughout, same process) surfaced an
intermittent segfault inside `PptIonizationRate::rate`. The revert
commit's isolation experiment (Kerr-only config never crashed) correctly
implicated `apply_plasma_radial`'s parallel branch, but its diagnosis —
"an out-of-bounds write during the `n_threads=4` plasma call corrupts the
allocator's view of nearby memory" — was wrong about *where* the bug
lived.

**Actual root cause, found 2026-07-11 by installing ASAN
(`rustup +nightly` with `-Z build-std` — plain `-Z sanitizer=address`
silently fails to catch heap corruption on this toolchain because the
prebuilt stdlib allocator shim isn't itself instrumented) and Valgrind,
then reproducing the crash directly:**
- An isolated Rust-only repro of `apply_plasma_radial()` alone (40 varied
  cycles, real ASAN instrumentation) never crashed — ruling out a
  self-contained out-of-bounds write in that function.
- The **full pipeline** (checked out `d15a25c` into a worktree, ran a
  real `RustNativeStepper` radial+plasma solve at `native_threads=4`)
  reproduced a genuine `SIGSEGV` reliably. `coredumpctl`/gdb's backtrace
  showed a rayon worker thread segfaulting *inside*
  `PptIonizationRate::rate()`, called concurrently by multiple worker
  threads via `plasma_rate_at`/`rayon::iter::plumbing::bridge_producer_consumer`.
- Manual review of `CubicSplineLUT::evaluate()`'s binary search and its
  `get_unchecked(low)` call: logically correct and thread-safe as
  written — `low` is provably `< segments.len()` by the search's own
  invariant, and it only reads immutable `&self` data with no interior
  mutability. So the crash meant the `Vec<SplineSegment>`'s own heap
  metadata (pointer/len/capacity) was **already corrupted** before this
  call ran.
- Tracing how `plasma_ion_ptr`/`plasma_adk_ptr` (`*const
  PptIonizationRate`/`*const AdkIonizationRate` in `native.rs`) get set
  found the real defect: `native_set_plasma_params` stores this raw
  pointer *directly*, re-dereferencing it on every future `native_step`
  call for the sim's entire lifetime — unlike every other kernel's
  setup (Raman/dispersion oscillator data), which Rust copies into its
  own `Vec` once and never touches the original pointer again. The
  pointee is Julia-allocated-and-owned: `Ionisation.jl`'s
  `RustIonizationHandle`/`RustAdkHandle` carry a GC finalizer
  (`free_ppt_ionization_lut`/equivalent) that frees the Rust memory.
  `RustNativeStepper` (`RK45.jl`) never stored a Julia-level reference
  to `irf`/`irf.rust_handle` after construction — only the raw pointer
  value crossed the FFI boundary. Since Julia's `ccall` releases the GC
  safepoint for the duration of a foreign call (letting other Julia
  threads/the GC run concurrently, with *no visibility into native Rust
  threads* still holding the pointer), the handle was eligible for GC
  finalization **at any point after construction**, including mid-solve
  while rayon worker threads were still concurrently dereferencing it —
  a textbook FFI use-after-free. Threading only widened the race window
  (longer-lived worker threads doing more concurrent work, more
  allocation pressure to trigger a GC cycle during the window); it did
  not cause the unsoundness — the same latent bug existed on the
  sequential path too, just with a race window too small to observe in
  practice.
- **Fix** (applied to `main` independently of re-landing threading, since
  it's a real latent bug regardless): added a `_gc_roots::Vector{Any}`
  field to `RustNativeStepper` (`RK45.jl`) that the constructor populates
  with every Julia handle object (`irf.rust_handle`) whose raw pointer
  the Rust-side `NativeSim` stores persistently, across all three
  geometries (mode-avg/radial/modal) and both ionization models
  (PPT/ADK) — six call sites total. Nothing reads `_gc_roots`; its only
  job is keeping those objects Julia-reachable for the stepper's whole
  lifetime, preventing early finalization.
- **Verified the fix holds**: re-applied the Phase 3 diff (`fftw.rs`'s
  `unsafe impl Sync`, `native.rs`'s `ReadOnlyPtr`/`plasma_rate_at`/
  parallel FFT-and-plasma branches, `RK45.jl`'s `native_threads` kwarg,
  both bit-identical tests) on top of the GC-root fix, then re-ran the
  exact crash-reproducing script for 8 cycles alternating N=32/N=128 and
  `native_threads` 1/4 with `GC.gc()` forced between every cycle (to
  maximize GC pressure) — all 8 completed cleanly, versus crashing at
  cycle 2 before the fix. `n_threads=1` and `n_threads=4` results were
  bit-identical across every repeated size (confirmed via `yn` norm
  equality, not just non-crashing). Full 7-group gate green (see "Done
  (recent)").
- **Lesson for future FFI kernel wiring**: any `native_set_*_params` call
  that stores a raw pointer into Julia-owned, Rust-allocated memory
  *persistently* (re-dereferenced across multiple future calls, not just
  copied once at setup) needs its Julia-side handle rooted somewhere
  that outlives construction — `RustNativeStepper._gc_roots` is that
  place going forward. Kernels that copy data into their own `Vec` at
  setup time (the common pattern) don't need this.
1. 🟢 **Done 2026-07-10.** `native_set_threads(handle, n)` FFI, wired from
   `Threads.nthreads()`, default 1 (bit-identical to today — verified via
   full 7-group gate, purely additive plumbing, `n_threads` not yet read
   by any RHS code).
2. `rhs_radial`: rayon over radial nodes, each node's own FFTW
   new-array-execute call against one shared plan; one scratch slab per
   rayon worker (never shared — this is precisely the bug the Julia
   `Threads.@threads` `pointcalc!` race had, see Phase B / ARCHIVE.md's "Done (recent)").
   **Re-scoped after investigation:** an Explore survey found the two
   per-column FFT loops and `apply_plasma_radial`'s per-column loop
   already operate on disjoint slices of matrix-shaped
   (`n_time_over*n_r`) buffers, not shared per-call scratch — safe for
   `par_chunks_mut` without new per-worker scratch structures. Only
   `apply_raman_radial`'s single shared `raman_solver` (and, when
   `raman_thg==false`, shared Hilbert scratch) has the genuine
   shared-mutable-state hazard the backlog warns about. **🟢 Done
   2026-07-19 (S2 Phase 4 item 1):** parallelized by partitioning the
   r-columns into ≤`n_threads` *contiguous* groups, each rayon task owning
   its **own cloned `TimeDomainRamanSolver`** and **own Hilbert scratch**
   (no `current_thread_index`, no interior mutability, provably disjoint
   column slices; `solve()` resets oscillator state at entry, so a cloned
   solver is bit-identical to the shared one). No new GC-root hazard — the
   solver is Rust-owned and only *cloned*, not a persistent raw pointer
   into Julia-owned memory (contrast the plasma-pointer UAF fixed on
   `main`). Verified to the full S2 bar: `n_threads=1`-vs-`4`
   **bit-identical** (`s.yn` exact) in `test/test_native_radial_raman.jl`,
   plus an 8-cycle forced-`GC.gc()` stress repro with no crash.
   Modal (item 3) and free-space (item 4) threading remain the open
   follow-ups (the latter needs an isolated multi-threaded FFTW plan under
   `PLANNER_LOCK`: `fftw.rs`'s `unsafe impl Sync` only holds for
   single-threaded plans, so an `nthreads>1` plan would deadlock under the
   existing concurrent `fftw_execute_dft` path).
   **Measured 2026-07-10** (temporary `Instant` profiling, reverted after
   reading, same discipline as S1.6): FFT-loop + plasma-loop share of
   `rhs_radial` time, at N=32/N=128 r-points, with/without plasma:
   | N | plasma | FFT | QDHT | plasma | other |
   |---|---|---|---|---|---|
   | 32 | off | 35.9% | 46.7% | — | 17.4% |
   | 32 | on | 20.9% | 28.3% | 40.5% | 10.2% |
   | 128 | off | 18.6% | 72.6% | — | 8.8% |
   | 128 | on | 14.1% | 54.8% | 24.5% | 6.6% |

   Unlike S1.6's ~2% ceiling, FFT+plasma combined is **38-61% of
   `rhs_radial` time** for plasma-enabled configs — clears the bar for
   proceeding. QDHT dominates the rest (already internally
   parallel/BLAS-backed via S1 item 5's BLAS-3 QDHT fix — out of scope
   for this item).
3. 🟢 **Modal: DONE 2026-07-20.** rayon over the cubature batch's nodes
   inside both `integrand_v` callbacks (`modal_integrand_v` /
   `modal_integrand_v_full`) — cubature's own adaptive node *placement* stays
   sequential/deterministic; only the per-node integrand evaluation is
   parallelized. **Measured first** (temp `Instant` counters on `rhs_modal`,
   add/measure/revert per S1.6/S2-Phase-2 discipline): the integrand loop is
   **90.3%** (full=false, 1 mode) / **95.6%** (full=true) / **82.8%**
   (2-mode) of `rhs_modal` wall time — well above the proceed bar (radial was
   38-61%; S1.6 parked at ~2%). Batch sizes are small for full=false (~4
   nodes/batch) but each node is ~27-33µs of FFT-heavy work, so threading
   still nets a win; full=true batches ~25 nodes. **Measured wall-clock
   speedup (`native_threads`=1→4, this 12-core host, 300-step fixed-dt solve,
   min-of-3):** full=false 1-mode **1.31×**, full=false 2-mode **1.52×**,
   full=true **2.64×** — every config a genuine speedup (even the small-batch
   full=false regime S3.3/S1.6 warned could go net-negative), which also
   proves the parallel branch actually engages (the bit-identical test isn't
   vacuously passing on a silently-sequential path). No min-npt guard needed.
   **Refactor:**
   `rhs_modal_pointcalc` (a `&mut self` method scribbling on ~13 shared
   `self.modal_*`/`raman_*` scratch buffers) was split into a `Sync`
   read-only view (`ModalRO`: `&[..]`/`Copy`/`Option<&Plan>`, FFT wrappers
   already `Sync`) + per-worker `ModalScratch` (all written buffers, pooled on
   `self.modal_scratch_pool`, entry 0 = sequential path) and a free-standing
   associated fn `modal_pointcalc(&ro, sc, r, θ, out)` used by BOTH paths (one
   code path, no duplicated 270-line body). Nodes split into `≤ n_threads`
   contiguous groups; each group's `out[p*fdim..]` is disjoint with no
   cross-node reduction ⇒ **bit-identical** `n_threads`=1-vs-4 (the S2 gate,
   not a tolerance). Raman-modal threaded too: each worker's `ModalScratch`
   carries its **own cloned** `TimeDomainRamanSolver` + Hilbert scratch (same
   discipline as radial item 1; `solve()` resets state at entry ⇒ clone ==
   shared, bit-identical); the Hilbert FFT plan is shared read-only
   (`fftw_execute` thread-safe against distinct buffers). No new GC-root
   hazard — the solver is Rust-owned/cloned, not a persistent raw pointer into
   Julia memory (contrast the plasma-pointer UAF fixed on `main`). New
   `test/test_native_modal_threading.jl` (Kerr full=false/full=true/2-mode/
   npol=2, Raman :N2, + forced-`GC.gc()` stress loop): all bit-identical
   1-vs-4, native-vs-Julia unchanged at ~2e-16 (Kerr) / ~1e-6 (Raman ADE-vs-
   FFT floor). 70/70 Rust unit tests; clean `-D warnings` build.
4. 🟢 **3-D free-space FFT: DONE 2026-07-22 — S2 track now fully closed.**
   `fftw_plan_with_nthreads`/`fftw_init_threads` resolved from the *same*
   combined `libfftw3.so` FFTW_jll already ships (no separate
   `libfftw3_threads` needed on this build; silent fallback if a future build
   splits it out). The stated `unsafe impl Sync` blocker dissolved rather
   than needing a workaround: free-space has no per-column loop — the joint
   3-D transform runs once per RK stage from one thread — so `RealFft3d`/
   `ComplexFft3d` never need `Sync` and deliberately don't implement it
   (unlike the 1-D types rayon workers share). Threading is baked into the
   plan via a new `with_nthreads_plan` (symmetric to `with_single_threaded_plan`,
   under `PLANNER_LOCK`, restoring the global planner thread count on exit so
   the 1-D per-column plans stay single-threaded). Measured 2.46–2.51× on the
   isolated transform, 1.43–1.51× end-to-end at every size. `n_threads=1`-vs-`4`
   bit-identical, `test/test_native_free_threading.jl`; `free` group 197/197.
5. 🟢 Error-norm reduction stays sequential (determinism, ties to S5.2) —
   confirmed already sequential (`weaknorm_c64`), untouched by this item.
- Gate: universal + fixed-step equivalence under `JULIA_NUM_THREADS=4` +
  ≥2× radial/free benchmark at 4 threads + n=1 bit-identical to pre-track.
  For radial specifically: since the parallelized loops write disjoint
  memory with no cross-column reduction, `n_threads=1` vs. `n_threads=4`
  must be **bit-identical**, not merely within tolerance — a stronger,
  more testable guarantee than typical parallel-code equivalence.

### 🟡 S3 — GPU-resident propagation (suggestion 1) — the broken slice is fixed (item 0); scope beyond mode-averaged Kerr(+PPT) is still unbuilt
*Large (5+ sessions). Plan's own stated dependency (GPU CI) is **still** not
met — see "GPU CI coverage" below, and note that item 0 is precisely what
that gap allowed to happen. This machine has real GPU hardware
(RTX 5060 Ti, driver 610.43.02, CUDA 13.3) usable for manual verification of
future slices, confirmed 2026-07-11, 2026-07-23 and 2026-07-25. On the
sandbox: `nvidia-smi` needed the sandbox disabled in earlier sessions, but
reached the driver directly in the 2026-07-25 agent session — treat this as
environment-dependent, not a fixed rule.*
Already landed (2026-07-05/07, ahead of the plan's own sequencing): the
`NativeBackend` trait extraction, `CudaNativeSim` scoped to mode-averaged
RealGrid Kerr-only (not "+plasma" — see item 1 below, plasma was never
implemented), verified on real hardware, wired behind
`AMALTHEA_USE_RUST_CUDA_NATIVE=1`. Still open, per the original design:

0. 🟢 **The GPU-resident RHS contributed no nonlinearity at all — found
   2026-07-23, FIXED and hardware-verified 2026-07-25.** The fix is
   summarized in the resume queue at the top of this file and recorded in
   full in `native-port/portlog-inbox/gpu-nonlinearity.md` + `PORT_LOG.md`.
   It also closes **item 6** below (the `n_time`-vs-`n_time_over` sizing gap,
   which is not separable from Steps 1/5) and fixed a second, undiagnosed
   bug: `CudaNativeSim::set_field` never seeded `ks_d[0]`, so the first
   `step()` read uninitialized device memory. **The diagnosis below is
   retained as-written for provenance — it is what the fix was built from.**
   For the exact config
   `test_native_cuda.jl`'s Kerr-only testitem uses (125µm He capillary,
   1 bar, 800nm, 1µJ, 30fs), the GPU backend's stage derivatives measure
   `max|kᵢ| = 3.5e-13` against the CPU backend's **12225**, and its accepted
   step equals pure linear propagation `exp(L·h)·y₀` to 15 digits — i.e. the
   nonlinear term is absent, not merely inaccurate. Reproduced on real
   hardware (RTX 5060 Ti, driver 610.43.02), and confirmed **pre-existing**
   by re-measuring against a build with the 2026-07-23 `cuda_native.rs`
   changes reverted (identical numbers to the last digit).
   - **Why every GPU test passes anyway:** `test_native_cuda.jl` asserts
     `rel_solve < 1e-3`, but the *entire* nonlinear effect for that config is
     ~4.5e-4 over the solve (1.3e-4 per step). The tolerance is looser than
     the physics being tested, so "GPU matches Julia" is vacuously true —
     the same failure mode as the Phase I plasma-density bug
     (`VANILLA_LUNA_ISSUES.md` §1), where every equivalence test ran in a
     regime where the missing term was negligible.
   - **Not yet diagnosed.** Prime suspects, in order: `set_mode_avg_params`
     discards `pre_re`/`pre_im`/`beta`/`sidx`/`nlscale`/`sqrt_aeff`
     (all `_`-prefixed there) which the mode-averaged RHS needs; the
     `n_time > 0 && fft_r2c != 0 && fft_c2r != 0` guard around the whole
     nonlinear block, which fails *silently* if `cufftPlan1d` fails (its
     return code is discarded); and the `n_time`-vs-`n_time_over` sizing
     mismatch (item 6) feeding the cuFFT plans.
   - **Resume diagnosis (2026-07-25, traced against the CPU RHS):**
     `cufftExecZ2D`/`cufftExecD2Z` return codes are now checked, but both
     `cufftPlan1d` return codes are still discarded. More importantly,
     `set_mode_avg_params` visibly discards `_owin`, `_sidx`, `_pre_re`,
     `_pre_im`, `_beta`, `_nlscale`, and `_sqrt_aeff`, and the GPU RHS has
     no equivalents of CPU Steps 2 and 5–7. Implement those missing
     scaling/normalization/window steps before expanding GPU scope, then
     tighten the test below the config's independently measured nonlinear
     share so a zero-nonlinearity implementation must fail.
1. 🟢 **Done 2026-07-11.** Design doc reconciliation
   (`docs/dev/native-port/GPU.md`). Rewrote §8 (was still the stale
   2026-07-05, pre-hardware "untested" text — the 2026-07-07 verification
   pass and Julia wiring had updated `BACKLOG.md` but never made it back
   into this doc) and §7 (claimed V1 scope was "Kerr (+plasma)"; actual
   shipped scope is Kerr-only — every `set_*_params` beyond
   `set_mode_avg_params`, including `set_plasma_params`, returns `-1`,
   confirmed by reading `cuda_native.rs` directly). §4's `enum{Cpu,Gpu}` vs
   the actual `Box<dyn NativeBackend>` deviation: kept as a documented,
   deliberate divergence (dynamic-dispatch cost is one vtable call per
   accepted step, immaterial next to actual kernel-launch cost; it's also
   what lets `CpuNativeSim`/`CudaNativeSim` share one FFI surface) rather
   than treated as a TODO to "fix" — the doc previously implied this
   should eventually match §4, which it never needs to. No code changed;
   documentation-only.
2. 🟡 **PPT plasma done 2026-07-11 (mode-avg only); Raman, radial/modal/
   free-space, ADK still open.** Added to `CudaNativeSim`
   (`cuda_native.rs`/`kernels.cu`/`cuda.rs`): `set_plasma_params` uploads
   the same `SplineSegment` LUT format `PptIonizationRate::rate_vector_gpu`
   already uses (reused directly, no new upload format), then `step()`
   runs a 5-kernel sequence per RK stage — `ppt_ionization_kernel` (reused
   from the standalone `AMALTHEA_USE_RUST_IONISATION` path, parallel over
   `n_time`) → `plasma_fraction_kernel` (fused cumtrapz+ρ-transform,
   single-thread sequential scan) → `plasma_phase_kernel` (parallel) →
   `plasma_current_kernel` (fused cumtrapz+loss-current, single-thread) →
   `plasma_polarization_kernel` (fused cumtrapz+accumulate-into-Pto,
   single-thread) — mirroring `native.rs`'s `apply_plasma_real` exactly.
   Single-thread sequential kernels for the 3 cumtrapz stages, not a
   work-efficient parallel prefix scan (GPU.md's original sketch,
   item 4 below) — a deliberate V1 tradeoff: `n_time` (~2^13-2^16) is
   small enough that one thread looping over it is negligible next to this
   step's FFT/launch cost at mode-averaged scale; would matter more for
   radial's much larger per-column state, not in scope here. `_gpu_native_eligible`
   (`RK45.jl`) relaxed to allow exactly one plain Kerr response plus at
   most one PPT-only plasma response (ADK still returns `-1` from
   `set_plasma_params_adk`, unimplemented) — the shared FFI call
   (`native_set_plasma_params`) needed zero Julia-side changes beyond the
   eligibility gate, since it already dispatches through the same
   `Box<dyn NativeBackend>` handle both CPU and GPU sims share.
   **Real pre-existing bug found and fixed while wiring this in**:
   `rhs_mode_avg_real_kernel`'s call site passed `(eto_d, pto_d, ...)` but
   the kernel's own declared parameter order is `(pto, eto, ...)` —
   swapped, so the kernel's `pto` write target was actually bound to
   `eto_d` (overwriting the just-FFT'd field with the Kerr result, silently
   discarded before ever being read again) and its `eto` read source was
   bound to `pto_d` (stale/uninitialized memory, not the field) — meaning
   every accepted step's forward FFT (`cufftExecD2Z` on `pto_d`) transformed
   whatever was left in `pto_d` from a previous call, not the real Kerr
   polarisation. Present since the 2026-07-05/07 GPU work, never caught by
   the existing Kerr-only equivalence test because that test's energy is
   weak enough that the resulting error stayed under the test's existing
   ~4.5e-4-driven `<1e-3` tolerance regardless. Fixed by correcting the
   argument order to match the kernel's declaration (also documented
   in-line in `cuda_native.rs`, right at the call site, so it can't silently
   regress again). Every new kernel-arg array is bound through named `let`
   locals, not inline temporaries — the exact UB pattern (`&mut {expr} as
   *mut _`) that caused a real `SIGSEGV` inside `libcuda.so` in the original
   2026-07-07 verification pass; caught in review before this landed.
   **Verified on real CUDA hardware** (RTX 5060 Ti): new
   `test/test_native_cuda.jl` testitem (`"...Kerr+plasma"`) passes,
   `rel_solve ≈ 2.0e-2` at `gas=:Ar, energy=6e-6`. This is *not* the same
   tolerance tier as the Kerr-only sibling test (~1e-3) — diagnosed, not
   assumed: the CPU-resident native path (`AMALTHEA_USE_RUST_NATIVE=1`, no
   CUDA — proper `n_time_over`-sized buffers) matches the Julia oracle for
   this exact config to `1.3e-16`, and sweeping energy 1e-7→6e-6 (60×)
   showed `rel_solve` scaling almost exactly linearly with energy at every
   point — the signature of the same pre-existing, documented
   `n_time`-vs-`n_time_over` Kerr buffer-sizing gap (item 6 below; GPU.md
   §8) amplified roughly 40× by plasma's Keldysh-exponential sensitivity to
   field amplitude, not a new bug. Full 7-group gate green, zero
   regressions (rust 42117/42117 = 42113 baseline + 4 new assertions).
3. 🟢 **Done (2026-07-16).** Problem-size dispatch threshold — measured, not
   guessed, on real hardware (RTX 5060 Ti). Benchmarked `native_step`
   CPU-vs-GPU directly (mode-avg, RealGrid, fixed-step, 10-iteration average
   after warmup) across a size sweep before writing any dispatch code, per
   this backlog's own "benchmark first" discipline (the r2c/c2r item, #3 in
   Phase I above, was investigated the same session and *failed* that same
   gate — no benchmark existed there, so it was correctly left alone).
   **Two very different regimes, not one crossover:**
   - **Kerr-only**: GPU is slower below ~n=8,193 (breakeven ~1.3x there),
     then wins increasingly — 5x at n=16,385, 14x at n=65,537, 27x at
     n=262,145. Dominated by cuFFT throughput at scale; small-n loss is
     CUDA kernel-launch/sync overhead (`cuda_native.rs::step`'s
     `launch_checked` synchronizes after every one of ~dozens of per-stage
     launches).
   - **Kerr+plasma (PPT)**: GPU is 20-30x *slower* than CPU at every size up
     to n=131,073 tested, and the gap *widens* with n — the opposite trend
     from Kerr-only. Root cause: `cuda_native.rs`'s plasma kernels
     (`plasma_fraction_kernel`/`plasma_current_kernel`/
     `plasma_polarization_kernel`) are single-GPU-thread sequential scans
     (item 2's documented V1 tradeoff, item 4/6 below), so they don't
     benefit from n the way cuFFT does — they're pure serial overhead that
     scales with grid size. No crossover exists in the tested range, so a
     single numeric threshold across both regimes would be actively
     misleading.
   - **Fix**: `AMALTHEA_NATIVE_GPU=off/on/auto` (`Config.jl`'s new
     `gpu_dispatch::Symbol` field), layered on top of
     `AMALTHEA_USE_RUST_CUDA_NATIVE`'s existing master opt-in (unchanged).
     `off` forces CPU; `on` restores the old unconditional-GPU behavior
     (kept for forcing GPU on a small/known config, e.g. reproducing a
     specific benchmark); `auto` (**new default**) requires a plasma-free
     config at `n >= _GPU_KERR_ONLY_N_THRESHOLD = 16384` (chosen with margin
     above the measured n=8,193 breakeven, since GPU time isn't cleanly
     monotonic in n at small sizes) — a plasma-bearing config is rejected by
     `:auto` outright, at any size, since no threshold is supported by data
     there. `RK45._gpu_native_eligible(f!, linop)` split into a pure
     config-shape check (`_gpu_kernel_supports`, unchanged logic) and the
     new size/policy-aware `_gpu_native_eligible(f!, linop, n)` (now 3-arg;
     `n = length(y0)`, threaded through from `RustNativeStepper`'s existing
     `n`). Full measured table and reasoning live in
     `RK45._GPU_KERR_ONLY_N_THRESHOLD`'s docstring, next to the code it
     justifies. Both existing `test_native_cuda.jl` GPU-vs-CPU equivalence
     tests now explicitly set `AMALTHEA_NATIVE_GPU=on` (they test raw kernel
     correctness at deliberately small/known configs, independent of the
     dispatch heuristic — including the Kerr+plasma test, which
     intentionally still drives the known-slow path for numerical
     verification). New `test/test_native_gpu_dispatch.jl` covers the
     `:off`/`:on`/`:auto` decision matrix directly (pure Julia-side logic,
     no `ccall`, so it runs without GPU hardware, unlike the sibling
     equivalence tests).
4. Raman ADE kernel, ADK plasma kernel, radial/modal/free-space geometries
   — or explicit `NativeIneligible`-style eligibility split keeping those
   configs on CPU for GPU v1, documented as such. A work-efficient parallel
   prefix scan for cumtrapz (superseding item 2's single-thread kernels)
   would matter here, at radial's much larger per-column plasma-state size.
5. `test/test_native_gpu.jl`-style coverage of the above, mirroring the
   phase-test structure used throughout the CPU native port.
6. The `n_time`-vs-`n_time_over` Kerr/plasma buffer-sizing fidelity gap
   itself (GPU.md §8) — not fixed by item 2 above, which worked within the
   existing buffer sizing rather than changing it.

### 🟢 S5 — Numerics options (suggestions 10, 11, 12) — COMPLETE (all 3 items, closed 2026-07-23)
*Item 2 done 2026-07-11 (re-scoped). Items 1 and 3 investigated 2026-07-19
— both re-scoped after measurement (item 1: bar not cleared, reverted;
item 3: backlog premise wrong, the DP5 5th-order continuous extension is
not the coefficient swap the entry implied — deferred as a larger item).
Item 3 then landed 2026-07-23, together with a fix for the FSAL/k1 bug that
had been holding *every* stepper's dense output at first order; see the
"Open remainders" list above and `native-port/portlog-inbox/dense-order5.md`.*
1. 🟢 **Done 2026-07-19 — measured, bar not cleared, reverted (S1.6
   discipline).** Mixed-precision spike (item 10). Added a timeboxed
   Criterion bench (`amalthea/benches/mixed_precision_bench.rs`, since
   reverted) microbenchmarking the precision-sensitive inner arithmetic of
   one accepted Dormand-Prince step — the two 7-term stage combinations
   (`yn = field + dt·Σ b5ᵢ·kᵢ`, `yerr = dt·Σ errᵢ·kᵢ`) plus the weak error
   norm (`native.rs`'s `native_step`, lines ~4272-4363) — in f64 vs. an
   f32-mixed variant (stages held/combined in f32, norm reduced in f64:
   the most generous case for the error estimate).
   - **Measured speedup (`target-cpu=native`, this 12-core host): ~1.0-1.06×**
     — f64 vs f32-mixed at n=8192: 65.8µs vs 64.8µs (1.02×); n=16384:
     132.9µs vs 131.4µs (1.01×); n=65536: 577µs vs 546µs (1.06×). Far
     below the >1.4× gate (a). The loop is compute/FMA-bound and already
     well auto-vectorised in f64, not the memory-bandwidth-bound loop that
     would have let f32's halved byte-count translate to speed — so f32
     buys almost nothing here.
   - Gate (b) would also fail regardless of (a): the error estimate is the
     b5−b4 near-total cancellation (TESTING.md §3 / CLAUDE.md's Phase-2
     gotcha — a ~1e-15 summation-order difference already amplifies into a
     ~20% relative `err` swing near the FP-noise floor, which the PI
     controller turns into a different step path). f32's ~1e-7 relative
     precision on the cancelling `Σ errᵢ·kᵢ` sum cannot keep the adaptive
     step sequence identical, so even a hypothetical speed win would change
     the propagation.
   - **Decision: do not implement; bench reverted** (only the numbers above
     retained, per the S1.6 add/measure/revert precedent).
2. 🟢 **Done 2026-07-11 — re-scoped, backlog's own premise was partly
   wrong.** Deterministic mode (item 11). Investigated before writing any
   code (per the S1.4/S4 pattern of checking the backlog's premise against
   current architecture, not just its literal wording):
   - **"Pinning `dispatch.rs` to the portable lane" — dropped.**
     `dispatch.rs`'s `SimulationEngine`/`HardwarePath` is a vestigial
     hardware-path *selector*, never wired into any RHS kernel's actual
     codegen (same finding class as S1.4: real vectorization comes from
     `target-cpu=native` at compile time, not a runtime branch this enum
     could gate). There is nothing to "pin."
   - **"Forcing sequential reductions" — the default native path already
     *is* run-to-run deterministic on one machine.** With S2 Phase 3
     reverted, no RHS code reads `n_threads` yet. Every parallel seam that
     exists today (the QDHT Rayon fallback, the older per-kernel
     `AMALTHEA_USE_RUST_QDHT` batch loops in `ffi.rs`) is embarrassingly
     parallel — each output row/column computed independently with no
     cross-thread reduction — and native FFTW plans only ever use
     `FFTW_ESTIMATE` (no wisdom-dependent plan selection by default, see
     S1 item 1). So a naive "two runs bit-identical" test would pass
     whether or not a new toggle did anything — and a first implementation
     attempt shipped exactly that vacuous test before catching it.
   - **The one real, addressable lever: process-global BLAS-eligibility.**
     `amalthea/src/blas.rs`'s `BLAS_API` is an `OnceLock`, populated only
     by the older per-kernel `AMALTHEA_USE_RUST_QDHT`+`AMALTHEA_QDHT_BLAS` path
     (`NonlinearRHS._init_rust_qdht_blas`). Once populated, it silently
     makes *every later* native-path radial-QDHT call in that process
     BLAS-eligible too, even though nothing in the native construction
     path asked for it — a process-global-state contamination hazard in
     the same family as S1 item 1's wisdom-pool finding. BLAS-3 `dgemm`
     (OpenBLAS/MKL) and the row-parallel Rayon fallback sum in a different
     order, so which one silently gets taken is a real,
     configuration-order-dependent effect on the result.
   - **Fix:** `native_set_deterministic(handle, bool)` FFI (`native.rs`,
     `NativeBackend` trait + `CpuNativeSim`/`CudaNativeSim` impls) forces
     the native-port radial `QdhtFfiHandle` to skip the BLAS branch
     regardless of `BLAS_API` state; a second FFI,
     `qdht_ffi_set_deterministic` (`ffi.rs`), does the same for the
     per-kernel handle — needed because that's the only call site that
     ever populates `BLAS_API` in the first place, so leaving it unwired
     would make the flag inert exactly where contamination originates.
     `deterministic::Bool` added to `Config.jl`'s `BackendConfig`
     (`AMALTHEA_NATIVE_DETERMINISTIC`, default off), read at both call sites
     (`RK45.jl`'s native construction, `NonlinearRHS._make_rust_qdht_handle`).
     **What this guarantees: BLAS-eligibility invariance** — the result no
     longer depends on whether some unrelated earlier construction in the
     same process touched `AMALTHEA_QDHT_BLAS`, nor on which BLAS
     implementation/thread-count Julia happens to have linked. **What it
     does NOT guarantee:** bit-identical results across different
     machines/CPU targets (`target-cpu=native` means a different build
     host takes a different SIMD/libm path); it is also not "fixing" a
     same-process run-to-run instability that was never observed to exist
     at these problem sizes.
   - **Test:** `test/test_native_deterministic.jl`, tagged `:rust`.
     Deliberately does *not* stop at "two runs bit-identical" (that alone
     can't distinguish a working flag from an inert one, as above) —
     T1 establishes the two-runs-bit-identical baseline before `BLAS_API`
     is ever touched; the test then explicitly contaminates process-global
     `BLAS_API` via `NonlinearRHS._make_rust_qdht_handle` (mirroring a real
     session that mixes the per-kernel and native-port Rust paths); T2
     asserts `deterministic=false` vs. `deterministic=true` now produce
     numerically *different* results (`s_blas.yn != s_det.yn`) — proof the
     flag actually gates the BLAS branch rather than being unreachable;
     T3 re-confirms bit-identical repeats under `deterministic=true` even
     after contamination; T4 confirms toggling back off doesn't crash and
     still produces a finite result. Gate: full 7-group suite —
     rust 42111/42111 (42101 baseline + this item's 10 new assertions,
     zero drift elsewhere), physics/sim-interface/sim-multimode/
     sim-propagation/io/fields all green (see ARCHIVE.md's "Done (recent)" for the
     dated full run).
3. 🟢 **Done 2026-07-23 — genuine order-5 continuous extension plus the
   FSAL/k1 correctness fix.** The landed implementation uses the
   Calvo–Montijano–Rández extra-stage tableau on both Julia and resident
   native steppers; see the open-remainders summary above and
   `native-port/portlog-inbox/dense-order5.md`.

   **Historical 2026-07-19 investigation (superseded by the implementation
   above):**
   The entry read: "Shampine's DP5 continuous extension in
   `native.rs::interpolate` and `RK45.jl`'s `interpC`, same commit — removes
   the Julia-vs-native saved-grid tolerance tier entirely." Two structural
   facts (config-independent, not just measured) make this incorrect:
   - **There is no `native.rs::interpolate`.** Dense output for the native
     stepper is reconstructed *in Julia* (`interpolate(s::RustNativeStepper,
     ti)`, `RK45.jl:2053`): it fetches the 7 resident RK stages via the
     `get_ks_stage` FFI and applies the *same* `interpC` quartic as
     `PreconStepper` (which reads them from `s.ks`), then re-expresses the
     polynomial at the query time via `native_apply_prop`. Rust exposes the
     stages and the propagator; the interpolation math is Julia-side for
     *both* steppers.
   - **Changing interpolant order therefore cannot change native-vs-Julia
     agreement.** Both sides evaluate the identical interpolant from the
     identical stages, so their saved-grid difference is the Phase-1
     native-vs-Julia *method* tolerance (FFT/summation-order + Rust-`exp`
     vs Julia-`exp` in the propagator), not an interpolation-order effect.
     Measured (Kerr-only default config, `test_native_phase8.jl`): dense
     output (saveN=50) native-vs-Julia rel = **2.2e-11**, essentially the
     same order as the saveN=2 endpoints, **1.1e-11** — no interpolation-
     driven tier exists to "remove." (The old test threshold was `1e-8`,
     ~3 orders looser than reality; tightened to `1e-9` this pass — the one
     shippable piece of the item. Aside: the endpoint comment claims
     "~1e-13"; the real figure is ~1e-11, 2 orders looser than documented —
     unrelated doc drift, left as-is.)
   - **A genuine 5th-order continuous extension is real but different work.**
     DP5's 7-stage FSAL "free" interpolant is provably *order 4* (Hairer &
     Wanner II.6; it's the same MATLAB `ntrp45`/scipy interpolant), so
     reaching order 5 requires *extra function evaluation(s)* per
     interpolated step (Shampine 1986), i.e. a lazy extra-stage machinery on
     both the resident Rust stepper (a new FFI to evaluate + return the
     extra stage) and Julia. Its benefit is better accuracy of dense output
     **against the true solution** — which no current test measures (the
     whole suite's only native dense-output check is the native-vs-Julia one
     above; there is no dense-vs-analytic tier where 4th-order interpolation
     error is the limiter). So it neither shrinks the native-vs-Julia tier
     nor tightens any existing test; it's a standalone accuracy improvement,
     deferred until someone wants it. **Future implementer:** add an extra
     stage to `native.rs` exposed via a new `get_extra_stage`-style FFI,
     port the order-5 continuous-extension coefficients into a shared Julia
     helper used by both `interpolate` methods, and add a *new*
     dense-vs-fine-reference test (not a native-vs-Julia one) to show the
     order gain.

### ⚪ S6 — Distribution & ecosystem (suggestions 9, 13, 14)
*Items 1-2 are implemented. Item 3 was measured and parked; the live S6 work
is the v1.0.0 asset-name repair/validation and the example repairs in the
resume queue.*
1. 🟢 **Done 2026-07-11.** Prebuilt binaries (item 13).
   `.github/workflows/release.yml`: triggered on `v*` tags (same tags
   TagBot.yml pushes after a Julia registry release) or manual dispatch;
   builds `libamalthea` on the same three CI hosts `run_tests.yml` already
   tests on (`ubuntu-latest`→`x86_64-unknown-linux-gnu`, `macos-latest`→
   `aarch64-apple-darwin`, `windows-2025-vs2026`→`x86_64-pc-windows-msvc`),
   deliberately with `RUSTFLAGS=""` (portable, no `target-cpu=native` —
   unlike the dev/test build path — since a downloaded binary must run on
   any user's CPU, not just the builder's); stages each asset as
   `libamalthea-<triple>.<ext>` with a per-asset `.sha256` file, then a
   `publish` job merges all `.sha256` files into one `SHA256SUMS.txt` and
   uploads everything to a GitHub Release via `softprops/action-gh-release`.
   `deps/build.jl` gained `try_download_prebuilt(rust_dir)`, tried before
   the existing `cargo build --release` fallback: resolves the release
   asset from `Project.toml`'s version + the running platform's target
   triple (only the 3 triples above; anything else — e.g. linux non-x86_64
   — returns `nothing` and falls straight to source), downloads the binary
   + `SHA256SUMS.txt`, verifies the asset's sha256 against the manifest
   entry, and only then moves it into the exact
   `amalthea/target/release/<libname>` path every `_libamalthea_path*()`
   helper across `src/*.jl` already resolves to — so no Artifacts.toml or
   separate lookup path was needed, just placing the file where the
   existing from-source build already puts it. Any failure at any step
   (network, unsupported platform, missing release, checksum mismatch) is
   caught, logged via `@info`, and falls back to `cargo build --release`
   silently — never `rethrow`s from the download path itself. Opt-out:
   `AMALTHEA_RUST_SKIP_DOWNLOAD=1` forces straight to source (useful for local
   dev iteration on `amalthea/`, where a stale downloaded binary would
   silently shadow local changes). **Verified:** the fallback path returns
   `false` cleanly, leaves any existing library file untouched (mtime
   unchanged), and leaves no `.download`/`SHA256SUMS.txt` temp files behind.
   The download+verify+install happy path was verified in isolation
   against a local HTTP server serving a real build of `libamalthea.so`
   plus its real sha256 manifest line — confirmed the checksum-match branch
   installs correctly. **Live release defect, verified 2026-07-25 with
   `gh release view v1.0.0`:** the tag exists, but its assets use legacy
   `libluna_rust-*` names while current `deps/build.jl` requests
   `libamalthea-*`; the real download therefore misses and falls back to
   Cargo. Resume-queue item 4 owns the compatibility fix and end-to-end
   clean-install validation.
2. 🟢 **Done 2026-07-19 (commit 05c4a4e).** Rust-side scan HDF5 writer
   (item 9) — `io.rs` `scan_write_point(...)` (+ `create_dataset_nd_julia`)
   writes a finished scan point's field/z arrays directly from native buffers,
   matching HDF5.jl's column-major dim-reversal so Julia reads them back
   untransposed; optional scan-queue `FlockLock`/`LockFileEx` coordination.
   Exposed via `scan_write_point_ffi` + the opt-in
   `Output.write_scan_point_native` (default Julia `HDF5Output` path
   unchanged). Also fixed a latent `io::H5T_COMPOUND` constant bug (was 3 =
   H5T_STRING). Test: `test/test_scan_native_write.jl` (:rust).
3. 🔴 **Measured, then parked — recommend against building as specified.**
   Standalone CLI (item 14). See
   `docs/dev/native-port/PLANS.md §4` for the full feasibility writeup.
   Finding: a Julia-free CLI needs the *entire* `prop_capillary` setup path
   (grid construction, mode dispersion, pulse synthesis, gas properties)
   reimplemented in Rust with nothing left to fall back on — exactly the
   one-time setup code `ARCHITECTURE.md` §6a already classified as
   "stays Julia by design, porting it buys nothing" for the per-step-loop
   goal, being asked for again for a different reason. Two pieces are
   genuine new dependencies, not mechanical ports: `PhysData.density`
   (needed for the Kerr coefficient) calls the external CoolProp real-gas
   library, and mode-averaged `Aeff` needs a Bessel-J evaluator + Bessel-
   zero root-finder (cubature.rs already supplies the quadrature
   primitive, so this one is smaller than first assessed, but still new
   special-function surface). The comparison-against-Julia acceptance
   test also would not be bit-parity — it inherits every setup-path
   numerical divergence, the same situation as Phase 7's β1
   analytic-vs-FD gap. TOML/cargo-feature gating itself is *not* a
   blocker (confirmed: `optional = true` deps + `required-features` on
   the `[[bin]]` leaves plain `cargo build --release` unaffected).
   Recommended alternative if this is ever picked up: a much smaller
   "dump-and-replay" CLI — Julia serializes the exact arrays it already
   passes to `native_set_mode_avg_params`/etc. once, and `luna-cli`
   replays them through the unmodified native stepper — which needs no
   new setup-porting work and gives a genuine bit-identical acceptance
   test, at the cost of not being Julia-free from a cold start. WASM
   (the item's stated follow-on) is blocked separately regardless: FFTW
   and HDF5 are both `dlopen`ed native libraries with no general `dlopen`
   equivalent in WASM.

**Current execution order:** use the dated resume queue at the top of this
file. The older track-order plan has been completed or superseded.

## Open items

### 🟢 Native-Rust backend port (phased)

**Goal:** make the propagation backend run **exclusively in Rust** — no Julia
callback in the per-step hot loop. **Finding:** even with all five
`AMALTHEA_USE_RUST_*` toggles ON, ~80% of per-step cost is still Julia. The Rust
stepper (`precon_step_ffi`) drives the loop but calls Julia `fbar!`/`prop!` back
through C function pointers on **every** RK stage, so every FFT (there is no Rust
FFT), Kerr, plasma `cumtrapz`, window/norm broadcasts, and the `exp(linop)`
application stay in Julia. "Exclusively Rust" therefore requires a **resident
`NativeSim`** field + native RHS + FFTW binding, not a default-flip.

Design docs (read before starting any phase):
[`docs/dev/native-port/ARCHITECTURE.md`](native-port/ARCHITECTURE.md) ·
[`docs/dev/native-port/MATH.md`](native-port/MATH.md) ·
[`docs/dev/native-port/TESTING.md`](native-port/TESTING.md) ·
[`docs/dev/native-port/PORT_LOG.md`](native-port/PORT_LOG.md) ·
agent workflow `AGENTS.md`. New toggle: `AMALTHEA_USE_RUST_NATIVE`.

Phases (each independently shippable; gate = single-step ~1e-13 **and**
full-`solve` ~1e-6 vs the Julia oracle — see TESTING.md §3 nondeterminism floor):

- ✅ **Phase 0 — Foundations.** `NativeSim` opaque handle; FFTW binding;
  `init_native_sim` / `free_native_sim` / `set_field` / `get_field`; callback-free
  `native_step` (`RustNativeStepper` in `src/RK45.jl`). Replaces callback round-trip
  in `amalthea/src/ffi.rs:1002` + `src/RK45.jl:309-319`. Gate passed: set/get
  bit-exact; no-op RHS rel_solve < 1e-6 (zero-RHS → bit-exact). 41928/41928 rust
  group pass. Test `test/test_native_phase0.jl`. ✔
- ✅ **Phase 1 — Mode-averaged + Kerr (RealGrid).** `rhs_mode_avg_real` +
  `native_set_mode_avg_params`; ports `to_time!`/`to_freq!`, Kerr, windows,
  `norm_mode_average`, exp-linop prop. Replaces `TransModeAvg`
  (`src/NonlinearRHS.jl:531`) + Kerr (`src/Nonlinear.jl:81`). First fully-Rust
  `prop_capillary(:HE11, Kerr)`. Gate passed: single-step ≤1e-13, full-solve
  5.8e-13. Test `test/test_native_phase1.jl`. ✔
- ✅ **Phase 2 — Plasma + EnvGrid Kerr.** `rhs_mode_avg_env` (EnvGrid c2c Kerr,
  3/4 SVEA factor) + `native_set_plasma_params`/plasma current assembly (rate
  LUT already Rust). Replaces `PlasmaCumtrapz` (`src/Nonlinear.jl:161`) +
  EnvGrid Kerr. Gate passed: Phase 2a (EnvGrid Kerr) single-step <1e-13,
  full-solve 3.2e-17; Phase 2b (RealGrid+plasma) single-step 3.8e-17,
  full-solve 2.7e-16 — all with fixed step size (see PORT_LOG 2026-07-01: the
  adaptive PI controller's near-cancellation error estimate is FP-noise
  sensitive and not itself a meaningful equivalence signal). Also fixed a
  latent `RustNativeStepper.s.y`-not-updated bug that broke `interpolate()`.
  Test `test/test_native_phase2.jl`. ✔
- ✅ **Phase 3 — Radial (TransRadial) + resident QDHT.** `rhs_radial` reuses the
  existing `QdhtFfiHandle` directly (no FFI round-trip per RHS) + a
  precomputed complex `(n_ω, n_r)` normalization array (folds `norm_radial` +
  `ωwin`, valid for a z-invariant `normfun`). Replaces `TransRadial`
  (`src/NonlinearRHS.jl:663`). Scope: RealGrid + scalar Kerr only (EnvGrid and
  plasma-radial deferred). Gate passed: single-step 1.1e-17, full-solve
  1.3e-16 (fixed step size, per the Phase 2 lesson — see PORT_LOG
  2026-07-01). Test `test/test_native_radial.jl`. ✔
- ✅ **Phase 4 — Raman.** `rhs_mode_avg_real` gains an additive Raman term via
  the resident `TimeDomainRamanSolver` (already-existing ADE solver, reused
  directly) + `native_set_raman_params`. Replaces `RamanPolarField`
  (`src/Nonlinear.jl:357`). Scope: RealGrid, `thg=true` only, all-SDO
  density-independent-τ2 eligibility (same criteria as `AMALTHEA_USE_RUST_RAMAN`).
  Gate passed: full-solve Rust-vs-Julia 4.2e-8 — independently verified
  non-vacuous (Raman changes the Julia oracle's full-solve result by 1.1e-4,
  self-validated in-test; a single 1cm z-step alone shows Raman's
  contribution below the FP floor relative to Kerr — the effect is
  cumulative over propagation, see PORT_LOG 2026-07-01). Test
  `test/test_native_raman.jl`. ✔
- ✅ **Phase 5 — Modal (TransModal), narrow scope.** Binds the *same*
  `libcubature` C library Julia's `Cubature.jl` wraps (`Cubature_jll`,
  dlopened at runtime like FFTW — not a reimplemented cubature algorithm, so
  adaptive node placement is bit-identical, not just close). Per-node
  evaluation reuses the existing rank-1 FFT plans + Kerr formula. Scope:
  RealGrid, constant-radius `MarcatiliMode` with `kind=:HE, n=1` only (needs
  only `besselj(0,·)`/`besselj(1,·)`, already in `diffraction.rs`),
  `full=false` (the radial modal integral — what `Interface.needfull`
  already selects for `HE,n=1` mode collections, not an artificial
  restriction), Kerr-only, `shotnoise=false`. Replaces `TransModal`
  (`src/NonlinearRHS.jl:421`, `pointcalc!` `:363`, `Erω_to_Prω!` `:401`) within
  that scope. Keeps the integration loop **sequential** (prior
  `Threads.@threads` race). Gate passed: two-mode (HE11+HE12) single-step
  1.4e-19, full-solve 4.0e-16 (fixed step size) — independently verified
  non-vacuous (HE11→HE12 energy transfer is 2.0e-5 of total energy, far above
  any noise floor). General-order modes (`TE`/`TM`/`n>1`), tapered radius,
  `full=true`, EnvGrid, and Raman/plasma-in-modal are deferred (see MATH.md
  §3.3). Test `test/test_native_modal.jl`. ✔
- ✅ **Phase 6 — Free-space (TransFree).** A genuine 3-D FFTW plan
  (`fftw.rs::RealFft3d`, new `fftw_plan_dft_r2c_3d`/`_c2r_3d` symbols — same
  libfftw3 binary Julia's `FFTW.jl` uses, not a new library) replaces the
  QDHT-plus-1-D pattern Phase 3 used for radial. Dimension order
  (`(n_x,n_y,n_t)` reversed for Julia's column-major `(n_t,n_y,n_x)`) and the
  `1/(n_t·n_y·n_x)` round-trip normalization were verified against a literal
  `FFTW.rfft` reference before being trusted, not assumed from the
  row/column-major rule alone. Scope: RealGrid, `const_norm_free`
  (z-invariant), scalar Kerr, `shotnoise=false`. Replaces `TransFree`
  (`src/NonlinearRHS.jl:826`) within that scope. Gate passed: single-step
  7.05e-18, full-solve 5.01e-17 (fixed step size). EnvGrid (c2c 3-D) and a
  z-dependent `normfun` are deferred (see MATH.md §3.4). Test
  `test/test_native_free.jl`. ✔
- ✅ **Phase 7 — z-dependent linop assembly (narrow scope).** Ports the
  mode-averaged, graded-core constant-radius `MarcatiliMode` case
  (`Capillary.gradient(gas,L,p0,p1)`, a two-point pressure-gradient
  capillary) resident — `NativeSim::ensure_linop_at(z)`. `dens(pressure)` is
  a **transferred** `HermiteSpline` (Julia's own `Maths.CSpline`
  `(x,y,D)`, not re-fit — re-fitting a different spline through sampled
  values is a spline-of-a-spline problem that doesn't converge, see
  PORT_LOG). `β1(z)` is an **exact analytic closed form**, not a LUT: since
  `εco(ω;z)-1` is separable and `nwg(ω)` is z-independent, β1(z) reduces to
  4 z-independent constants computed once via `Maths.derivative` fed a
  `BigFloat` argument — see `docs/dev/native-port/BETA1_ANALYTIC.md` for the
  derivation, why this is *more* accurate than Julia's own adaptive-FD
  `Modes.dispersion`, and the resulting tolerance tradeoff (this is the
  first phase where Rust deliberately diverges from the Julia oracle to be
  more correct, rather than a faithful bit-parity port). Also fixed: the
  nonlinear RHS's `kerr_fac`/`beta[i]` must be rescaled by `dens(z)` every
  RK stage too (`TransModeAvg` re-evaluates `densityfun(z)` fresh each
  stage) — missing this caused a ~9% full-solve mismatch, found by isolating
  that a `kerr=false` control run matched Julia while `kerr=true` didn't.
  Scope: RealGrid, Kerr-only, two-point gradient only (multi-point gradient
  and radial/free-space/modal z-dependent `nfun` deferred — see MATH.md
  §3.5). Test `test/test_native_zdep_linop.jl`. ✔
- ✅ **Phase 8 — Default-flip + cleanup.** `AMALTHEA_USE_RUST_NATIVE` default flipped
  to `"1"`; every Phases 1-7 scope restriction converted from a hard `error()`
  to a new `NativeIneligible` exception, caught by `solve_precon` and silently
  (one-time `@warn`) falls back to the Julia stepper — native being opt-in
  used to make a scope-restriction crash the right behavior; being default
  makes it a user-facing regression instead, so it must fall back. Running
  the *entire* test suite (not just the phase-specific groups) surfaced four
  real, pre-existing gaps invisible while native was opt-in: an unrecognized
  `f!` silently ran with zero nonlinearity (`test_rk45.jl`'s raw closures);
  gas mixtures crashed with a `MethodError` instead of falling back
  (non-scalar `densityfun`); `RamanPolarEnv` (GNLSE/envelope Raman) silently
  vanished (no `isa` branch matched it — closed generally with a catch-all,
  not a special case); and, most significantly, the resident field never saw
  `Luna.run`'s per-step windowing at all (`native_step` overwrites `s.yn`
  from Rust's own state, discarding whatever Julia wrote into the pointer
  between calls) — invisible because no native-specific test drives the
  stepper through `Luna.run`, fixed via a new `native_resync_field` FFI
  called after `stepfun`. A related, separate bug (dense output between
  accepted steps was linear, not Julia's quartic `interpC`) explained nearly
  every remaining general-suite failure at once — fixed via a new
  `get_ks_stage`-based `interpolate(s::RustNativeStepper)` and
  `native_apply_prop` FFI. Full details, including the tolerance-vs-bug
  triage for each affected general-purpose test, in `PORT_LOG.md`. Test
  `test/test_native_phase8.jl`. Gate met: `LUNA_TEST_GROUP=All` — 46590
  passed, 0 failed, 0 errored, 12 broken (pre-existing), with the baseline
  (`AMALTHEA_USE_RUST_NATIVE=0`) independently confirmed 100% green first. ✔

**Native-Rust backend port (Phases 0-8) complete.**

### 🟢 Per-kernel Rust FFI wiring — complete
All kernels in the original per-kernel wiring list are now wired. PPT
ionization is opt-in via `AMALTHEA_USE_RUST_IONISATION=1`.
The pattern: Rust exports an opaque handle lifecycle + a vector-eval FFI function;
Julia stores the handle in the struct and routes the in-place vector call through
`ccall`; a `@testitem tags=[:rust]` equivalence test guards the boundary.

Completed kernels:
1. ✅ **PPT ionization** (`IonRatePPTAccel`) — `AMALTHEA_USE_RUST_IONISATION` toggle —
   `test/test_ionisation_rust.jl`
2. ✅ **Time-domain Raman** (`raman.rs` `TimeDomainRamanSolver`) — toggle
   `AMALTHEA_USE_RUST_RAMAN`, `init_raman_solver` / `free_raman_solver` / `raman_solve`
   exported, wired into `Nonlinear.jl` hot loop for carrier-field SDO responses
   (`CombinedRamanResponse` with all-SDO `Rs`, density-independent τ2) —
   `test/test_raman_rust.jl`. Follow-ups: rotational multi-oscillator (FFI already
   supports n_osc>1; needs Julia-side extraction of per-J Ω/K arrays);
   density-dependent τ2 (add `raman_update_coeffs` FFI entry); intermediate-broadening
   (Gaussian damping — ~~stays Julia indefinitely~~ now native via the resident
   FFT-conv kernel, Phase I item 2, 2026-07-08); envelope (`RamanPolarEnv`) Rust path
   (~~needs real-buffer copy~~ now native, Phase F item 2). Note these follow-ups
   landed in the *native-port* path, not this older per-kernel
   `AMALTHEA_USE_RUST_RAMAN` wiring, which keeps its original narrower scope.
3. ✅ **Waveguide dispersion — Zeisberger** (`dispersion.rs` `ZeisbergerNeff`) — toggle
   `AMALTHEA_USE_RUST_DISPERSION`, `init_zeisberger_neff` / `free_zeisberger_neff` /
   `zeisberger_neff_vector` exported, wired into `Antiresonant.jl` via a specialised
   `neff_β_grid(grid, ::ZeisbergerMode, λ0)` that batch-evaluates neff over the
   positive-frequency grid per propagation step — `test/test_dispersion_rust.jl`.
   Full Zeisberger eq.(15) parity: all four mode kinds (HE/EH/TE/TM), ϕ wall-thickness
   phase, σ⁴ real (C) and imaginary (D·loss_scale) terms. Equivalence at ~1e-12
   (same formula + Julia-supplied nco/ncl → only float-reassociation differences).
   Follow-ups: Rust-side multi-term Sellmeier (offload nco/ncl computation too);
   const-linop one-time setup path (negligible cost, left on Julia indefinitely).

3a. ✅ **Waveguide dispersion — MarcatiliMode** (`dispersion.rs` `MarcatiliNeff`) — same
    `AMALTHEA_USE_RUST_DISPERSION` toggle; `init_marcatili_neff` / `free_marcatili_neff` /
    `marcatili_neff_vector` exported. Wired into the constant-radius specialisation
    `neff_β_grid(grid, ::MarcatiliMode{<:Number}, λ0)` in `Capillary.jl`. Nwg(ω)
    precomputed ONCE at setup (cladding-dependent, z-independent) and stored in the
    Rust handle; per step only nco(ω; z) is passed. Also adds z-level memoization
    even on the Julia-only fallback path (batching all sidcs before returning cached
    values). Equivalence is bitwise (0.0 rel error) — same IEEE 754 formula + same
    Float64 inputs. Model `:full` (`sqrt(εco-nwg)`) and `:reduced` (`1+(εco-1)/2-nwg`)
    both wired. Tests: `test/test_dispersion_rust.jl` (second `@testitem`).
4. ✅ **QDHT batch transform** — toggle `AMALTHEA_USE_RUST_QDHT`, `init_qdht_ffi` /
   `free_qdht_ffi` / `qdht_ffi_mul_real` / `qdht_ffi_ldiv_real` / `qdht_ffi_mul_cplx` /
   `qdht_ffi_ldiv_cplx` exported. Wired into `TransRadial` in `NonlinearRHS.jl` via
   type-stable `_qdht_mul!` / `_qdht_ldiv!` dispatch. Stores Julia's T matrix
   (transposed to row-major at init); per-call transform uses Rayon parallel
   row-vector dot products with pre-allocated scratch (4×n_r×n_time), avoiding
   the two `permutedims` allocations that Julia's dim=2 QDHT path incurs.
   Handles both `Float64` (RealGrid) and `ComplexF64` interleaved (EnvGrid).
   Equivalence: ~1e-13 relative error vs Julia BLAS path (summation order differs).
   Tests: `test/test_qdht_rust.jl` (`@testitem tags=[:rust]`).
   Follow-ups: wire `TransFree` (2D Cartesian FFT, different transform type — stays Julia);
   consider cblas/openblas DGEMM binding for peak throughput.
5. ✅ **RK45 interaction-picture PreconStepper** — Dormand-Prince 5(4) with FSAL and Lund PI
   step control implemented in `ffi.rs` (`init/free/precon_step_ffi`); Julia side in
   `src/RK45.jl` (`RustPreconStepper`, `AMALTHEA_USE_RUST_STEPPER=1`).  Key fix: `@cfunction`
   pointers must be created in `RK45.__init__` (not as precompile-image `const`s).
   Tests: `test/test_stepper_rust.jl` (physical equivalence < rtol=1e-6).

- **Safety net:** `test/test_rust_ffi.jl`, `test/test_ionisation_rust.jl`,
  `test/test_raman_rust.jl`, `test/test_dispersion_rust.jl`, and
  `amalthea/tests/*.jl` (`@testitem tags=[:rust]`, auto-discovered).

### 🟢 Windows scan-lock validation — done (2026-07-08, found already validated by existing CI)
`amalthea/src/scans.rs` `FlockLock::lock`/`unlock` call real Win32 `LockFileEx`/
`UnlockFileEx` on non-Unix targets (commit `febdde1`, 2026-06-28). This entry
previously claimed "no Windows CI runner exists" — **that was stale, and had
been since the entry was written**: `.github/workflows/run_tests.yml`'s `rust`
group has included `windows-2025-vs2026` in its OS matrix since long before this
BACKLOG entry existed (`cargo test` runs there on every push/PR). Verified
directly against a real run
(`gh api repos/vdiego28/Amalthea.jl/actions/jobs/85999378123/logs`, run
28980961881, 2026-07-08):
```
test scans::tests::test_flock_lock_new_error ... ok
test scans::tests::test_flock_lock_new ... ok
test tests::test_scan_queue_flock ... ok
test result: ok. 37 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
`test_scan_queue_flock` calls `checkout_next_index`/`mark_completed`, which call
`FlockLock::lock`/`unlock` — the real `LockFileEx`/`UnlockFileEx` path on this
runner, not a stub — and it passed. (The test does self-skip if HDF5 can't be
`dlopen`'d, printing to stdout; that branch wasn't taken here, since
`test_hdf5_io_basic` — same `get_hdf5_api()` call — also passed earlier in the
same job, before Julia's own HDF5 install step even ran, meaning the Windows
runner image already has a loadable HDF5.) **No code change was needed** — this
was a stale-documentation item, not an untested-code item.
- Currently latent in production regardless of platform — `ScanQueue`/
  `init_scan_queue` is only reachable via FFI, which `src/*.jl` doesn't call yet
  (confirmed: no `init_scan_queue` reference in any Julia source file). Relevant
  again once the Rust-side scan HDF5 writer (S6.2) is wired up.

### 🟡 GPU CI coverage
`amalthea/tests/test_gpu_cuda.jl` and the GPU numerical-equivalence tests in
`amalthea/src/lib.rs` self-skip when no GPU/CUDA is present, so the GPU code paths
(CUDA/Vulkan dispatch, GPU QDHT/Raman/ionization) are never exercised in CI.
- **Fix:** add a GPU-equipped CI runner (or a scheduled job) that actually executes them.

### ⚪ Historical 2026-07-05 GPU-resident review — superseded

> This subsection is retained as provenance for the scaffolding review. Its
> "verified" and "not wired" statements describe intermediate states and are
> not current. The authoritative status is S3 above: the narrow slice is
> wired, repaired, and hardware-verified, but has no standing GPU CI.

A prior agent pass (2026-07-05, external review — see ARCHIVE.md's "Done
(recent)" entry for the GPU ionisation clamp, and `docs/dev/native-port/GPU.md`)
left a large uncommitted working-tree diff implementing three of `SUGGESTIONS.md`'s
performance ideas. Reviewed and tested (full `rust` gate: 42004/42004, no regression) —
findings below.
- ✅ **GPU PPT ionization clamp parity** (`kernels.cu`/`ionization.rs`): the CPU-side
  `strict` clamp-vs-error behavior (`Ionisation.jl`/`ionization.rs::rate`) was missing from
  the CUDA kernel (`ppt_ionization_kernel` unconditionally errored above `e_max`). Now takes
  `strict` as an argument and clamps `abs_e = e_max` when `strict == 0`, matching the CPU
  path exactly. Verified correct by inspection against `ionization.rs`'s own `strict` field.
- 🟡 **BLAS-3 QDHT** (`ffi.rs`, new `blas.rs`): `QdhtFfiHandle::apply_real`/`apply_cplx` now
  call `cblas_dgemm` via a runtime-`dlopen`ed BLAS (`init_blas_path`, new FFI export) when
  available, falling back to the existing Rayon path otherwise. Safe (inert until called) —
  but **no Julia-side caller exists** (`grep` for `init_blas_path` in `src/*.jl` finds
  nothing), so this is currently dead code, not a wired optimization.
- 🟠 **GPU-resident stepper V1** (`native.rs`: `NativeSim` → `NativeBackend` trait +
  `Box<dyn NativeBackend>`, `CpuNativeSim` = the renamed original; new `cuda_native.rs`:
  `CudaNativeSim`, scoped to mode-averaged RealGrid Kerr(+plasma) only, all other
  `set_*_params` return `-1` matching `docs/dev/native-port/GPU.md`'s stated V1 scope):
  - **Bug found and fixed (2026-07-05):** `CudaNativeSim::step` ran the 6 internal RK
    stages via `rk45_accumulate_stage_kernel` (`DP_B`, the intra-stage a-coefficients) but
    never performed the final 5th-order solution accumulation the CPU reference does in
    `native.rs` (`let b0 = dt*DP_B5[0]; ...` block, ~line 2521) — it just re-propagated the
    untouched old field, silently dropping the entire nonlinear RK contribution on every
    accepted step. Fixed by adding one more `rk45_accumulate_stage_fn` launch, in place on
    `field_d`, using `DP_B5` weights, gated on `locextrap != 0` exactly like the CPU
    reference, right before the existing final `apply_prop` call. Compiles clean, all 37
    Rust unit tests and the full `rust` Julia gate (42004/42004) still pass — **but this
    fix has still never executed on real CUDA hardware** (only checked for logical parity
    against `CpuNativeSim::step`, not numerically verified end-to-end).
  - **Opt-in gate added:** `init_cuda_native_sim` (the only FFI entry point that constructs
    a `CudaNativeSim`) now refuses to initialize — returns null and prints a warning to
    stderr — unless `AMALTHEA_USE_RUST_CUDA_NATIVE=1` is set, and prints a second warning on
    successful opt-in. Deliberately stricter than the usual `AMALTHEA_USE_RUST_*` toggles
    (which default-on once verified): this one requires explicit opt-in until verified on
    real GPU hardware. Covered by `test_cuda_native_sim_ffi_gated_by_env_var` in `lib.rs`.
  - **Not wired to Julia at all**: no `src/*.jl` file references `init_cuda_native_sim` or
    the CUDA path; `RK45.jl`'s native dispatch is untouched. Purely additive scaffolding —
    zero risk to the existing (CPU) native-port default, doubly so now with the opt-in gate.
  - **Untestable on this machine**: no `nvcc` in `PATH` or at `/usr/local/cuda/bin/nvcc`
    (only the NVIDIA driver is present), so `build.rs` falls back to a dummy PTX and
    `CudaNativeSim::new` fails to load real kernels — `lib.rs`'s
    `test_cuda_native_sim_basic` self-skips (confirmed via `--nocapture`), so the GPU path
    has never actually executed on real hardware here.
  - Design-doc deviation: `docs/dev/native-port/GPU.md` §4 specifies an `enum { Cpu, Gpu }`
    dispatch ("no `Box<dyn>`... avoids dynamic dispatch overhead") but the implementation
    uses `Box<dyn NativeBackend>` instead — functionally fine, just not what was designed.
  - **Update 2026-07-07: verified on real CUDA hardware (RTX 5060 Ti, CUDA 13.3) and wired
    into `RK45.jl`** — see ARCHIVE.md's "Done (recent)" for the full list of bugs found and fixed
    along the way (this section's text above is left as historical record of the pre-hardware
    state).

### 🟡 Distribution & example-code maintenance — smoke coverage landed; repairs remain

Salvaged 2026-07-22 from a retrospective architecture review
(`ADR-001`, drafted 2026-07-20, not kept — see note at the end of this
subsection). Only the two findings that survived verification against the
tree are recorded here.

1. 🟢 **Install failure is documented and release machinery exists.**
   `README.md` and `deps/build.jl` explain the Cargo fallback and give an
   actionable error. `.github/workflows/release.yml` builds three portable
   assets and `deps/build.jl` verifies checksums before installing them.
   🟢 The `v1.0.0` release's legacy `libluna_rust-*` asset names are now
   handled by a bounded fallback (resume-queue item 4), and source checkouts
   (clones, `Pkg.develop`, CI) always compile from source rather than install
   a binary older than their own FFI surface.
2. 🟡 **Representative smoke CI landed; seven broken examples remain.**
   `test/test_examples_smoke.jl` runs eight representative files at a shrunk
   5 mm propagation length in the `examples` CI group (16/16 assertions,
   ~45 s package time). It deliberately stops before plotting. The audit
   found 44 Julia files across 11 low-level-example subdirectories and seven
   distinct broken files outside the smoke subset:

   - `full_modal/basic_modal_full_bothpolarisations.jl`
   - `full_modal/basic_modal_full.jl`
   - `polarisation/modal_vector_plasma.jl`
   - `polarisation/modal_nonvector_plasma.jl`
   - `polarisation/modal_vector_plasma_CP.jl`
   - `polarisation/modal_vector_plasma_45deg.jl`
   - `polarisation/elliptical_env.jl`

   The first six reference `linop` before assignment; the two `full_modal`
   files plus `elliptical_env.jl` pass `grid.ω` to `norm_modal` instead of
   the required grid object. Fix them, then add at least one regression case
   for each failure class to the smoke subset. Full measurements and the
   plotting/AST-harness rationale are preserved in
   `native-port/portlog-inbox/hygiene.md`.

*Three further items from the same review were dropped after checking them
against the tree, recorded here so they aren't re-raised: (i) "confirm CI
exercises the AVX2/AVX-512/CUDA/Vulkan dispatch paths" rests on a false
premise — `dispatch.rs` is detection-only and unwired (`HardwarePath`/
`SimulationEngine` appear nowhere outside that module and its own unit
tests; Vulkan has no implementation at all), real vectorization comes from
`target-cpu=native` + LLVM auto-vectorization, and the real GPU path is the
opt-in `AMALTHEA_USE_RUST_CUDA_NATIVE=1` `CudaNativeSim`; (ii) "write a
contributor guide splitting Julia-layer vs Rust-crate work" is already
served by `CLAUDE.md`, `AGENTS.md`, and `docs/dev/native-port/`; (iii)
"establish a process for tracking upstream Luna.jl changes" already exists
as `.github/workflows/upstream_sync.yml`. The ADR itself was not committed:
its central premise — automatic runtime hardware dispatch as a shipped
design decision — describes an architecture this repo does not have, and
that error propagated through its complexity assessment, risk analysis, and
consequences, so correcting it would have meant rewriting rather than
amending it.*

## Informational / no action planned

- ⚪ **`RUSTFLAGS` reach.** `RUSTFLAGS=-D warnings` now applies to every CI job's package
  build, because both workflows force the from-source path
  (`AMALTHEA_RUST_SKIP_DOWNLOAD=1`, see resume item 4) — so a new crate
  warning fails all jobs, not just the `rust` group's explicit steps, which
  neutralize `RUSTFLAGS` themselves.

- ⚪ `deps/build.jl` forwards `ENV["RUSTFLAGS"]` (defaulting to `""` if unset),
  which neutralizes `.cargo/config.toml`'s `target-cpu=native` for
  package-driven builds. This is the portability safeguard: the resulting
  binary uses the compiler's portable baseline plus ordinary LLVM
  auto-vectorization. `dispatch.rs` does **not** select a propagation ISA at
  runtime; it is detection-only and unwired. `target-cpu=native` applies only
  to a manual Cargo build that leaves the repository config active. The one
  explicit runtime SIMD kernel is Raman's AVX2 lane. **Note:**
  `actions-rust-lang/setup-rust-toolchain` sets
  `RUSTFLAGS=-D warnings` in CI, which propagates through `deps/build.jl` — so
  the package build runs under strict warnings (desired; keeps the Rust code clean).
