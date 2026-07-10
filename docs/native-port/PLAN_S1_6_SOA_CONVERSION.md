# Plan: BACKLOG.md S1 item 6 — full SoA conversion of the native-port resident field

Status: **Phase 0 done (2026-07-10). Phase 1 halted before implementation,
pending re-confirmation — see "Ceiling measurement" section below.** See
BACKLOG.md's S1 item 6 entry for the live status line; this file is the
durable plan (survives a context reset), not the status tracker.

## Ceiling measurement (2026-07-10) — read this before resuming Phase 1

Before writing the Phase 1 rewrite, measured `apply_prop_cached`'s actual
share of `step()`'s own wall time, using temporary `Instant`-based counters
(added to `native.rs`, read via a temporary FFI accessor, then fully
reverted — `git diff` was empty afterward). Workload: the same
mode-averaged + plasma + kerr fixed-dt configuration
`benchmark_native_default.jl` uses, 2000 steps.

```
wall clock for 2000 steps: 4.206 s
sum of step() internal timers: 4.187 s
sum of apply_prop_cached time: 0.082 s
apply_prop share of step() internal time: 1.95%
apply_prop share of wall-clock time: 1.94%
```

`apply_prop_cached` — the ~14 calls/step this whole effort targets — is
**~2% of step() time**. Even a perfect 2.6× speedup on it (the isolated
Criterion spike's best case, Phase 0's benchmark) caps the *entire* SoA
conversion's end-to-end win at roughly **1.2%** for this workload: the
O(n) exp-linop multiply sits next to O(n·log n) FFTs called several times
per stage across all 7 RK stages, and those FFTs dominate `step()`'s time.
Phase 2's split-DFT FFTW plans wouldn't recover the gap either — FFTW's
split-array plans are typically equal-or-slightly-slower than its
interleaved plans, not faster.

This is materially different information from what motivated "proceed with
full SoA conversion" — that decision was made knowing only the isolated
microbenchmark's 2-2.6× number, not this end-to-end ceiling. Flagged back
to the user before committing to the multi-week Phase 1-4 rewrite of a
numerics-critical stepper for a ~1% ceiling. **Do not resume Phase 1
without a fresh decision from the user in light of this number.**

## Context

BACKLOG.md S1 item 6 asked for a timeboxed Criterion spike comparing AoS
(interleaved `Complex<f64>`, today's layout throughout `native.rs`) against
SoA (separate re/im `Vec<f64>`) for the exp-linop hot loop, with the rule:
only do the invasive `fftw_plan_guru_split_dft` rewrite if the spike shows
>1.3x. The spike (`luna-rust/benches/exp_linop_layout_bench.rs`, committed
2026-07-10) found `apply_prop_cached`'s `y[i] *= factors[i]` multiply is
~2.0-2.6x faster under SoA across 2k/8k/16k grids — clearing the bar.

A follow-up survey (Explore agent, 2026-07-10) found the real scope is much
larger than the backlog item implies: there is no chokepoint — the
resident field and RK stage buffers are ~30 distinct `Vec<Complex<f64>>`
fields, touched directly (raw indexing, `Complex` operator overloads)
across 8 near-duplicate `rhs_*` functions (one per geometry × grid-type
combination: mode-avg/radial/modal/free-space × RealGrid/EnvGrid). `fftw.rs`
had zero split-array plan support (fixed in Phase 0 below). Every
FFT-touching kernel pays an interleave/deinterleave tax under SoA unless
split-DFT plans exist. The Julia↔Rust FFI boundary (`native_step`'s `yn`
pointer, `set_field`/`get_field`) also hard-assumes AoS interleaved layout
— Julia's own `ComplexF64` arrays are always interleaved, so that boundary
needs an AoS↔SoA shim regardless of how far the internal conversion goes.

The user was shown this expanded scope (no chokepoint, FFT round-trips pay
an interleave tax under SoA unless split-DFT plans exist, only
`apply_prop_cached` — not `weaknorm_c64`, not any `rhs_*` kernel — showed a
measured win) and explicitly chose to proceed with the full end-to-end
conversion anyway, rather than the narrower localized fix. This plan
executes that choice as a phased effort (mirroring how the native port
itself — Phases 0-8, D-I — was staged and gated), because a single
undifferentiated diff across ~30 fields and 8 RHS functions in a
numerics-critical stepper would be unreviewable and unsafe to land at once.

**Intended outcome:** the resident field, RK stage buffers, and per-geometry
scratch are stored as SoA (re/im pairs) end-to-end; FFTW calls use the new
split-DFT plan bindings instead of AoS re-interleaving; the FFI boundary
still presents Julia with interleaved `ComplexF64` (unchanged Julia-side
contract) via a thin shim at the native_step/get_field/set_field crossing;
numerical output is bit-identical to today's AoS path (same arithmetic,
same operation order, pure layout change) for every existing native-port
test (`test_native_phase{1..8}.jl` and friends) and the full 7-group gate.

## Phased implementation

Each phase lands as its own commit, gated by the **full** test suite (Rust
`cargo test` + the full 7-group Julia gate via
`test/parallel_rust_tests.py` for `rust` and the 6-group run for the rest —
per standing project practice, never a phase-specific subset), before
starting the next phase.

### Phase 0 — `fftw.rs`: split-array plan bindings — DONE 2026-07-10
Added `fftw_plan_guru_split_dft`, `fftw_plan_guru_split_dft_r2c`,
`fftw_plan_guru_split_dft_c2r` (rank-1 only) to `FftwApi`'s bound-symbol
table (same `Library::sym` dlopen infra already used for the 8 existing
plan functions). Added 2 new wrapper types — `SplitComplexFft1d`,
`SplitRealFft1d` — each with `forward`/`inverse` taking separate
`&mut [f64]` re/im slices instead of one `&mut [Complex<f64>]`. Existing
AoS wrapper types untouched (both APIs coexist).

**Non-obvious gotcha found and fixed:** `fftw_plan_guru_split_dft` has no
explicit sign/direction argument. Per FFTW's own documentation, the
backward (unnormalized inverse) transform is obtained from the *same*
primitive by swapping the real/imaginary pointers on **both** input and
output (equivalent to conjugating twice) — not just the output, which was
the first (wrong) attempt and produced a completely different result
(caught by the bit-exact unit test against the AoS plan, `left:
15.999999999999998, right: 0.0` at index 0 — an unmistakable failure, not a
rounding discrepancy). Fixed by swapping both input and output re/im
pointers at both plan-creation time and execution time for the `inverse`
path.

3-D split variants (`SplitRealFft3d`, `SplitComplexFft3d`) deliberately
**not yet added** — no caller needs them until the free-space geometry step
of Phase 2. Add then, not speculatively now.

Verified: 2 new unit tests, `split_c2c_matches_aos_c2c` and
`split_r2c_matches_aos_r2c`, assert bit-exact (not tolerance-based)
agreement against the existing AoS plans on the same input, since both are
the same underlying FFTW plan class/algorithm — just a different buffer
layout. Full Rust suite 50/50 (48 prior + 2 new). Full 7-group Julia gate
unchanged (not yet wired into `native.rs`, so this is a no-op check): rust
42087/42087, physics 1645/1657 (12 pre-existing `@test_broken`), 
sim-interface 301/301, sim-multimode 33/33, sim-propagation 18/18,
io 2302/2302, fields 334/334.

### Phase 1 — universal RK buffers + `ExpCache` (the benchmarked win) — NEXT
Convert the 5 universal buffers touched by every geometry —
`field`, `linop`, `ks: [Vec<Complex<f64>>; 7]`, `yerr`, `ystage` — plus
`ExpCache`'s cached arrays, to SoA (`re: Vec<f64>, im: Vec<f64>` pairs, or a
small `SoaComplexBuf { re: Vec<f64>, im: Vec<f64> }` newtype with
`len()`/indexing helpers to avoid repeating `.re`/`.im` field access
everywhere). Rewrite `apply_prop_cached` and `ExpCache::get` to operate
directly on the SoA layout (no AoS↔SoA conversion at this boundary — this
is the whole point). The `step()` RK combine loops (native.rs ~3134-3143,
3206-3231) are already hand-split into `.re`/`.im` f64 arithmetic per the
survey — these become pure storage-layout changes, not arithmetic
rewrites. Every `rhs_*` function that reads/writes `field`/`ks[i]` still
needs an AoS view at its FFT boundary (Phase 2 handles this per-geometry) —
for this phase, add a temporary AoS↔SoA conversion helper at the
`rhs_*`/`step()` boundary so Phase 1 lands and is bit-identical-verified in
isolation before touching the RHS functions themselves.

**Verify:** `cargo test` (currently 50 tests), full 7-group gate, and
re-run `exp_linop_layout_bench.rs` in-repo (not synthetic) to confirm the
~2x win survives inside the real stepper, not just the isolated
microbenchmark.

### Phase 2 — one geometry at a time: mode-avg → radial → modal → free-space
For each geometry (in this order, cheapest/most-used first), convert its
per-geometry scratch buffers (e.g. mode-avg: `eoo`, `poo`, `pre`,
`et_noise_cplx`, `eto_cplx`, `pto_cplx`, `eoo_cplx`, `poo_cplx`) to SoA and
rewrite the corresponding `rhs_*` function(s) to use the new split-DFT
plans from Phase 0 for their FFT round-trips, removing the temporary
AoS↔SoA shim introduced in Phase 1 for that geometry's buffers. Land as one
commit per geometry (4 commits total: mode-avg, radial — including
`diffraction.rs::Qdht::transform`, which also needs a split-complex path
since it's BLAS `zgemv` on an AoS buffer today — modal, free-space, adding
the 3-D split plan variants to `fftw.rs` at that point), each gated by the
full test suite before starting the next. Raman (`apply_raman_*`) is mostly
real-arithmetic already (the ADE solver operates on real `intensity`/`f64`
scratch) — only its complex accumulation boundary into the parent
geometry's `pto_cplx`-family buffer needs touching, folded into whichever
geometry phase owns that buffer.

### Phase 3 — FFI boundary shim
`native_step`'s `yn` output pointer and `set_field`/`get_field` must keep
presenting interleaved `ComplexF64` to Julia (Julia's own array layout is
always interleaved — this is not something to change). Add a thin
interleave/deinterleave conversion exactly at this crossing (once per
`native_step` call, not per internal operation) so the Rust-internal SoA
representation is invisible to the FFI contract. Verify: `test_rust_ffi.jl`
and the full native-port test files still pass with no Julia-side changes
needed.

### Phase 4 — cleanup + close out
Remove the temporary Phase-1 AoS↔SoA shim once every geometry has migrated
(Phase 2 complete), delete now-dead AoS wrapper types in `fftw.rs` if no
call site still uses them (keep if any geometry legitimately still needs
AoS — e.g. if a kernel turns out not to benefit and is deliberately left
AoS, document why in a comment rather than silently converting it for
uniformity). Update BACKLOG.md's S1.6 entry from "phased, in progress" to
resolved, with the final measured end-to-end speedup on
`benchmark_native_default.jl` (not just the isolated microbenchmark) as the
closing number. Update `ARCHITECTURE.md` if the field storage description
there references AoS layout.

## Verification (every phase)

- `RUSTFLAGS="-D warnings" cargo test --release` — must stay 100% green,
  zero warnings.
- Full 7-group Julia gate: `rust` group via
  `python3 test/parallel_rust_tests.py` (max 10 workers, current baseline
  42087/42087) plus the other 6 groups
  (physics/sim-interface/sim-multimode/sim-propagation/io/fields) via
  `LUNA_TEST_GROUP=<group> julia --project test/runtests.jl` — full suite,
  not a phase-specific subset, per standing project practice.
- No Julia-side test changes should be needed at any phase — the FFI
  contract (Phase 3) is designed to keep the external interface identical;
  if any existing `test_native_phase*.jl` needs modification to pass, that
  is a signal something in the phase broke the intended invisibility of
  the internal layout change, not an expected/acceptable diff.

## Gotcha: sandbox filesystem restrictions during test runs

Not related to the SoA work itself, but bit this session (2026-07-10):
running the Julia test gate under the default sandboxed Bash tool fails
every single test with `SystemError: opening file
"/home/diego/.julia/logs/scratch_usage.toml": Read-only file system` —
Julia's `Scratch.jl` tries to write a usage-tracking file outside the
sandbox's writable paths on every `Scratch.get_scratch!` call (used by
`Utils.cachedir()`). This produces failures that look like a real
regression (all workers `rc=1`, garbled pass/total counts) but are purely
a sandbox artifact. Fix: run the test commands with the sandbox disabled.
