# Implementation Plan — Fix FFTW Wisdom Persistence in the Native Path

Status: **planning only, not yet implemented.** Written 2026-07-09. This file is
self-contained; a future session can execute it with only the repo checked out
and no memory of the conversation that produced it. Do the code work in a fresh
session — this document deliberately stops at the plan.

Related tracking: `BACKLOG.md` → "🟡 S1 — Hot-loop CPU performance", item 1
("Known open risk" + "Second risk, found 2026-07-09"). Cross-reference:
`docs/native-port/VANILLA_LUNA_ISSUES.md`. Do not treat this plan or that doc as
more authoritative than the primary sources (the code + `BACKLOG.md`).

---

## 1. Context (why this matters — read first)

The native-Rust resident stepper (`RustNativeStepper` in `src/RK45.jl`, backed by
`NativeSim`/`CpuNativeSim` in `luna-rust/src/native.rs`) owns its own resident
FFTW plans for the lifetime of a `solve()`. Earlier today (commit `1d8ab82`,
re-verified `f88ea43`) we added **FFTW planner-wisdom persistence** to this path,
mirroring how vanilla Julia Luna's `Utils.loadFFTwisdom`/`saveFFTwisdom` wrap
every `prop_capillary`/`prop_gnlse` setup. The mechanism:

- `luna-rust/src/fftw.rs`: `FftwApi` binds `fftw_import_wisdom_from_filename` /
  `fftw_export_wisdom_to_filename` (dlsym, both best-effort — any failure returns
  `false`, never panics).
- `luna-rust/src/native.rs`: `set_fftw_plans` takes a trailing `wisdom_path` and
  **imports wisdom right after `FftwApi::load`, before any plan is created**; a
  new `wisdom_export` trait method + `native_export_fftw_wisdom` FFI lets Julia
  trigger an export.
- `src/RK45.jl`: the `RustNativeStepper` constructor **always** computes a wisdom
  path (`joinpath(Utils.cachedir(), "native_fftw_wisdom_$(threads)threads")` — a
  *new, dedicated* file, deliberately not shared with `Utils`' own
  `FFTWcache_*threads`/`mkpidlock` file) and — critically — **exports wisdom once
  at the end of every single construction**, not once per process.

Two problems were flagged (both in the `BACKLOG.md` S1 item-1 entry):

**Problem 1 — performance (flagged, never benchmarked).** Exporting on *every*
construction means every `RustNativeStepper()` pays disk I/O + FFTW's internal
planner-lock, even for workloads that build thousands of short-lived steppers
(parameter scans). This could be a net regression versus not persisting at all.

**Problem 2 — reproducibility (found + empirically confirmed today).** Before this
change, native plans were always fresh `FFTW_ESTIMATE` plans with no wisdom
involved — deterministic by grid size alone. Now that import runs *before*
planning, plan selection can vary with accumulated wisdom (both the on-disk file
shared across all machine runs, AND FFTW's in-process wisdom pool that grows as
plans are created). Via the RK45 adaptive controller's documented near-cancellation
sensitivity (`b5-b4≈0`; CLAUDE.md "Phase 2" gotcha, TESTING.md §3), a different
plan's ~1e-15 summation-order difference can nudge the controller onto a different
`dt` sequence and change the step count. Empirically: running the `rust` Julia test
group's 43 files as **one** process gives 42098/42098 assertions; **splitting** into
many processes (concurrent *or* fully sequential — a write race was ruled out)
consistently gives 42081/42081. A stable 17-assertion swing from process topology
alone, no failures either way. Final accuracy still respects `rtol`/`atol`; only the
step-path / bit-level reproducibility is weakened.

**The single most important technical fact for choosing a fix** (verify the pointers
below yourself before acting; they were true at write time):

> **Export throttling fixes Problem 1 only. Problem 2 is driven by *import* + the
> in-process wisdom pool — never by export.** Any "fix reproducibility by exporting
> once per process" idea is wrong. Keep this distinction load-bearing.

There are therefore **two independent channels** feeding Problem 2:

1. **Disk channel** — the persisted on-disk file. This is what makes *consecutive
   runs on one machine diverge*: run 2 imports what run 1 wrote. Fully eliminable.
2. **Pool channel** — FFTW's process-global in-process wisdom pool. Native
   `dlopen`s the **same** `FFTW_jll.libfftw3` that Julia's `FFTW.jl` uses
   (`src/RK45.jl:963` passes `FFTW.FFTW_jll.libfftw3`; `luna-rust/src/fftw.rs:64`
   opens it `RTLD_GLOBAL`), so there is exactly **one** shared global wisdom pool
   for the whole process. Planning always consults it regardless of the
   `FFTW_ESTIMATE` flag. **No import-side change can fully eliminate this channel**
   without `fftw_forget_wisdom()`, which is process-global and would corrupt
   Julia's own FFTW.jl plans — not viable. The real determinism boundary is
   therefore *"one fresh process per simulation."*

**Pre-existing-in-Luna caveat (verify, don't just repeat).** Vanilla Luna's default
FFTW mode is `FFTW.PATIENT` (`src/Luna.jl:13`), and `loadFFTwisdom`/`saveFFTwisdom`
wrap every setup entry point (over a dozen call sites in `src/Luna.jl`; the funcs
are `src/Utils.jl:84` and `:99`). `saveFFTwisdom` even `rm`s and rewrites the whole
file on *every* call, guarded by `mkpidlock`. So (a) "export on every setup" is not
novel — Julia already does it; and (b) PATIENT does empirical timing search, which
is *more* nondeterministic than what we added. **But note the asymmetry that makes
our case weaker, not stronger:** the native path plans with `FFTW_ESTIMATE`
(`src/RK45.jl:985` passes `64` = `1<<6`), whose whole point is to skip measurement
and pick a plan from cheap heuristics. ESTIMATE neither runs timings nor produces
meaningful new wisdom; imported wisdom only helps if a *MEASURE/PATIENT* plan for
that exact size+flags already exists in the file. Since the native path *only ever*
plans with ESTIMATE, it likely never writes wisdom worth importing — meaning **the
speedup Problem-1's feature is supposed to buy may be near-zero here.** This is the
crux of the recommendation and must be **measured, not asserted** (see §6).

---

## 2. Confirmed current state (exact locations, verified at write time)

`luna-rust/src/fftw.rs`
- L152–156: `ImportWisdom` / `ExportWisdom` fn-ptr typedefs (return `c_int`,
  nonzero = success — FFTW's convention, opposite this file's usual 0=success).
- L171–172, 218–221, 235–236: `Option<…>` fields on `FftwApi`, dlsym'd in `load`
  (miss → `None` → later no-op).
- L247–255: `import_wisdom_from_filename(&self, path) -> bool` — takes
  `PLANNER_LOCK`, calls the fn, `!= 0`.
- L260–268: `export_wisdom_to_filename(&self, path) -> bool` — same shape.
- Note L16–22: the planner-flag doc — bit-parity requires matching the flag used
  by the run under test; tests use `:estimate`, package default `:patient`.

`luna-rust/src/native.rs`
- L2419: `set_fftw_plans` trait sig — trailing `wisdom_path: *const c_char`.
- L2436–2445: `wisdom_export` trait method + doc ("Call once, after all
  `set_*_params` … wisdom is process-global").
- L2553–2589: `CpuNativeSim::set_fftw_plans` impl. **L2569–2573**: the import —
  `if !wisdom_path.is_null() { … import_wisdom_from_filename(wisdom_str) }`,
  before the `RealFft1d::new` / `ComplexFft1d::new` plan creation at L2579–2585.
- L2590–2599: `CpuNativeSim::wisdom_export` impl (returns 1 if `fftw_api` is
  `None` or path null/bad, else 0/1 from the export).
- L3496–3509: `native_set_fftw_plans` FFI export (threads `wisdom_path` through).
- L3511–3524: `native_export_fftw_wisdom` FFI export.
- (Not shown but stated in BACKLOG: `CudaNativeSim::wisdom_export` is a stub
  returning 1/"unavailable" — confirm before editing if touching the GPU path.)

`src/RK45.jl` (all inside the `RustNativeStepper` constructor)
- L963: `lib_path = FFTW.FFTW_jll.libfftw3`.
- L965–981: computes `native_wisdom_path` (`try joinpath(Utils.cachedir(),
  "native_fftw_wisdom_$(Utils.FFTWthreads())threads") catch "" end`).
- L983–987: `native_set_fftw_plans` ccall — passes `native_wisdom_path` and the
  literal `64` (FFTW_ESTIMATE) as `flags`.
- L1830–1837: **the per-construction export** — `if !isempty(native_wisdom_path)
  … mkpath … ccall(:native_export_fftw_wisdom …)`. This is the line that runs on
  every construction.
- Constructor spans ~L900–L1844; `step!` at L1846; `interpolate` at L1882.

Vanilla comparison
- `src/Luna.jl:13` — `settings["fftw_flag"] => FFTW.PATIENT`.
- `src/Utils.jl:84` `loadFFTwisdom` / `:99` `saveFFTwisdom` (`mkpidlock`,
  `FFTWcache_$(FFTWthreads())threads`).

Benchmark harness (already exists — Phase G.3)
- `test/benchmark_native_default.jl`. Run standalone:
  `julia --project test/benchmark_native_default.jl`. Sets `:estimate` mode.
  Check 1 (`@assert`): default `prop_capillary` selects `RustNativeStepper`
  (`RK45._LAST_STEPPER_TYPE[]`). Check 2: native's own per-step wall time on a
  **fixed-dt** run (adaptive-path confound deliberately removed), written to
  `test/benchmark_result.json` (`customSmallerIsBetter`). **This harness does not
  currently construct many steppers in a loop — it builds one. It must be extended
  (see §6) to exercise the scan/perf question; it cannot answer Problem 1 as-is.**

---

## 3. Candidate fixes (with tradeoffs)

Throughout: "import" = reading the on-disk file into the shared pool before
planning (the reproducibility-relevant action); "export" = writing the pool to
disk after planning (the perf-relevant action).

### C1. Export once per process (guard flag), keep import every construction
- *How:* a Julia-side `const _NATIVE_WISDOM_EXPORTED = Ref(false)` (or a Rust
  `AtomicBool`); export block runs only on the first construction.
- *Pros:* trivial; kills the per-construction disk-write + planner-lock that is
  Problem 1's dominant cost. Minimal surface.
- *Cons:* **does not touch Problem 2** — import still runs every construction, the
  disk channel still diverges run-to-run, the pool channel is untouched. Also
  loses wisdom that *later, larger-grid* constructions in a long session would
  have contributed (first construction "wins" the export). For ESTIMATE-only
  planning that lost wisdom is probably worthless anyway (see §1 crux), which is
  itself an argument that the whole feature is low-value.

### C2. Construction-count / elapsed-time threshold (export every Nth, or ≤1×/M s)
- *Pros:* bounds export cost while still refreshing wisdom occasionally.
- *Cons:* introduces an arbitrary tuning constant with no principled value;
  still doesn't address Problem 2; more code/state than C1 for no reproducibility
  gain. Rejected — complexity without solving the harder problem.

### C3. Env-var opt-in, **default OFF** (mirror the `LUNA_USE_RUST_CUDA_NATIVE`
gating pattern)
- *How:* Julia reads e.g. `LUNA_NATIVE_FFTW_WISDOM` once. Default/unset ⇒ pass an
  **empty** `wisdom_path` to `native_set_fftw_plans` (Rust already treats `""` as
  "no import" — `CStr::from_ptr("").to_str()` is `Ok("")`, opening `""` fails the
  best-effort import) **and** skip the export ccall entirely.
- *Pros:* solves Problem 1 **completely** (zero import, zero export, no I/O, no
  planner-lock, no guard machinery). Solves the **disk channel** of Problem 2
  completely — restores the pre-`1d8ab82` clean per-grid-size determinism as the
  default; consecutive identical runs on one machine stop diverging. Simplest
  possible diff (one env read, gate two existing blocks). Matches the fork's
  established "native accel is on, extra risk features are opt-in" convention.
- *Cons:* users get zero wisdom benefit unless they opt in — **but only a con if
  that benefit is real, which under ESTIMATE-only planning is doubtful and must be
  measured (§6).** Does not touch the **pool channel** (single-process scans keep
  a shared pool) — but nothing short of `fftw_forget_wisdom` can, and that's not
  viable (see §1). Opt-in still available for anyone who wants persistence.

### C4. Default ON but export-once-per-process; add a kill-switch env var
- Essentially C1 + an escape hatch. *Pros:* preserves any speedup by default while
  bounding export cost; kill-switch gives reproducibility-sensitive scans an out.
  *Cons:* more machinery (guard flag *and* env read); **default still imports
  every construction, so the default still has Problem 2's disk channel** — the
  worst reproducibility flavor stays on by default. Strictly more complex than C3
  and worse on reproducibility-by-default.

### Where the fix should live
Prefer **Julia (`src/RK45.jl`)**: the env read + gating both blocks there means
**no Rust rebuild** is needed and the empty-path / skip-export contract already
exists on the Rust side (import treats `""` as no-op; export is simply not called).
The Rust import/export plumbing (`fftw.rs`, `native.rs`) stays as-is — it is
correct and reusable; we only change *when Julia asks for it*. Touch Rust only if
you later choose a Rust-side `AtomicBool` guard (not recommended — keep state in
Julia where the policy lives).

---

## 4. Recommended approach

**Adopt C3: make on-disk wisdom persistence opt-in, default OFF, and make the
default decision benchmark-gated.**

Justification against the tradeoffs:
- It solves Problem 1 outright (no export cost at all by default) — strictly
  better than throttling (C1/C4) and with less machinery.
- It is the safer reproducibility default: it removes the **disk channel**, which
  is the specific mechanism that makes *consecutive identical runs on one machine
  diverge*. That is the worst nonreproducibility flavor and the one we can kill
  cleanly. The **pool channel** remains (single-process multi-sim), but that is a
  fundamental consequence of sharing one global FFTW pool and is honestly bounded
  by "one process per sim," which is how scans already run (`scans.rs`' `ScanQueue`
  parallelizes across *processes*).
- The task's "opt-in defeats the point of Problem-1's feature" objection only
  bites if the ESTIMATE-path wisdom speedup is real. **Do not assert it either
  way — measure it** (§6). If the benchmark shows negligible speedup, default-OFF
  wins with the objection moot. If it shows a real gain, flip to C4 as the
  documented fallback (see §7).

### Concrete steps

1. **`src/RK45.jl` — read the toggle once.** Near the top of the module (beside
   the other `LUNA_USE_RUST_*` reads), add a small helper, e.g.
   `_native_wisdom_enabled() = get(ENV, "LUNA_NATIVE_FFTW_WISDOM", "0") == "1"`
   (choose the exact name to match repo convention — grep existing `LUNA_` env
   reads first; `LUNA_USE_RUST_CUDA_NATIVE` uses `"1"` truthiness). Default OFF.

2. **`src/RK45.jl` — gate the import path (L965–986).** Only compute a non-empty
   `native_wisdom_path` when the toggle is on; otherwise pass `""` to the
   `native_set_fftw_plans` ccall (keeps the `Cstring` arg type-stable; Rust
   no-ops on `""`). The `flags=64` argument is unchanged. Keep the whole thing in
   the existing `try/catch → ""` guard so a `cachedir()` failure still can't throw.

3. **`src/RK45.jl` — gate the export block (L1830–1837).** Wrap in the same
   toggle. When off, the block is skipped entirely — no `mkpath`, no ccall. (The
   existing `!isempty(native_wisdom_path)` check already makes this fall out
   naturally if step 2 leaves the path empty when off, but gate explicitly for
   clarity.)

4. **No Rust changes required** for the recommended default. Leave
   `fftw.rs`/`native.rs` wisdom plumbing intact — it is exercised only when a user
   opts in. Do **not** delete it: it's the mechanism the opt-in and the C4
   fallback both rely on.

5. **Docs.** Update `BACKLOG.md` S1 item 1 (mark the open risk resolved with the
   chosen default + the benchmark number that justified it). Add one line to
   `CLAUDE.md`'s env-toggle list and to `docs/native-port/ARCHITECTURE.md` §6
   (fallback/opt-in catalog) documenting `LUNA_NATIVE_FFTW_WISDOM`. Note the
   determinism boundary ("one process per sim") in `VANILLA_LUNA_ISSUES.md`.

### New tests

- **T1 (opt-out is the default, and is a true no-op).** A `@testitem tags=[:rust]`
  with the usual FFTW skip-guard: construct several `RustNativeStepper`s in one
  process with the toggle **unset**; assert the dedicated wisdom file
  (`joinpath(Utils.cachedir(), "native_fftw_wisdom_*threads")`) is **not**
  created/modified (snapshot mtime/existence before and after). This pins "default
  OFF writes nothing to disk."
- **T2 (opt-in exports at most once, and imports).** With the toggle **on**,
  construct N steppers in one process; assert the wisdom file exists after the
  first and (if C4 is ever chosen) that export ran once. For pure C3 (export every
  construction *when opted in*), instead assert the file exists and is non-empty
  and that construction 2 does not error — the once-per-process assertion only
  applies if you also add the guard. Keep T2 minimal; its job is "opt-in path
  still works," not perf.
- **T3 (reproducibility — the real verification, see §5).**

Testing note: T1/T2 must set/unset the env var *within* the test and restore it in
a `finally`, and must run in a process where `Utils.cachedir()` is writable. Guard
with the same "no FFTW → skip" pattern as `test/test_native_phase*.jl`.

---

## 5. How to verify the fix does not reintroduce the reproducibility gap

**Do not promise the 43-file single-vs-split count returns to 42098.** That number
(42098 single / 42081 split) reflects *both* channels. Turning persistence off
removes the disk channel but leaves the pool channel, so a residual single-vs-split
difference *may persist* and would be a **test-topology artifact, not a user bug.**
Frame verification around what the fix actually guarantees, and design the
experiment to *distinguish the two channels*:

1. **Disk-channel determinism (the real win — this MUST hold).** With the toggle
   **off** (the new default), run the same simulation (or the 43-file `rust` group)
   **twice in a row on the same machine, same process topology**, starting from a
   *pre-deleted* cache dir the first time. The two runs must produce **identical**
   assertion counts / step paths. Before the fix, run 2 could differ from run 1
   because run 1 wrote wisdom that run 2 imported. After the fix, no file is
   written, so run-to-run on fixed topology is deterministic. This is the headline
   guarantee and the acceptance criterion.

2. **Isolate the residual (pool) channel.** Re-run the 43-file group **single
   process vs split**, toggle **off**, cache dir deleted first. Two outcomes, both
   acceptable and both informative:
   - Counts now **match** ⇒ the disk channel was the sole driver; determinism is
     fully restored for practical purposes. Best case.
   - Counts **still differ** ⇒ the residual is the shared in-process pool
     (a single-process test artifact). Confirm by a **third** control: run the 43
     files split into **one process each** (fully isolated) — that must be
     internally reproducible run-to-run and is the configuration real scans use.
     Document the single-process residual as an inherent consequence of the shared
     global FFTW pool (§1 pool channel), not a regression.

3. **Confirm the disk channel really is off.** After a full default-toggle run,
   assert the `native_fftw_wisdom_*threads` file does not exist / was not touched
   (this is T1 at gate scale).

Interpretation: the acceptance bar is **(1) holds** (fixed-topology run-to-run
determinism restored) and **(3) holds** (no disk writes by default). A residual in
(2) is expected and documented, not a failure.

---

## 6. Benchmark that gates the default decision

The whole "opt-in defeats the point" objection hinges on whether ESTIMATE-path
wisdom actually speeds anything up. Measure it before finalizing the default.

- **Extend `test/benchmark_native_default.jl`** (or add a sibling
  `test/benchmark_native_wisdom.jl` so the tracked CI number isn't perturbed) to
  add a **scan-shaped** case: construct N (e.g. 200–2000) `RustNativeStepper`s in a
  loop over a spread of grid sizes, timing total construction wall-clock, under
  three configs in **separate fresh processes** (cache dir cleared between):
  1. persistence **off** (proposed default);
  2. persistence **on, export every construction** (today's behavior);
  3. persistence **on, export once per process** (C1/C4 shape), if you want the
     C4 fallback number too.
- **Report:** (a) total construction time off vs on-every (Problem 1 magnitude);
  (b) whether config-2 constructions are *faster to plan* than config-1 (does
  imported wisdom help ESTIMATE at all?); (c) per-construction export cost
  (config-2 minus config-3).
- **Decision rule:** if config-2 planning is not measurably faster than config-1
  (expected, given ESTIMATE), **default OFF (C3) is justified with no downside.**
  If config-2 *is* meaningfully faster, reconsider C4 (default on, export-once) and
  record the numbers in `BACKLOG.md`.

Run standalone, uncontended: `julia --project test/benchmark_native_default.jl`
(never inside the shared test matrix — wall-clock needs a dedicated runner).

---

## 7. User-facing behavior changes

- **New default (persistence OFF):** out-of-the-box native runs no longer read or
  write the `native_fftw_wisdom_*threads` cache file. Effect on a **one-shot**
  simulation: none measurable (ESTIMATE plans were already the behavior; the cache
  bought little). Effect on **scans / long REPL sessions:** they lose whatever
  cross-run wisdom accumulation they currently get — but per §1/§6 that benefit is
  probably ~0 under ESTIMATE, and in exchange they gain **run-to-run
  reproducibility** and drop the per-construction disk-write. Net: acceptable, and
  arguably strictly better for a scientific tool where reproducible scans matter.
- **New opt-in (`LUNA_NATIVE_FFTW_WISDOM=1`):** restores today's persistence for
  anyone who measures a benefit on their hardware/config. Documented as
  "speed-over-strict-reproducibility." If C4 is adopted instead of C3 as default,
  the flag becomes a **kill-switch** (`=0`) with the opposite polarity — pick the
  polarity to match whichever default the §6 benchmark selects, and state it
  explicitly in `CLAUDE.md`.
- **Determinism boundary to communicate:** even with persistence on, exact
  step-path reproducibility is only guaranteed **one fresh process per simulation**
  (shared global FFTW pool). Scans that need bit-identical step paths should run
  one sim per process (which `ScanQueue` already does) and/or keep persistence off.
- **Open verify-later note (do not resolve now):** native always sets
  `FFTW_UNALIGNED` in its plan flags (`fftw.rs`, `ComplexFft1d::new`/
  `RealFft1d::new`). FFTW wisdom is keyed on exact flags, so it is **uncertain
  whether native's plans ever actually match/reuse Julia's FFTW.jl wisdom** (Julia
  doesn't force `FFTW_UNALIGNED`). This further weakens the "shared speedup"
  argument for persistence but is not needed to decide this fix — flag it for a
  future session, don't chase it here.

---

## 8. Execution checklist (for the future session)

1. `grep` existing `LUNA_` env reads in `src/RK45.jl` to match naming/truthiness
   convention; pick the flag name.
2. Edit `src/RK45.jl`: add the toggle helper; gate the import path (L~965–986) to
   pass `""` when off; gate the export block (L~1830–1837) to skip when off.
3. Add tests T1/T2 (`test/test_native_fftw_wisdom.jl` or fold into an existing
   `test_native_phase*.jl`), env-var set/restore in `finally`, FFTW skip-guard.
4. Extend the benchmark (§6); run it standalone; record numbers.
5. Run verification §5 steps 1 & 3 (must pass); run step 2 and document the
   residual.
6. Update `BACKLOG.md` S1 item 1 (resolve the open risk), `CLAUDE.md` toggle list,
   `docs/native-port/ARCHITECTURE.md` §6, `VANILLA_LUNA_ISSUES.md`.
7. Full 7-group gate as the final correctness check (expect `rust` green; the exact
   count may legitimately shift from 42098 now that default import is off — that is
   the *point*, not a regression; confirm no *failures*).
8. Commit (no `Co-Authored-By` trailer — user preference).
