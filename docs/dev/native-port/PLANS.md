# Native-port plans and design records

The four per-task plan documents that used to live beside this file, merged
2026-07-22 so the native-port doc set is one reference set plus one plans file.
Each section below is the original document verbatim apart from its status
line, which has been refreshed against the current code.

**These are decision records, not a queue.** Two of them (S1 item 6, S6 item 3)
exist to record a deliberate *no* — read them before proposing that work again.
Live status for every item is in [`../BACKLOG.md`](../BACKLOG.md); this file is
the durable rationale.

| Section | Item | Status |
|---|---|---|
| [FFTW wisdom persistence](#1-fftw-wisdom-persistence-in-the-native-path-backlog-s1-item-1) | S1 item 1 | **Implemented** — opt-in via `AMALTHEA_NATIVE_FFTW_WISDOM`, default OFF |
| [SoA conversion](#2-full-soa-conversion-of-the-resident-field-backlog-s1-item-6) | S1 item 6 | **Parked — negative ROI.** Phase 0 committed as infrastructure; Phases 1-4 will not be built |
| [Threading the native RHS](#3-threading-the-native-rhs-backlog-s2) | S2 | **Mostly landed.** Only free-space 3-D FFT threading remains |
| [Standalone CLI](#4-standalone-cli-luna-cli-backlog-s6-item-3) | S6 item 3 | **Parked — recommend against building as specified.** A smaller "dump-and-replay" alternative is sketched |

---


## 1. FFTW wisdom persistence in the native path (BACKLOG S1 item 1)

Status: **Implemented.** Written 2026-07-09 as a plan; the fix has since
landed — native-path wisdom persistence is opt-in via
`AMALTHEA_NATIVE_FFTW_WISDOM`, **default OFF** (`src/Config.jl:96`,
`src/RK45.jl:50`, `test/test_native_fftw_wisdom.jl`). The analysis below is
retained because it is the only written record of *why* persistence is
opt-in rather than on: the two risks it identifies (per-construction export
cost, and wisdom-dependent plan selection making runs non-reproducible
across machines) are the reasons for the default.

Related tracking: `BACKLOG.md` → S1 item 1 (now in
[`../ARCHIVE.md`](../ARCHIVE.md)). Cross-reference:
[`VANILLA_LUNA_ISSUES.md`](VANILLA_LUNA_ISSUES.md). Do not treat this plan or
that doc as more authoritative than the primary sources (the code +
`BACKLOG.md`).

---

### 1. Context (why this matters — read first)

The native-Rust resident stepper (`RustNativeStepper` in `src/RK45.jl`, backed by
`NativeSim`/`CpuNativeSim` in `amalthea/src/native.rs`) owns its own resident
FFTW plans for the lifetime of a `solve()`. Earlier today (commit `1d8ab82`,
re-verified `f88ea43`) we added **FFTW planner-wisdom persistence** to this path,
mirroring how vanilla Julia Luna's `Utils.loadFFTwisdom`/`saveFFTwisdom` wrap
every `prop_capillary`/`prop_gnlse` setup. The mechanism:

- `amalthea/src/fftw.rs`: `FftwApi` binds `fftw_import_wisdom_from_filename` /
  `fftw_export_wisdom_to_filename` (dlsym, both best-effort — any failure returns
  `false`, never panics).
- `amalthea/src/native.rs`: `set_fftw_plans` takes a trailing `wisdom_path` and
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
   (`src/RK45.jl:963` passes `FFTW.FFTW_jll.libfftw3`; `amalthea/src/fftw.rs:64`
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

### 2. Confirmed current state (exact locations, verified at write time)

`amalthea/src/fftw.rs`
- L152–156: `ImportWisdom` / `ExportWisdom` fn-ptr typedefs (return `c_int`,
  nonzero = success — FFTW's convention, opposite this file's usual 0=success).
- L171–172, 218–221, 235–236: `Option<…>` fields on `FftwApi`, dlsym'd in `load`
  (miss → `None` → later no-op).
- L247–255: `import_wisdom_from_filename(&self, path) -> bool` — takes
  `PLANNER_LOCK`, calls the fn, `!= 0`.
- L260–268: `export_wisdom_to_filename(&self, path) -> bool` — same shape.
- Note L16–22: the planner-flag doc — bit-parity requires matching the flag used
  by the run under test; tests use `:estimate`, package default `:patient`.

`amalthea/src/native.rs`
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

### 3. Candidate fixes (with tradeoffs)

Throughout: "import" = reading the on-disk file into the shared pool before
planning (the reproducibility-relevant action); "export" = writing the pool to
disk after planning (the perf-relevant action).

#### C1. Export once per process (guard flag), keep import every construction
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

#### C2. Construction-count / elapsed-time threshold (export every Nth, or ≤1×/M s)
- *Pros:* bounds export cost while still refreshing wisdom occasionally.
- *Cons:* introduces an arbitrary tuning constant with no principled value;
  still doesn't address Problem 2; more code/state than C1 for no reproducibility
  gain. Rejected — complexity without solving the harder problem.

#### C3. Env-var opt-in, **default OFF** (mirror the `AMALTHEA_USE_RUST_CUDA_NATIVE`
gating pattern)
- *How:* Julia reads e.g. `AMALTHEA_NATIVE_FFTW_WISDOM` once. Default/unset ⇒ pass an
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

#### C4. Default ON but export-once-per-process; add a kill-switch env var
- Essentially C1 + an escape hatch. *Pros:* preserves any speedup by default while
  bounding export cost; kill-switch gives reproducibility-sensitive scans an out.
  *Cons:* more machinery (guard flag *and* env read); **default still imports
  every construction, so the default still has Problem 2's disk channel** — the
  worst reproducibility flavor stays on by default. Strictly more complex than C3
  and worse on reproducibility-by-default.

#### Where the fix should live
Prefer **Julia (`src/RK45.jl`)**: the env read + gating both blocks there means
**no Rust rebuild** is needed and the empty-path / skip-export contract already
exists on the Rust side (import treats `""` as no-op; export is simply not called).
The Rust import/export plumbing (`fftw.rs`, `native.rs`) stays as-is — it is
correct and reusable; we only change *when Julia asks for it*. Touch Rust only if
you later choose a Rust-side `AtomicBool` guard (not recommended — keep state in
Julia where the policy lives).

---

### 4. Recommended approach

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

#### Concrete steps

1. **`src/RK45.jl` — read the toggle once.** Near the top of the module (beside
   the other `AMALTHEA_USE_RUST_*` reads), add a small helper, e.g.
   `_native_wisdom_enabled() = get(ENV, "AMALTHEA_NATIVE_FFTW_WISDOM", "0") == "1"`
   (choose the exact name to match repo convention — grep existing `AMALTHEA_` env
   reads first; `AMALTHEA_USE_RUST_CUDA_NATIVE` uses `"1"` truthiness). Default OFF.

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
   `CLAUDE.md`'s env-toggle list and to `docs/dev/native-port/ARCHITECTURE.md` §6
   (fallback/opt-in catalog) documenting `AMALTHEA_NATIVE_FFTW_WISDOM`. Note the
   determinism boundary ("one process per sim") in `VANILLA_LUNA_ISSUES.md`.

#### New tests

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

### 5. How to verify the fix does not reintroduce the reproducibility gap

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

### 6. Benchmark that gates the default decision

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

### 7. User-facing behavior changes

- **New default (persistence OFF):** out-of-the-box native runs no longer read or
  write the `native_fftw_wisdom_*threads` cache file. Effect on a **one-shot**
  simulation: none measurable (ESTIMATE plans were already the behavior; the cache
  bought little). Effect on **scans / long REPL sessions:** they lose whatever
  cross-run wisdom accumulation they currently get — but per §1/§6 that benefit is
  probably ~0 under ESTIMATE, and in exchange they gain **run-to-run
  reproducibility** and drop the per-construction disk-write. Net: acceptable, and
  arguably strictly better for a scientific tool where reproducible scans matter.
- **New opt-in (`AMALTHEA_NATIVE_FFTW_WISDOM=1`):** restores today's persistence for
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

### 8. Execution checklist (for the future session)

1. `grep` existing `AMALTHEA_` env reads in `src/RK45.jl` to match naming/truthiness
   convention; pick the flag name.
2. Edit `src/RK45.jl`: add the toggle helper; gate the import path (L~965–986) to
   pass `""` when off; gate the export block (L~1830–1837) to skip when off.
3. Add tests T1/T2 (`test/test_native_fftw_wisdom.jl` or fold into an existing
   `test_native_phase*.jl`), env-var set/restore in `finally`, FFTW skip-guard.
4. Extend the benchmark (§6); run it standalone; record numbers.
5. Run verification §5 steps 1 & 3 (must pass); run step 2 and document the
   residual.
6. Update `BACKLOG.md` S1 item 1 (resolve the open risk), `CLAUDE.md` toggle list,
   `docs/dev/native-port/ARCHITECTURE.md` §6, `VANILLA_LUNA_ISSUES.md`.
7. Full 7-group gate as the final correctness check (expect `rust` green; the exact
   count may legitimately shift from 42098 now that default import is off — that is
   the *point*, not a regression; confirm no *failures*).
8. Commit (no `Co-Authored-By` trailer — user preference).


---


## 2. Full SoA conversion of the resident field (BACKLOG S1 item 6)

Status: **Parked 2026-07-10 (negative ROI). Phase 0 stays committed as
infrastructure; Phases 1-4 will not be built.** Explicitly flagged for
reconsideration once the rest of the current BACKLOG.md work is finished
— see "Ceiling measurement" below for why, and revisit if a future
workload profile changes the picture. See BACKLOG.md's S1 item 6 entry
for the live status line; this file is the durable writeup (survives a
context reset), not the status tracker.

### Ceiling measurement (2026-07-10) — read this before resuming Phase 1

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

**Cross-workload confirmation (2026-07-10), at the user's request before
any final call:** re-ran the same measurement on two more workloads.
Radial geometry (QDHT, N=32 r-points, 500 steps): apply_prop ~2.5% of
step() time. A 16×-larger mode-avg+plasma grid (trange=4ps, n_spec=16385
vs. the default benchmark's ~1025, 500 steps): apply_prop ~1.9%. The
ceiling does not grow with grid size (an O(n) op next to O(n log n) FFTs
should, if anything, shrink its share as n grows — consistent with what
was measured) and doesn't change materially across geometry. This is not
an artifact of the one default-benchmark workload — the ~2% figure is
representative.

### Context

BACKLOG.md S1 item 6 asked for a timeboxed Criterion spike comparing AoS
(interleaved `Complex<f64>`, today's layout throughout `native.rs`) against
SoA (separate re/im `Vec<f64>`) for the exp-linop hot loop, with the rule:
only do the invasive `fftw_plan_guru_split_dft` rewrite if the spike shows
>1.3x. The spike (`amalthea/benches/exp_linop_layout_bench.rs`, committed
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

### Phased implementation

Each phase lands as its own commit, gated by the **full** test suite (Rust
`cargo test` + the full 7-group Julia gate via
`test/parallel_rust_tests.py` for `rust` and the 6-group run for the rest —
per standing project practice, never a phase-specific subset), before
starting the next phase.

#### Phase 0 — `fftw.rs`: split-array plan bindings — DONE 2026-07-10
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

#### Phase 1 — universal RK buffers + `ExpCache` (the benchmarked win) — NEXT
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

#### Phase 2 — one geometry at a time: mode-avg → radial → modal → free-space
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

#### Phase 3 — FFI boundary shim
`native_step`'s `yn` output pointer and `set_field`/`get_field` must keep
presenting interleaved `ComplexF64` to Julia (Julia's own array layout is
always interleaved — this is not something to change). Add a thin
interleave/deinterleave conversion exactly at this crossing (once per
`native_step` call, not per internal operation) so the Rust-internal SoA
representation is invisible to the FFI contract. Verify: `test_rust_ffi.jl`
and the full native-port test files still pass with no Julia-side changes
needed.

#### Phase 4 — cleanup + close out
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

### Verification (every phase)

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

### Gotcha: sandbox filesystem restrictions during test runs

Not related to the SoA work itself, but bit this session (2026-07-10):
running the Julia test gate under the default sandboxed Bash tool fails
every single test with `SystemError: opening file
"/home/diego/.julia/logs/scratch_usage.toml": Read-only file system` —
Julia's `Scratch.jl` tries to write a usage-tracking file outside the
sandbox's writable paths on every `Scratch.get_scratch!` call (used by
`Utils.cachedir()`). This produces failures that look like a real
regression (all workers `rc=1`, garbled pass/total counts) but are purely
a sandbox artifact. Fix: run the test commands with the sandbox disabled.


---


## 3. Threading the native RHS (BACKLOG S2)

Status: **Phases 1-3 (radial: plumbing, measurement, FFT+plasma) + Phase 4
Raman-radial + Phase 4 modal all done. RE-LANDED 2026-07-11; modal threading
landed 2026-07-20.** Only free-space 3-D FFT threading (Phase 4, last bullet)
remains open. Phase 3 was originally
implemented, committed, pushed, then REVERTED (2026-07-10) after a
post-push wall-clock benchmark surfaced an intermittent segfault. The
revert's isolation experiment correctly implicated
`apply_plasma_radial`'s parallel branch but misdiagnosed the mechanism as
a Rust-side out-of-bounds write. **Actual root cause (found 2026-07-11 via
ASAN + Valgrind + a direct crash repro with `coredumpctl`/gdb): a missing
GC root, not a memory-safety bug in the parallel code.**
`native_set_plasma_params` stores a raw pointer into a Julia-owned,
Rust-allocated `PptIonizationRate`/`AdkIonizationRate`, re-dereferenced on
every future `native_step` call — but `RustNativeStepper` never kept a
Julia-level reference to the owning handle after construction, so Julia's
GC (which runs concurrently with a blocking `ccall`, invisible to native
Rust threads) could finalize/free it mid-solve. Threading only widened
the race window; the same latent bug existed on the sequential path too.
Fixed via a new `_gc_roots::Vector{Any}` field on `RustNativeStepper`
(`RK45.jl`) that roots every such handle for the stepper's lifetime — see
BACKLOG.md's S2 entry for the full postmortem, repro method, and
verification (8-cycle stress repro with forced `GC.gc()` between cycles,
full 7-group gate). This file is the durable plan (survives a context
reset), not the status tracker — see BACKLOG.md's S2 entry for the live
status line.

### Context

S1.6 (the SoA/split-complex conversion) was measured and parked: the
exp-linop multiply it targeted is only ~2% of `step()` time, so its
ceiling was too small to justify the rewrite. That investigation also
confirmed FFT/RHS work dominates `step()` time — which is exactly what S2
targets: the resident RHS (`rhs_radial`, `rhs_modal`, `rhs_free`) runs
single-threaded per stage today.

An Explore survey (2026-07-10) found the risk profile is better than the
backlog's own framing suggested. The backlog warns "one scratch slab per
rayon worker (never shared) — this is precisely the bug the Julia
`Threads.@threads` `pointcalc!` race had." In `rhs_radial`
(`amalthea/src/native.rs:1343+`), the two candidate FFT loops (Step 1
`fft.inverse` per r-column, Step 6 `fft.forward` per r-column) already
operate on **disjoint column slices of column-major, matrix-shaped
buffers** (`radial_eoo`/`radial_eto`/`radial_pto`/`radial_poo`, each sized
`n_time_over*n_r` or `n_spec_over*n_r`) — not a single shared per-call
scratch reused sequentially. FFTW's own documented contract (`fftw.rs:32-34`)
is that `fftw_execute*` (unlike `fftw_plan_*`) is thread-safe when called
concurrently against one shared plan with distinct buffer arguments. Same
finding for `apply_plasma_radial` (native.rs:982-1032): `plas_rate`/
`plas_fraction`/`plas_phase`/`plas_j`/`plas_p` are already `n_time_over*n_r`
matrices indexed by the same `start..end` column slices — also
safe-by-construction. The one genuine shared-mutable-state hazard is
`apply_raman_radial`'s single resident `raman_solver` and (when
`raman_thg == false`) the shared Hilbert-transform scratch — both reused
sequentially across r-columns today, which *would* race under naive
parallelization. QDHT itself is a single batched dense operation across
all columns, not a per-column loop — it already has its own internal
rayon fallback path when no BLAS is found.

**Scope for this pass:** radial geometry only (BACKLOG.md S2 items 1-2).
Modal (item 3) and free-space (item 4) are structurally independent and
left as separate, not-started follow-ups. Item 5 (error-norm reduction
stays sequential) needed no code change — confirmed already sequential.

### Phase 1 — plumbing: `native_set_threads` + per-sim thread pool — DONE 2026-07-10

Added `n_threads: usize` (default 1) and `thread_pool: Option<rayon::ThreadPool>`
to `CpuNativeSim`, a `set_threads` `NativeBackend` trait method (no-op
stub for `CudaNativeSim`), and a `native_set_threads` FFI entry point,
wired from Julia's `Threads.nthreads()` at `RustNativeStepper`
construction (`src/RK45.jl`, right after `native_set_fftw_plans`).
Deliberately builds a **per-sim** `rayon::ThreadPool` (not
`rayon::ThreadPoolBuilder::build_global()`) — the global pool can only be
configured once per process and would conflict with `QdhtFfiHandle`'s
existing rayon fallback usage (currently ungated) and with multiple
sims/tests coexisting in one Julia process. Purely additive — `n_threads`
is stored but not yet read by any RHS code. Verified behavior-neutral:
full 7-group gate matches all baselines exactly.

### Phase 2 — measurement (temporary instrumentation, reverted after reading) — DONE 2026-07-10

Added temporary `Instant`-based profiling to `rhs_radial` (same pattern as
the S1.6 ceiling measurement: add, measure, revert, never commit), timing:
(a) the two FFT-per-column loops, (b) `qdht.apply_real` calls, (c)
`apply_plasma_radial`, (d) everything else (Kerr, window, noise). Measured
on `N=32` and `N=128` r-points, with and without plasma (500/200 fixed-dt
steps respectively):

| N | plasma | FFT | QDHT | plasma | other |
|---|---|---|---|---|---|
| 32 | off | 35.9% | 46.7% | — | 17.4% |
| 32 | on | 20.9% | 28.3% | 40.5% | 10.2% |
| 128 | off | 18.6% | 72.6% | — | 8.8% |
| 128 | on | 14.1% | 54.8% | 24.5% | 6.6% |

**Decision: proceed to Phase 3.** Unlike S1.6's ~2% ceiling, FFT+plasma
combined is 38-61% of `rhs_radial` time for plasma-enabled configs —
comfortably clears the bar for the parallelization work below. QDHT
dominates the rest but is already internally parallel/BLAS-backed (S1
item 5) — out of scope for this item. Instrumentation was fully reverted
(`git diff` empty on `native.rs`) before Phase 3 started.

### Phase 3 — thread the safe seams: FFT-per-column + plasma-per-column — NEXT

Gated behind `self.n_threads > 1` (via
`self.thread_pool.as_ref().unwrap().install(|| { ... })`, building the
pool lazily on first use):
- Step 1 (`fft.inverse` per r-column) and Step 6 (`fft.forward` per
  r-column) in `rhs_radial`: replace the sequential loop with
  `par_chunks_mut`/zipped `par_iter` over the disjoint column slices.
  Same for `rhs_radial_env` (EnvGrid, `ComplexFft1d`/complex chunks).
- `apply_plasma_radial`'s r-loop: same `par_chunks_mut` treatment — a
  direct swap of the loop construct, not a data-layout change (buffers
  are already matrix-shaped).
- **Raman stays sequential in this phase** (`apply_raman_radial` keeps
  its single shared `raman_solver` — parallelizing it safely needs one
  `TimeDomainRamanSolver` instance per rayon worker, a separate follow-up,
  not bundled here).
- QDHT, Kerr, windowing, noise injection: left as-is (QDHT already has
  its own internal parallelism; the others are cheap flat elementwise ops
  — the "other" column above is 6.6-17.4%, not worth the same treatment
  unless a future measurement says otherwise).

**Verify — bit-identical, not tolerance-based:** every parallelized loop
writes to disjoint memory with no cross-column reduction, so
`n_threads=1` vs. `n_threads=4` must produce **exactly** the same output.
Add this as an explicit assertion in `test_native_radial.jl` and
`test_native_radial_plasma.jl`: run the same solve at both thread counts,
assert exact equality (not `norm(...) < tol`). Any mismatch is a real bug
(an assumed-disjoint slice aliased, or a chunk-size miscalculation at a
specific `n_r % n_threads`), never a tolerance to loosen. Full 7-group
gate at `n_threads=1` (must match all baselines exactly) plus a second
`rust`-group-only run with 4 threads.

### Phase 4 — Raman per-worker solver, modal, free-space

- **Raman per-worker `TimeDomainRamanSolver` — DONE 2026-07-20** (radial,
  item 1 of the BACKLOG S2 entry). Each rayon task owns its **own cloned**
  solver + Hilbert scratch over a contiguous r-column group (no
  `current_thread_index`, no interior mutability). `solve()` resets state at
  entry, so a clone is bit-identical to the shared one.
- **Modal (`rhs_modal` batch callbacks) — DONE 2026-07-20.** The two
  cubature `integrand_v` callbacks (`modal_integrand_v` /
  `modal_integrand_v_full`) now hand off to `CpuNativeSim::modal_eval_pairs`,
  which parallelizes the batch's nodes across rayon workers when
  `n_threads > 1`. The core refactor: `rhs_modal_pointcalc` (a `&mut self`
  method scribbling on ~13 shared `self.modal_*`/`raman_*` scratch buffers)
  became a free-standing associated fn `modal_pointcalc(ro: &ModalRO,
  sc: &mut ModalScratch, r, θ, out)` — read-only sim state in a `Sync`
  `ModalRO` view (`&`/`Copy`/`Option<&Plan>`, FFT wrappers already `Sync`),
  every written buffer in a per-worker `ModalScratch` (pooled on
  `self.modal_scratch_pool`, entry 0 = sequential path). Nodes are split into
  `≤ n_threads` contiguous groups (`par_chunks` over coords + `fvals` zipped
  with `par_iter_mut` over the pool), each group's `out[p*fdim..]` disjoint,
  no cross-node reduction ⇒ **bit-identical** `n_threads`=1-vs-4. Measured
  parallelizable fraction (temp `Instant` counters, reverted): 90.3%
  (full=false 1 mode), 95.6% (full=true), 82.8% (2-mode) — well above the
  proceed bar (radial was 38-61%, S1.6 parked at 2%). Raman-modal threaded
  too: each worker's `ModalScratch` carries its own cloned solver + Hilbert
  scratch (same discipline as radial), Hilbert FFT plan shared read-only. No
  new GC-root hazard (the solver is Rust-owned/cloned, not a persistent raw
  pointer into Julia memory — contrast the plasma-pointer UAF fixed on
  `main`). Test: `test/test_native_modal_threading.jl` (Kerr full=false/
  full=true/2-mode/npol=2, Raman :N2, + a forced-`GC.gc()` stress loop).
- Free-space 3-D FFT threading via `fftw_init_threads`/
  `fftw_plan_with_nthreads` (new bindings in `fftw.rs`, a planning-time
  setting under `PLANNER_LOCK`) — **still open**. `fftw.rs`'s
  `unsafe impl Sync` only holds for single-threaded plans, so an
  `nthreads>1` plan would deadlock under the existing concurrent
  `fftw_execute_dft` path; needs an isolated multi-threaded plan first.

### Verification (every phase)

- `RUSTFLAGS="-D warnings" cargo test --release` — must stay green.
- Full 7-group Julia gate, sandbox disabled (see gotcha in
  `PLANS.md` §2): `rust` via
  `python3 test/parallel_rust_tests.py` (baseline 42087/42087), other 6
  via `LUNA_TEST_GROUP=<group> julia --project test/runtests.jl`.
- Phase 3's bit-identical n_threads=1-vs-4 assertion is the key new
  correctness gate — treat any mismatch as a bug, not noise.


---


## 4. Standalone CLI, `luna-cli` (BACKLOG S6 item 3)

Status: **Measured, then parked — recommend against building the item as
currently specified. A much smaller alternative ("dump-and-replay") is
sketched below as the only version of this idea that looks worth building
today.** This is a feasibility/design pass, not an implementation — no
Rust source was written, no crate dependencies were added, no existing
code was changed. Research was cut short by a mid-session scope change
("wrap up promptly, free the machine"); several sub-questions are marked
**OPEN** below rather than filled in from inference. Treat this doc as a
first pass to be extended, not a closed investigation.

### The item as currently written

> Standalone CLI (item 14) — `amalthea/src/bin/luna-cli.rs` (`[[bin]]` in
> the existing crate, not a new workspace member). TOML config in, scoped
> to native-eligible configs, HDF5 out; compare against a Julia run of the
> same config as the acceptance test. WASM demo only after the CLI exists.

Note: BACKLOG.md's own text still says `amalthea/src/bin/luna-cli.rs`
(the crate directory is `amalthea/` — confirmed via `amalthea/Cargo.toml`'s
`[package] name = "amalthea"`). Stale path, same pattern CLAUDE.md already
flags elsewhere in this file's prose.

### Why this is not just "add a `[[bin]]`"

`docs/dev/native-port/ARCHITECTURE.md:161-195` (§6, written when the
native-port phases closed out) already answers the question this plan set
out to ask, in general terms:

> **(a) One-time setup and orchestration — stays Julia by design.** The
> high-level `Interface.jl` / `Output.jl` / `Grid.jl` / `Fields.jl` setup
> code (grid construction, mode solvers, Sellmeier/gas data, pulse
> synthesis), HDF5 output ... This runs once per simulation, costs
> milliseconds, and porting it buys nothing. The port's goal was "the
> *per-step hot loop* is 100% Rust", and that is achieved ...
>
> **(c) Arbitrary user closures — the one genuine barrier.** ... Closing
> this category fully would mean a declarative config format instead of
> Julia closures — that is SUGGESTIONS.md item 14's CLI, which sidesteps
> Julia entirely rather than porting it.

In other words: the project already looked at "should we port setup to
Rust" and answered no, because the entire point of the native port was
the *per-step* loop, and setup is a one-time, millisecond-scale cost. A
standalone CLI needs setup ported anyway — for a different reason (no
Julia present at all, not perf) — but it is the same porting work
ARCHITECTURE.md already classified as high-effort/low-value for the
per-step goal. This plan's job was to find out whether that porting work
is actually small (making the CLI cheap despite ARCHITECTURE.md's framing)
or actually large (confirming it). Evidence below points to large.

### Setup-path inventory: mode-averaged RealGrid, Kerr-only, single gas, constant pressure

Target config (the narrowest genuinely useful case, matching
`test/benchmark_native_default.jl`'s flagship workload minus plasma):
`prop_capillary(radius, flength, gas, pressure; λ0, λlims, trange, energy,
τfwhm, modes=:HE11, kerr=true, plasma=false, raman=false)`. Traced
`src/Interface.jl:385-450` (`_prop_capillary_args`) end to end. Classified
each piece the CLI would need to reproduce in Rust with no Julia present:

| Piece | Source | Classification | Evidence |
|---|---|---|---|
| Grid ω/t axes, windows | `Grid.jl` `RealGrid(...)` | **(b) small port** | `Grid.jl:38` builds axes via `FFTW.fftfreq`/`fftshift` (`Grid.jl:150,219,224`) — standard closed-form DFT-frequency arithmetic, no adaptive logic. Windows (`ωwin`/`twin`) are separately-defined closed-form tapers. |
| `MarcatiliMode` neff/β(ω) | `Modes.β` → `Capillary.jl` | **(a) already in Rust** | `amalthea/src/dispersion.rs` has `MarcatiliNeff`/`ChebyshevDispersion` (CLAUDE.md-confirmed, matches Julia bit-for-bit per existing kernel wiring). |
| `β1const` (linop group-velocity term) | `Modes.dispersion(mode,1,ω0)` = `Maths.derivative(...)` | **(b) small port** | `Modes.jl:188-199` — a genuine one-time scalar finite difference (not the Phase-7 analytic closed form, which is only for the two-point z-dependent case). Cheap to replicate once at setup. |
| `pre` (nonlinear prefactor) | `NonlinearRHS.jl:602-614` | **(b) small port** | Closed-form arithmetic on `grid.ω` + `PhysData.c`/`ε_0` constants. |
| `kerr_fac` (`density·ε_0·γ3`) | `RK45.jl:1244`, `PhysData.density`/`PhysData.γ3_gas` | **(c) large / hard dependency — worse than first assessed** | `γ3_gas` (`PhysData.jl:668-699`) is closed-form arithmetic, but it scales off `dens_1atm_0degC[material]` (`PhysData.jl:798`), a module-load-time constant computed via `density()`. **`PhysData.density(material,P,T)` (`PhysData.jl:784-795`) calls `CoolProp.PropsSI("DMOLAR",...)`** — a real-gas equation of state from the external C++ `CoolProp` library (`PhysData.jl:3` imports it), not the ideal-gas law. Reproducing Julia's actual number needs either dlopening CoolProp itself (same pattern as HDF5/FFTW, but a new native dependency with its own portability story) or accepting an unquantified systematic offset from an ideal-gas approximation at real operating pressures. This is a genuine hard dependency, not a "big but mechanical" table-transcription job as first assessed. |
| `sqrt_aeff` (effective-area normalization) | `Modes.Aeff(mode)` → `Capillary.jl:290-297` | **(b/c) medium port — smaller than first assessed** | `Aeff = radius²·aeff_intg`, where `aeff_intg = Aeff_Jintg(n,unm,kind)` is a 1-D `hquadrature` over `r·J_{n-1}(u_nm·r)^4` (`Capillary.jl:290-295`) and `unm` is a Bessel-zero from `FunctionZeros.besselj_zero` (`Capillary.jl:249-259`). The needed *quadrature primitive already exists*: `amalthea/src/cubature.rs` dlopens the native C `libcubature` (`pcubature_v`/`hcubature_v_2d`, lines 148-172) and is usable standalone today, with no Julia dependency, for exactly this kind of integral. What's actually missing is a Bessel-J evaluator and a Bessel-zero root-finder — both well-known, portable, closed-form-adjacent algorithms with no external-library dependency required — not a from-scratch quadrature engine. One-time/scalar (evaluated once per mode construction, not per RHS step). |
| Setup FFI hand-off | `RK45.jl:1250-1259` `ccall(:native_set_mode_avg_params,...)` → `amalthea/src/native.rs:5129` | confirms the shape of the gap | Rust's FFI entry point stores every argument (`towin`, `owin`, `sidx`, `pre` re/im, `beta`, `kerr_fac`, `nlscale`, `sqrt_aeff`) verbatim — it does **not** recompute any of them today. A standalone CLI must become the thing that computes all of the above, in Rust, with no fallback to ask Julia. Confirmed separately: `dispersion.rs`'s `MarcatiliNeff`/`ZeisbergerNeff` (already-in-Rust per the per-kernel wiring table) are **not** on this codepath at all — Julia's `beta` array here traces back through `NonlinearRHS.norm_βfun`/`Modes.dispersion`, never through a `ccall` into those Rust kernels. So "dispersion is already in Rust" (true for the older per-kernel accelerator) does not actually close this particular gap. |
| Pulse synthesis (`GaussPulse`/`SechPulse` → initial `Eω`) | `src/Fields.jl:105-195,654-664` | **(b) small port** | `PulseField` builds `Et` from a closed-form envelope (`Maths.gauss`/`sech`), transforms via `FT*Et` (RealGrid rfft — already in `fftw.rs`), and normalizes energy via `energyfuncs` (`Fields.jl:654-664`), a Simpson's-rule integral over `|Et|²`/`|Eω|²` — no root-finding, no external dependency. |
| Shot noise (`Et_noise`/`Emω_noise`) | `Interface.jl` `makenoise` | **OPEN — not independently verified this session** (no fork thread reached it before the cutoff). Needed even for a "Kerr-only" config since `shotnoise=true` is `prop_capillary`'s default; flag as a prerequisite check for whoever resumes this. |

**Reading across the table:** the grid axes, `pre`, `β1const`, and pulse
synthesis are genuinely small, closed-form ports. But two pieces are not:
**`kerr_fac` needs `PhysData.density`'s real-gas EOS, which is CoolProp —
an external C++ library, dlopenable in principle but a wholly new native
dependency with its own portability story**, and `Aeff` needs a Bessel-J
evaluator + Bessel-zero root-finder (smaller than first assessed, since
`cubature.rs` already supplies the quadrature primitive, but still new
special-function surface). This is materially more than "add a `[[bin]]`
and a TOML parser" — it is a second, smaller-but-real porting project
layered on top of the one that's already done, for code the native
port's own architecture doc already decided wasn't worth porting. Also
confirmed (fork B): `dispersion.rs`'s already-ported `MarcatiliNeff` is
not actually on the mode-averaged `beta`-array codepath at all (see table
above) — the "dispersion is already in Rust" comfort from the per-kernel
wiring table does not transfer to this use case.

**Additional module-level findings (fork B, `amalthea/src/*.rs` inventory):**
- `grid.rs` (58 lines) is a bare data struct with a plain positional
  constructor — no logic today builds `RealGrid`/`EnvGrid` from
  `(zmax,λ0,λlims,trange,δt)` in Rust; the native stepper receives
  Julia-built `towin`/`ωwin`/`sidx` as flat arrays already. Confirms the
  "small port" classification above is about *effort to add*, not
  "already partially done."
- `ionization.rs`: `PptIonizationRate` is a generic spline-LUT engine
  (`::from_samples`) — the actual PPT rate formula
  (`src/Ionisation.jl:467`, special-function-heavy) exists only in Julia.
  `AdkIonizationRate` looks more self-contained but wasn't fully traced
  back to its constant sources — flag as open if plasma is ever in scope.
- `raman.rs`: the ADE solver math is fully self-contained Rust; only the
  oscillator-parameter *derivation* (rotational energy levels etc., from
  `src/Raman.jl:217`) is Julia-only. Out of scope for the Kerr-only V1
  target anyway.
- `cubature.rs`, `io.rs`: both already general-purpose and
  Julia-independent (confirmed **(a) already-in-Rust-standalone**) —
  the real gaps are the physics that would need to *call* them (Bessel
  functions for `cubature.rs`'s integrand; `Output.jl`'s stats/meta
  schema for `io.rs`'s writer, addressed next).

**`Output.jl`'s actual `HDF5Output` schema** (fork A, `Output.jl:161-259`):
a top-level extensible `"Eω"` dataset (chunked, z as last index), a `"z"`
1-D extensible dataset, `"stats/*"`, `"meta/*"` (sourcecode, git commit,
cache hash, `meta/cache` sub-group), and a `"prop_capillary_args"`
string-dict group (`Interface.jl:462-468`). Dim ordering matches the
column-major reversal `io.rs`'s existing `scan_write_point`/
`create_dataset_nd_julia` already handles — the field+z part is
essentially already solved by existing Rust infra; only `meta`/`stats`
would need adding for a byte-for-byte-comparable file, and (per
`Stats.default`, `Stats.jl:495-532`) stats are not required for a V1
acceptance test — comparison can be done directly on `Eω`/`z`, with
Julia able to recompute any stats post-hoc from a Rust-written
field/z file if deeper comparison is later wanted.

**Not verified this session (flag for whoever resumes):**
- `src/Interface.jl`'s `makenoise` (shot noise) — no fork thread reached
  this before the cutoff.
- `AdkIonizationRate::rate`'s (`ionization.rs:410-458`) full independence
  from Julia-only constants — read the struct, didn't trace every
  constant to its `PhysData.jl`/`Ionisation.jl` origin. Only matters if
  plasma is ever brought into CLI scope.
- ~~Whether `optional = true` deps + `required-features` on a `[[bin]]`
  leaves `cargo build --release` (no `--features`) unaffected~~ —
  **resolved**: empirically verified in a scratch crate (not this repo,
  per the "no Cargo.toml changes" constraint): a `[[bin]]` with
  `required-features = ["cli"]` paired with `optional = true` deps gated
  behind `cli = ["dep:serde"]` is correctly skipped by plain `cargo
  build` (no bin built, no optional deps compiled); `--features cli`
  builds both. The mechanism works exactly as hoped — this is not a
  blocker.

### "Scoped to native-eligible configs" — what that boundary actually is

`docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md` gives the current
eligibility surface: mode-averaged is the most complete geometry (Kerr,
plasma, Raman, shot noise, Kerr-only mixtures, and a narrow
two-point-gradient z-dependent case are all native-eligible for RealGrid).
If "native-eligible" is read as broadly as the matrix allows rather than
just Kerr-only, the CLI's setup-porting burden grows further: plasma needs
`PhysData.ionisation_potential` plus PPT/ADK rate construction, Raman needs
gas-specific spectroscopic constants from `PhysData.jl`. None of that was
traced this session — flagged as **OPEN**.

### TOML config and the cargo dependency question

`amalthea/Cargo.toml` today has exactly three dependencies: `libc`,
`num-complex`, `rayon` (plus `windows-sys` gated on `cfg(windows)`) — **no
`serde`, `toml`, or `clap`**. The crate is `crate-type = ["cdylib",
"rlib"]`, built by `deps/build.jl` via plain `cargo build --release` and
loaded by every Julia session at package init.

The standard Cargo mechanism for adding an opt-in binary without touching
the default library build is: mark `serde`/`toml`/`clap` `optional = true`,
define a `cli` feature that enables them, and add `required-features =
["cli"]` to the `[[bin]]` entry. Under that shape, `cargo build --release`
(what `deps/build.jl` runs) should not build `luna-cli` or pull in its
dependencies at all — only `cargo build --release --features cli` would.
**Confirmed this session** (in a scratch crate, not this repo, per the
"no Cargo.toml changes" constraint): this shape works exactly as
described — plain `cargo build` skips both the bin and the optional deps,
`--features cli` builds both. This is not a blocker; the mechanism is
sound and this repo's `Cargo.toml` starts from zero `serde`/`toml`/`clap`
deps today, so there is nothing to disentangle from an existing graph.

### HDF5 output and the acceptance test

`amalthea/src/io.rs` already dlopens `libhdf5` at runtime and (S6 item 2,
done 2026-07-19) has `scan_write_point` + `create_dataset_nd_julia` for
writing arbitrary named ND datasets matching HDF5.jl's column-major
dim-reversal. That's a real head start — the low-level "can Rust write an
HDF5 file Julia can read back" question is already answered yes. What's
open is how much of `Output.jl`'s actual `HDF5Output` schema (stats
groups, `meta`/`meta/cache`, the `chash` consistency check, compression
options) a V1 CLI needs to reproduce for "compare against a Julia run of
the same config" to mean anything more than "the field arrays are
plausible." A defensible **V1 scope-down**: skip `stats` entirely (Stats.jl
computation is itself Julia-only machinery — energy, etc. — and isn't
needed to validate the field propagation itself), write only `Eω`/`z`
under the same dataset names Julia would use, and compare those two
arrays directly rather than trying to byte-match the whole HDF5Output
file structure.

**On the acceptance test's meaningfulness:** because the CLI (as
specified) reimplements setup independently in Rust rather than consuming
Julia's already-computed arrays, "compare against a Julia run of the same
config" is **not a bit-parity test** — it inherits every divergence in the
setup path (Sellmeier table transcription errors, `β1const`'s finite
difference implemented with a possibly-different step size or scheme than
Julia's `Maths.derivative`, `Aeff`'s quadrature/root-finder using
different convergence tolerances than `HCubature.jl`). This is
structurally the same situation CLAUDE.md documents for Phase 7's β1
analytic-vs-Julia-FD divergence (`docs/dev/native-port/BETA1_ANALYTIC.md`):
a real, small, systematic difference that can accumulate coherently over
a propagation. Expect a tolerance tier in the ~1e-4 to ~1e-6 range at best
(not the ~1e-8/~1e-12 tiers the per-kernel wiring table in CLAUDE.md
reports for kernels that consume identical, Julia-computed inputs) —
and only after every setup-path piece above is actually built and its own
divergence from Julia is characterized individually, the same way Phase 7
did (a `kerr=false` control run to isolate one variable at a time). No
number can be honestly stated before that work exists.

### WASM (per the item: "only after the CLI exists")

Independent of whether the CLI is built, WASM would additionally block on
the fact that **FFTW and HDF5 are both `dlopen`ed native shared libraries**
(`amalthea/src/fftw.rs`'s `Library::sym` dlopen infra; `io.rs` same
pattern). Neither `wasm32-unknown-unknown` nor WASI has a general
`dlopen` of arbitrary host `.so`/`.dylib` files — there is no library to
find at runtime in a WASM sandbox. A WASM build would need either (a)
statically-linked/vendored WASM builds of FFTW and HDF5 (a real question
mark for HDF5 in particular — this needs its own research, not assumed
possible), or (b) dropping HDF5 for the WASM target and replacing FFTW
with a pure-Rust FFT crate at the native-port's FFT boundary — itself a
non-trivial, separate rewrite of a hot-path dependency the rest of the
crate is built around. Either path is its own multi-session project,
independent of and larger than the CLI item itself.

### Alternative shapes considered

1. **The item as specified** (TOML → pure-Rust setup → HDF5 out, no Julia
   anywhere). Evidence above: large port (Bessel root-finder + quadrature
   for `Aeff`, full gas-table transcription, plus two open items — pulse
   synthesis and noise — that could add more). **Not recommended now.**

2. **Thin CLI that shells out to / embeds Julia.** Doesn't eliminate the
   Julia dependency at all — still needs a full Julia install +
   `Amalthea.jl` loaded, just wrapped in a different entry point. Defeats
   the stated purpose (a Julia-free artifact) and doesn't unblock WASM
   either, since embedding a Julia runtime in WASM is its own much larger
   problem than the FFTW/HDF5 one above. **Not useful as a way to satisfy
   this item's goal.**

3. **Dump-and-replay: Julia exports a one-time setup dump, a genuinely
   standalone Rust CLI replays it.** Julia runs `prop_capillary_args`
   once (as it does today) and, instead of constructing a
   `RustNativeStepper` in-process, serializes every argument the existing
   `native_set_mode_avg_params`/`native_set_plasma_params`/etc. FFI calls
   already pass (the exact arrays in the inventory table above — this
   reuses S6 item 2's `create_dataset_nd_julia` writer, or a simpler flat
   binary format) to a file. `luna-cli` (still behind an opt-in `cli`
   feature) loads that file and drives `native_step` to completion using
   the *existing, unmodified* `native.rs`, writing `Eω`/`z` out via
   `io.rs`. **This is dramatically smaller** — zero new setup-porting work
   (no Sellmeier tables, no Bessel root-finder, no pulse-synthesis port),
   because Rust never computes any of the setup; it only consumes
   Julia's numbers, exactly like every existing per-kernel wiring already
   does. The acceptance test is then genuinely bit-identical (same input
   arrays, same Rust code path, run once from Julia's process and once
   from `luna-cli` — the only variable is which process is driving
   `native_step`), which is a real, checkable acceptance criterion, unlike
   option 1. **Trade-off:** it is not what item 14 literally asks for — a
   Julia session still has to run once to produce the dump, so this
   doesn't remove Julia from the workflow, only from *repeated* runs of an
   already-configured simulation. Useful for exactly the parameter-scan
   use case the rest of S6 targets (re-running one setup many times with
   different seeds/energies without paying Julia startup cost per run) —
   a real, if narrower, value proposition. **This is the only version of
   "a standalone CLI" that looked buildable at reasonable cost from this
   session's evidence**, if there's appetite for something in this space
   at all.

### Recommendation

**Do not build the CLI as specified in BACKLOG.md's current one-paragraph
item.** The setup-porting cost is materially larger than the item implies
— it duplicates, for a different reason, exactly the work
`ARCHITECTURE.md` §6a already classified as high-effort/low-value
("porting it buys nothing"), and at least one piece (`Aeff`'s Bessel-zero
root-finder + adaptive quadrature) is genuine new special-function surface
this crate doesn't have today, not a mechanical translation. Two more
pieces (pulse synthesis, shot noise) weren't even confirmed closed-form
this session. No current user of this fork has asked for a Julia-free
artifact; the WASM motivation the item cites is blocked on a separate,
larger FFTW/HDF5-in-WASM problem regardless of whether the CLI exists.

**If there is appetite for something in this space**, option 3
(dump-and-replay) above is the only shape this session's evidence
supports as a reasonably-scoped V1: it reuses the entire existing
native-port stepper unmodified, needs no new numerics, and has a genuine
bit-identical acceptance test. It is a different, narrower feature than
"standalone CLI" — recommend re-titling/re-scoping the backlog item if
this path is chosen, rather than silently redefining item 14.

**A well-argued no is treated as a fully successful outcome for this
research task** (per the task's own framing, mirroring
`PLANS.md` §2's parked-with-reasoning precedent) — this is
that outcome.

### What would change this recommendation

- A concrete user/downstream request for a Julia-free CLI or WASM demo
  (nothing currently in the repo motivates this beyond the backlog item's
  own suggestion-list origin — worth checking `SUGGESTIONS.md` for the
  original ask's context if this is revisited).
- If the still-open items (pulse synthesis, shot noise, the exact
  `PhysData.jl` gas-table portability) turn out to be smaller than
  `Aeff`'s Bessel/quadrature dependency suggests the whole path is —
  possible, since they weren't checked this session, but nothing found so
  far points that way.
- If the dump-and-replay alternative (option 3) is deemed worth building
  as its own, smaller, differently-named item — that would be a
  reasonable next step; it is not this item as written.

### Session limitations (explicit, per the interrupted research)

This pass was cut short mid-investigation by a scope change to free the
machine, partway through four parallel research threads. All four did
return by the time this doc was finalized (the setup-path trace, the
`amalthea/src/*.rs` inventory, the `Fields`/`PhysData`/`Output`/`Stats`
trace, and the config/eligibility/Cargo-mechanics check), and their
findings are folded into the tables and sections above with file:line
citations. The one item no thread reached before the cutoff is
`Interface.jl`'s `makenoise` (shot noise construction) — left as an
explicit **OPEN** item above rather than inferred. Given the convergent,
mutually-reinforcing findings from three independent threads (the
CoolProp dependency, the `dispersion.rs`/`grid.rs` "already-in-Rust but
not on this codepath" pattern, and the dump-and-replay alternative being
independently proposed by two separate threads), the overall recommendation
below is reasonably well-supported despite the early cutoff — but
`makenoise` and `AdkIonizationRate`'s constant provenance (noted inline
above) should still be checked before treating this as a closed
investigation.


---

