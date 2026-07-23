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


## 5. Native multi-mode `StepIndexMode` (BACKLOG Phase I item 5)

Status: **Feasibility studied, not built. Recommend against building now — a
value/priority call, not a difficulty finding.** Unlike §4's CLI item, there
is no hard external dependency blocking this: no root-finder needs porting
(the crux fact below), the crate already binds the same `libcubature`/FFTW
this would reuse, and the net new surface is one special function plus a
parameter-shape widening — roughly a Phase-5-sized, 1-2 day slice, not a
multi-week project. It stays parked because (a) nothing in the repo
currently constructs this specific configuration (`prop_stepindex` is
single-mode only; multi-mode `StepIndexMode` only reaches `TransModal`
through `prop_capillary(...; modes=[m1,m2,...])`, an unusual combination —
solid-fibre modes inside a gas-capillary high-level entry point — that no
example or test exercises today), and (b)
`docs/dev/native-port/ARCHITECTURE.md` §6b already classifies exactly this
row of the support matrix as "ordinary numerics that could be ported... but
effort/value rules them out, not feasibility" — this write-up confirms that
classification rather than overturning it. **If a concrete consumer shows
up, this is a clean, boundable yes** — see "If someone picks this up" below
for the exact narrow scope and first steps.

### 1. What the native modal RHS actually needs from a mode (traced, not assumed)

`native_set_modal_params` (`amalthea/src/native.rs:5613`) takes, per mode:
`a` (one shared scalar radius), `unm`, `inv_sqrt_n` (`= 1/√N(m,z=0)`,
precomputed in Julia), `order` (`Int32`), `kind` (`u8` code 0/1/2 for
HE/TE/TM), `phi`, plus one shared `towin`/`kerr_fac`/`nlfac` array — see
`src/RK45.jl:1684-1791`. No Bessel-zero, root-find, or normalization math
crosses the boundary; only the already-solved scalars do.

The per-node hot loop (`modal_pointcalc`, `native.rs:1942`) confirms my
reading exactly: the mode field is synthesized as

```rust
let x = r * ro.modal_unm[m] / ro.modal_a;
let base = jn(ro.modal_order[m], x) * ro.modal_inv_sqrt_n[m];
let (ax, ay) = mode_angle_xy(ro.modal_kind[m], ro.modal_order[m], ro.modal_phi[m], theta);
// Ex = base*ax, Ey = base*ay
```

i.e. **exactly the factored form the task description predicted**:
`J_order(x)·(ax,ay)`, with `(ax,ay)` a closed-form trig function of `θ`
computed in Rust (`mode_angle_xy`, `native.rs:2988`) and `x` a linear
rescale of `r` by two precomputed-in-Julia scalars (`unm`, `a`). `modal_a`
does double duty here: it is *both* the cubature integration bound
(`pcubature_v(fdim, ..., 0.0, self.modal_a, ...)`, `native.rs:1909-1921`)
*and* the mode's own field-argument scale (`r * unm / a` above, and the
`r <= 0 || r >= modal_a → 0` early-out at `native.rs:1949`). For Marcatili
those two roles coincide (the mode is genuinely zero outside the capillary
core, `radius(m,z)`) — this coincidence is Marcatili-specific, not a
general property of `modal_a`, and it is exactly where §2 below breaks.

Normalization (`N(m)`) never calls a Bessel function inside Rust at all —
`MarcatiliMode` overrides the generic `Modes.N` with a closed form
(`Capillary.jl:285-288`), and Julia precomputes `1/√N` once, per mode, at
setup. This is why Phase 5's stated tolerance ceiling is set by `jn`'s
agreement with `SpecialFunctions.besselj`, not by any normalization
integral (MATH.md §3.3, TESTING.md §4).

### 2. How different `StepIndexMode`'s field actually is

`field(m::StepIndexMode, xs; z=0, ω=wlfreq(1030e-9))`
(`src/StepIndexFibre.jl:265-276`) is Snyder & Love 1983 Table 12-3, not a
Marcatili-shaped formula at all:

```julia
f1, f2, u, w, v = f1f2(radius(m,z), ω/c, m.coren(ω,z=z), m.cladn(ω,z=z), neff(m,ω;z=z), m.n)
a1 = (f2 - 1)/2 ; a2 = (f2 + 1)/2
fv = m.parity == :even ? cos(m.n*θ) : sin(m.n*theta)   # NB: `theta`, not `θ` — see below
gv = m.parity == :even ? -sin(m.n*θ) : cos(m.n*θ)
R = r/radius(m,z)
Er, Et = abs(R) <= 1.0 ? corefields(m.n,u,R,a1,a2,fv,gv) : cladfields(m.n,u,w,R,a1,a2,fv,gv)
SVector(Er*cos(θ) - Et*sin(θ), Er*sin(θ) + Et*cos(θ))
```

Contrast with Marcatili (`Capillary.jl:271-283`), which returns `(Ex,Ey)`
directly from one closed-form trig expression with no radial branch and no
polar-to-Cartesian rotation step. `StepIndexMode` differs in every
structural respect that matters for a port:

- **Two radial branches, not one.** `corefields` (Bessel **J**, orders
  `n-1`/`n+1`, evaluated at `u·R`) for `R ≤ 1`; `cladfields` (modified
  Bessel **K**, same orders, at `w·R`) for `R > 1`. Marcatili has no
  cladding branch — the mode is defined (and zero) only inside the core.
- **Wider domain.** `dimlimits(m::StepIndexMode) = (:polar, (0,0),
  (10*radius(m,z), 2π))` (`StepIndexFibre.jl:211`) vs. Marcatili's
  `(radius(m,z), 2π)` (`Capillary.jl:268`) — the field is genuinely
  evaluated (and integrated, for `N`/`Aeff`) out to 10 core radii into the
  cladding, not cut off at the core wall.
- **A real (Er,Eθ)→(Ex,Ey) rotation**, not a single closed-form Cartesian
  expression — `mode_angle_xy` has no equivalent step because Marcatili's
  formula is already Cartesian by construction.
- **A pre-existing bug in the `:odd`-parity branch**, found while reading
  this: line 271 above reads `sin(m.n*theta)` — lowercase `theta`, not the
  destructured `θ` (line 267 binds `θ`, never `theta`) — so `parity=:odd`
  throws `UndefVarError` today, on the Julia oracle, independent of any
  native work. No test in the repo exercises `parity=:odd` (confirmed:
  `test_interface.jl`'s only `StepIndexMode` test and `prop_stepindex`'s
  default both use `parity=:even`). This is a reason to scope any port to
  `:even` only, not a native-specific finding — `:odd` has no working
  oracle to validate against until this upstream bug is fixed.

### 3. The ω-dependence question — traced, and it is the single biggest simplification

`field`'s `ω=wlfreq(1030e-9)` default keyword, plus its own
`# TODO: how do we handle wavelength dependence?` comment, look like an open
design question. Tracing the actual call chain shows **Julia's own modal
pipeline never resolves it** — every caller invokes `field`/`Exy` without
an `ω` argument, so the keyword default is silently load-bearing everywhere,
for every mode type, not just `StepIndexMode`:

- `Modes.field(m::AbstractMode; z=0) = (xs) -> field(m, xs, z=z)`
  (`Modes.jl:48`) — no `ω`.
- `Modes.N` (`Modes.jl:55-66`) calls `field(m, z=z)` → the closure above.
- `Modes.Exy(m, xs; z=0.0) = field(m, xs, z=z) ./ sqrt(N(m, z=z))`
  (`Modes.jl:73`) — no `ω`.
- `Modes.ToSpace.to_space!` (`Modes.jl:491`), the function that actually
  drives `TransModal`'s per-node mode synthesis, calls
  `Exy(ts.ms[i], xs, z=z)` — no `ω`.
- `Modes.Aeff` (`Modes.jl:102-120`) uses `absE(m,z=z)` → `Exy(...)` → same
  chain, no `ω`.

So for **every** mode type reachable through `TransModal` (not a
`StepIndexMode`-specific carve-out), the *field profile itself* is always
evaluated at the fixed reference `ω = wlfreq(1030nm)`. For `MarcatiliMode`
this is invisible because its `field` has no `ω` parameter at all — the
weakly-guided capillary mode shape genuinely doesn't depend on frequency in
that model, only `neff(ω)`/`β(ω)` (evaluated separately, correctly,
per-frequency-bin, in `norm_modal`/the linop) do. `StepIndexMode`'s exact
Snyder & Love solution *does* have a real per-frequency mode shape (`u`,
`w` depend on `ω` through the dispersion relation) — but Julia's own
`TransModal` machinery never asks for it at any `ω` but the hardcoded
default, for every wavelength bin in the grid alike.

**Consequence for the port:** the native side only has to reproduce one
fixed `(u, w, a1, a2)` per mode — solved once, at one frequency, at setup —
not an `ω`-dependent field synthesized per grid bin. This collapses what
looked like "port a dispersion-relation root-finder into the RHS hot loop"
into "pass 4-6 more setup-time scalars per mode," exactly the same shape as
`unm`/`inv_sqrt_n` already are for Marcatili. **This is also a trap, not
just a simplification**: an implementer must reproduce this fixed-`ω`
behavior faithfully, including its physical inexactness, rather than
"fixing" it by making the native path genuinely `ω`-dependent. Native-only
`ω`-dependence would silently diverge from the Julia oracle with no
independent ground truth to justify the change (contrast Phase 7's β1,
which deliberately diverges from Julia but only after validating the new
value against an independent BigFloat truth — MATH.md §3.5, TESTING.md's
"deliberate divergence" tier). If the `# TODO` is ever resolved, it must be
resolved in Julia first — the oracle changes, then the port follows it,
never the other way round.

`Modes.N`/`Modes.Aeff` reach `field` the same no-`ω` way (confirmed above)
— no other call site in the low-level pipeline passes an explicit `ω`.

### 4. What new numerics Rust would actually need

**Reused, not new:** general-order Bessel **J** (`jn`, `diffraction.rs:102`,
Miller downward recurrence, already generalized past `n=1` for Phase E's
`TE`/`TM`/`HE_{n>1}` Marcatili support) covers the core-field Bessel J
evaluations (`besselj(n±1, u·R)`) directly — no new J-side code.

**Genuinely new: modified Bessel K, general integer order.** Confirmed
absent — `grep`ing the whole crate for `besselk`/`BesselK` returns nothing.
The cladding field needs `K_{n-1}`/`K_{n+1}` evaluated at `w·R` for
arbitrary `R ∈ (1, 10]` in the per-node hot loop (this cannot be
precomputed at setup — it varies with the cubature node's `r`, unlike
`u`/`w` themselves). Unlike `J_n` (which needs a *downward*-stable
recurrence because it decays), `K_n` is the numerically easier direction —
it is stably computed via the standard *upward* three-term recurrence
seeded from `K_0`/`K_1` (rational/Chebyshev approximations, e.g.
Abramowitz & Stegun 9.8.5-9.8.8 or Numerical Recipes' `bessk0`/`bessk1`),
since `K_n` grows (not decays) with order and the recurrence amplifies
error in the numerically safe direction. This is precedented work for this
crate — `j0`/`j1` (`diffraction.rs:36-84`) are already from-scratch
rational/power-series approximations of exactly this kind — but it is a
genuinely new special function, not a reuse.

**Can `u`/`w`/`v` (and more) be precomputed in Julia and passed as
constants?** Yes, and further than just `u`/`w`/`v`. Tracing exactly what
`field` computes and which pieces are `r`/`θ`-dependent (must run per
cubature node, in Rust) versus not (fixed once `ω`, `a`, `n`, `z` are fixed
— i.e. everything, given §3's finding that `ω` never varies):

| Quantity | Depends on `(r,θ)`? | Where it can live |
|---|---|---|
| `u`, `w` (dispersion-relation solve, `Roots.find_zeros` inside `findneff`) | no | Julia, precomputed once per mode |
| `v` | no — and unused after `f1f2` returns it (`field` never reads it) | dead value, doesn't need to cross FFI at all |
| `a1 = (f2-1)/2`, `a2 = (f2+1)/2` | no | Julia, precomputed once per mode |
| `j_u = besselj(n, u)`, `k_w = besselk(n, w)` (the *normalizing* denominators in `corefields`/`cladfields`) | no — `u`,`w` are fixed scalars | Julia, precomputed once per mode (Julia already has `SpecialFunctions.besselk` for this — no new Julia-side code needed either) |
| `besselj(n∓1, u·R)`, `besselk(n∓1, w·R)` | **yes** | must run in Rust, per node |
| `fv`, `gv` (parity-dependent `cos`/`sin` of `n·θ`) | yes | Rust, per node (cheap trig, same shape as `mode_angle_xy` already does) |
| the final `(Er,Eθ)→(Ex,Ey)` rotation | yes | Rust, per node |

So **no root-finder ports to Rust at all** — `find_zeros`/`findneff` stays
exactly where the accel-spline machinery already keeps it, in Julia, run
once per mode at setup (this is also why `test_interface.jl`'s
`StepIndexMode` test warns to always pass `accellims` for a real
propagation — that machinery exists for the `neff(ω)` sweep used
elsewhere, e.g. dispersion/`Stats.default`, not for this single-`ω` field
evaluation, but confirms the root-find is a known, already-mitigated setup
cost independent of this design). Per mode, Julia would hand Rust roughly
7 new scalars (`a_core`, `u`, `w`, `a1`, `a2`, `j_u`, `k_w`, a `parity` flag,
`n`) alongside the existing `inv_sqrt_n`/`order`/`kind`/`phi` — the same
shape of addition Marcatili's own `unm`/`inv_sqrt_n` already are, not a
new category of FFI payload.

**The one real structural (not just numerical) change: `modal_a` must
split into two radii.** Per §1, `modal_a` today is simultaneously the
cubature upper bound *and* the mode's field-argument radius
(`x = r·unm/a`) *and* the "return zero beyond here" cutoff — a coincidence
that only holds because Marcatili's mode is exactly zero outside its core.
`StepIndexMode` needs **two** distinct radii: `a_core` (core/clad branch
selector, `R = r/a_core`) and `a_outer = 10·a_core` (the actual cubature
bound, from `dimlimits`). `native_set_modal_params`'s signature and
`modal_pointcalc`'s cutoff/scaling logic both need widening for this — a
real (if small) change to the function's parameter shape and its hot-loop
branch structure, not a guard-only relaxation. Confirmed by grepping every
use of `self.modal_a`/`ro.modal_a` in `native.rs`: exactly the two roles
above, both currently assuming they're the same number.

### 5. Contrast with the sibling task (why this is not the same easy category)

`ZeisbergerMode`/`VincettiMode` (`src/Antiresonant.jl` `@eval` loops at
`:126` and `:381`) **delegate** `field`/`N`/`dimlimits` straight through to
a real, wrapped inner `MarcatiliMode` — their mode profile *is*
Marcatili's, bit-for-bit, by construction, not merely "similar." The
concurrent guard-relaxation for that case is exactly that: relax
`RK45.jl`'s `isa MarcatiliMode` check to also accept the wrapper types
(and unwrap `.m` before extracting `unm`/`kind`/`n`/`ϕ`) — the *existing*
`modal_pointcalc`/`mode_angle_xy`/`x = r·unm/a` Rust code runs completely
unmodified, because the underlying math genuinely is Marcatili's math.

`StepIndexMode` has no such inner `MarcatiliMode` and no delegation — §2
above shows its field formula is structurally unrelated (two radial
branches vs. one, Bessel K vs. no Bessel K at all, a genuine polar-to-
Cartesian rotation vs. a single closed-form Cartesian expression, a
different — and wider — integration domain). **The same guard line cannot
be widened to cover it**; supporting `StepIndexMode` needs a second,
parallel field-synthesis code path in Rust (new params, new per-node
branch, one new special function), not a relaxation of the existing one.
This is the single most useful fact this document settles: do not assume
the Zeisberger/Vincetti guard relaxation (landing concurrently with this
research) incidentally covers `StepIndexMode` too — it does not, and
cannot, by construction.

### 6. Verification strategy, if built

Two-tier equivalence per TESTING.md §4:

- **Single-step tier: expect ~1e-10, Phase-5-shaped.** Node placement is
  bit-identical (same `libcubature` binary, unchanged). The floor is set by
  Rust's new `besselk` agreeing with `SpecialFunctions.besselk` — verify
  this **standalone, before writing any native.rs code**, exactly like
  Phase 5 verified `j0` against `SpecialFunctions.besselj` first
  (MATH.md §3.3: "~1.5e-15 max absolute error... verified... not
  assumed"). Range to check: `w·R` for `R ∈ (1, 10]` at realistic `w`
  (order ~1-10 for typical step-index geometries) — this is the number
  that sets the honest tier to assert, not a guessed 1e-10.
- **Full-solve tier: ~1e-6 floor (fixed `max_dt=min_dt=dt`), per the
  standard discipline** (TESTING.md §3) — sidesteps adaptive-path
  divergence, isolates state accumulation.
- **Non-vacuousness, mirroring Phase 5's own lesson (MATH.md §3.3, "not
  skipped"):** use **two** co-eligible `StepIndexMode`s (different `n`/`m`,
  e.g. `HE11`+`HE12`-analogues) so the `to_space!` sum-over-modes matmul and
  the back-projection matmul are genuinely exercised — a single-mode test
  would leave both matmuls implicitly untested, exactly the Phase 5
  precedent this doc's task framing points at. Independently assert (before
  trusting the Rust-vs-Julia comparison) that inter-mode coupling actually
  moves energy between modes by a margin unmistakably above the FP floor —
  same two-part structure Raman's gate test used (MATH.md §5.3) after a
  first attempt at a Raman gate test passed for the wrong reason (an
  exact-zero effect, not a genuine null result).
- **OPEN, verify at implementation time, does not block this write-up:**
  whether `prop_capillary(...; modes=[stepindex1, stepindex2])` — a solid
  step-index-fibre mode basis inside a gas-capillary high-level entry
  point — actually constructs a well-formed, numerically sensible
  `TransModal` today (density/gas-fill semantics for a mode type that
  isn't gas-based are untested territory). If it doesn't construct
  cleanly, the equivalence test should build `TransModal` directly via the
  low-level API (`Modes.ToSpace`/`NonlinearRHS.TransModal`) instead of
  routing through `prop_capillary`, matching how several existing
  native-port tests already bypass the high-level entry points when the
  high-level combination isn't itself the thing under test.
- Scope the gate test (and any first implementation) to `parity=:even`
  only — §2's `theta`/`θ` typo means `:odd` has no working Julia oracle to
  validate against until that upstream bug is fixed independently of this
  work.

### If someone picks this up

1. **De-risk the one new special function first, in isolation.**
   Implement `besselk(order, x)` in `diffraction.rs` (or a new module) via
   upward recurrence from `K_0`/`K_1` rational approximations; write a
   standalone Rust unit test asserting agreement against
   `SpecialFunctions.besselk` (computed via a tiny throwaway Julia script,
   the same way `j0`/`jn`'s docstrings cite their verification) over
   `x ∈ (0, ~30]` (covers `w·R` for `R` up to 10 at realistic `w`). This
   single step sets the honest single-step tolerance tier for everything
   downstream and is almost all of the genuinely new work.
2. **Widen `native_set_modal_params`'s signature** to carry `a_core` (used
   for the `R = r/a_core` branch selector and the cubature's
   core/clad-transition point, not the outer bound) alongside a new
   `a_outer` (the `pcubature_v`/`hcubature_v_2d` domain bound,
   `= 10·a_core` from `StepIndexMode`'s own `dimlimits`) plus, per mode:
   `u`, `w`, `a1`, `a2`, `j_u`, `k_w`, a `parity` flag. Keep `unm`/existing
   Marcatili fields as-is (still used for co-propagating Marcatili modes in
   a mixed-type `TransModal` — check whether that combination needs
   supporting or can be rejected as out of scope for a first cut).
3. **Add a `StepIndexMode`-specific branch to `modal_pointcalc`** (or a
   parallel function selected by a per-mode type tag) implementing
   `corefields`/`cladfields`'s formula directly from §4's precomputed
   scalars plus the new `besselk`, followed by the `(Er,Eθ)→(Ex,Ey)`
   rotation — this is new Rust code, not a data-only change.
4. **Widen `src/RK45.jl`'s modal guard** (~line 1657) from `all(m -> m isa
   MarcatiliMode, modes)` to also accept `StepIndexMode`, dispatching to
   whichever native params-setup path matches each mode's concrete type;
   compute and pass the 7 new per-mode scalars from Julia (`f1f2`/`uwv`,
   already exported functions in `StepIndexFibre.jl`, reusable as-is for
   this purpose — no new Julia-side math needed, only new glue code calling
   already-existing functions and threading their outputs through the
   FFI call).
5. **Scope the first cut narrowly**, mirroring every prior phase's
   discipline: `parity=:even` only (§2), constant (non-tapered) radius
   only, Kerr-only to start (defer Raman — `modal_pointcalc`'s existing
   Raman block is written against the Marcatili scalar-field assumption
   and would need its own check for whether it generalizes unchanged),
   RealGrid only, `full=false` (radial-only cubature) only. Confirm the
   `prop_capillary(...; modes=[...])` reachability question (§6's OPEN
   item) before writing the equivalence test.
6. Follow `AGENTS.md` §3/§4 exactly: build the Rust lib, wire Julia, write
   the two-tier `@testitem tags=[:rust]` test, run the `rust` group plus a
   check for regressions in `sim-multimode`, and append a `PORT_LOG.md`
   entry recording both achieved tolerances and every gotcha found —
   in particular the `modal_a` split and the `theta`/`θ` typo, since a
   future reader will hit both again if they aren't written down here.

---

## 6. Beyond-Luna math options (BACKLOG Phase J item 6)

Feasibility/design pass over the three items `native-port/MATH.md` §8 +
`SUGGESTIONS.md` items 15-17 grouped as "beyond-Luna math options": direct
DP5(4) embedded-error coefficients, direct PPT evaluation replacing the
spline LUT, and short-kernel overlap-save Raman convolution. Each is
individually assessed below against the β1 bar (`BETA1_ANALYTIC.md`): a
divergence from the Julia oracle is acceptable only with a derivation, a
ground truth that isn't Julia itself, a quantified error against that
ground truth, and a stated validation blast radius. This is a
feasibility/design pass, not an implementation — no Rust or Julia source
was changed.

| Item | Candidate | Verdict |
|---|---|---|
| 6.1 | Direct DP5(4) embedded-error coefficients (SUGGESTIONS #15) | **Recommend against — already implemented on both sides.** The backlog's own premise (`errest` computed as a runtime `y5−y4` cancellation) does not match the current code. Same class of finding as S5 item 3 ("backlog premise wrong"), not a genuine open design question. |
| 6.2 | Direct PPT evaluation replacing the spline LUT (SUGGESTIONS #16) | **Recommend against, as specified.** The accuracy gain is real but physically immaterial (LUT error is already far below anything the physics cares about); the cost is not — plasma is already a measured 24.5-40.5% of `rhs_radial` time on the *cheap* path, and direct evaluation is 1-3 orders of magnitude more expensive per call, with an unbounded tail (BigFloat adaptive quadrature) that cannot run in a hot loop at all. The stated secondary motivation (structurally eliminating an out-of-range segfault) is also moot — that segfault was already fixed by a cheaper, already-shipped guard. |
| 6.3 | Short-kernel overlap-save Raman convolution (SUGGESTIONS #17) | **Recommend, narrowly scoped.** Not full block-based overlap-save — just shortening the zero-pad to the truncated kernel length. Complementary to, not superseded by or exclusive with, item J.3's r2c/c2r halving (BACKLOG open remainder 2); the two multiply. Scope is the FFT-convolution Raman paths only (native `:SiO2` kernel + Julia `RamanPolarEnv`); the default carrier-field path (`raman.rs`'s ADE solver) is already O(N) and out of scope. Unlike the other two items and unlike β1, this one need not diverge from the Julia oracle at all if the cutoff is chosen below the f64 noise floor — the ground truth is the existing full-grid convolution, not a "more correct" reformulation. |

---

### 6.1 Direct DP5(4) embedded-error coefficients

**Verdict: recommend against — the code already does this.**

#### What the backlog item asserts

`MATH.md` §8.1 and `SUGGESTIONS.md` #15 both describe the problem as: "Luna
computes the DP5(4) embedded error as the difference of the 5th- and
4th-order solutions — a near-total cancellation," and propose replacing it
with "the estimate directly as `err = Σᵢ eᵢ·kᵢ` with precomputed
`eᵢ = b5ᵢ − b4ᵢ` coefficients (exact rationals, no runtime cancellation)."

#### What the code actually does

Neither side forms two full solution vectors and subtracts them.

- **Rust (`amalthea/src/native.rs`):** `DP_ERREST` (`native.rs:5889-5897`) is
  an array of *directly-typed rational literals* — `-71.0/57600.0`,
  `71.0/16695.0`, `-71.0/1920.0`, `17253.0/339200.0`, `-22.0/525.0`,
  `1.0/40.0` — computed once at compile time. The error estimate
  (`native.rs:4668-4710`, inside `native_step`) is `Σᵢ eᵢ·kᵢ` directly:
  `e0*k0 + e1*k1 + ... + e6*k6`. There is **no `DP_B4` array anywhere in
  `native.rs`** (confirmed by grep) — the accepted solution `yn` is formed
  only from `DP_B5` (the 5th-order weights, `native.rs:4622-4665`); a
  4th-order solution is never computed, let alone subtracted from the
  5th-order one.
- **Julia (`src/dopri.jl:14-18`):** `b5` and `b4` are each defined as
  `Float64`-rounded rational literals, and `const errest = b5 .- b4` is
  computed **once at module load** (a `const`, not recomputed per step).
  `RK45.jl:288-291`'s `step!` then accumulates `s.yerr += s.dt*s.ks[ii]*errest[ii]`
  — the same `Σᵢ eᵢ·kᵢ` direct-dot-product form, not a per-step `y5−y4`
  subtraction of full state vectors. `PreconStepper` (`RK45.jl:239`) reuses
  the same generic `step!`.

So both backends already implement "the standard, numerically superior
formulation" the item proposes. There is no runtime `y5−y4` cancellation to
remove from either hot loop.

#### The real (much smaller) discrepancy, and why fixing it doesn't help

The one genuine gap: Julia's `errest[i]` is computed via
`Float64(b5ᵢ_rational) − Float64(b4ᵢ_rational)` (double-rounded), while
Rust's `DP_ERREST[i]` is `Float64(eᵢ_rational)` (singly-rounded, from the
already-reduced fraction). Checked numerically (Python, reproducing both
computations bit-for-bit):

| Coefficient | Julia (`b5−b4`) | Rust (`DP_ERREST`, direct) | Match |
|---|---|---|---|
| E1 | −0.0012326388888888873 | −0.0012326388888888888 | 1 ULP off |
| E3 | 0.004252770290506136 | 0.0042527702905061394 | 1 ULP off |
| E4 | −0.036979166666666674 | −0.03697916666666667 | 1 ULP off |
| E5 | 0.05086379716981132 | 0.05086379716981132 | exact |
| E6 | −0.04190476190476192 | −0.0419047619047619 | 1 ULP off |
| E7 | 0.025 | 0.025 | exact |

4 of 6 nonzero coefficients differ by exactly 1 ULP (~1e-15 to ~1e-17
relative to the coefficient's own magnitude — the same order as the
FFTW/BLAS summation-order noise already documented as unavoidable, TESTING.md
§3). This is a genuine, tiny, **one-time** (not per-step) discrepancy
between the two backends' constants — but it is not the mechanism CLAUDE.md's
"Phase 2 gotcha" and TESTING.md §3 describe. That mechanism is: the DP5(4)
embedded pair is constructed so its error estimate is, by design,
mathematically small whenever the true local truncation error is small —
which is exactly the regime a weakly-nonlinear early propagation step is
in. When the *true* `err` sits near the ~1e-15 floor, any independent
~1e-15-relative noise in the `kᵢ` themselves (from Julia's BLAS/FFTW vs.
Rust's Rayon-sequential summation order — an unrelated, unavoidable source,
same TESTING.md §3) dominates the signal, producing the documented ~20%
relative swings and the resulting adaptive-path divergence. That mechanism
lives entirely in the `kᵢ` (the RHS evaluations), not in how the fixed
`eᵢ` constants are computed or represented — and no reformulation of the
coefficients touches it.

**Concretely: aligning Julia's `errest` to Rust's directly-typed rational
literals** (a one-line, free, zero-runtime-cost change — replace
`errest = b5 .- b4` with the reduced-fraction literals, mirroring
`DP_ERREST`) **would close the ~1e-15-relative constant gap without
addressing the actual, much larger, and irreducible source of divergence.**
It is available and harmless, but should not be sold as "dissolving the
fixed-step test discipline" — it doesn't, because the dominant noise
source it leaves untouched (cross-backend `kᵢ` summation-order noise,
amplified by an intrinsically tiny true `err` for smooth problems) is
unrelated to coefficient precision.

#### Acceptance test / ground truth, if the micro-fix were made anyway

Not a "more correct than Julia" claim in the β1 sense — both constants are
already accurate to their own last bit; this is purely a
bit-for-bit-alignment question. Ground truth: the exact rational value of
each `eᵢ`, computed once and compared to both backends' current constants
(the table above already does this). No BigFloat/analytic derivation
needed — the "ground truth" is exact rational arithmetic, trivially exact.

#### Validation blast radius

Because the micro-fix changes a controller input (`err`) used by every
adaptive (non-fixed-`dt`) test, it is technically downstream of the entire
suite (Phase 8's gate: 46590 pass / 12 broken, 46602 total,
`TESTING.md` §4). In practice: the perturbation (~1e-15 to ~1e-17 per
coefficient) is smaller than the FFTW/BLAS run-to-run noise floor
(~2e-8, TESTING.md §3) already tolerated by every full-solve test, so no
currently-passing assertion is expected to flip. The honest statement is
"technically all ~46.6k assertions are downstream of the step controller;
practically none should move, because the change is smaller than noise
already priced into every tolerance" — not "zero blast radius." Given the
fix buys nothing for the actual documented failure mode, this doesn't
clear the bar for spending a validation pass on it.

---

### 6.2 Direct PPT evaluation replacing the spline LUT

**Verdict: recommend against, as specified** (replacing the LUT with a
per-call direct evaluation in the hot RHS loop). Below is the accuracy
case, the cost case, and the one narrower variant that might still be
worth a bench — restated as an open question with an explicit measurement
gate, not a recommendation.

#### What's being proposed

`amalthea/src/ionization.rs`'s `PptIonizationRate` evaluates a natural
cubic spline fit to `ln(rate)` over `|E|` (`CubicSplineLUT`, built from
`N=2^16` knots by default — `src/Ionisation.jl:533`'s
`IonRatePPTAccel(...; N=2^16, ...)` — with a strict lower-bound cutoff
`e_min` and a clamp-not-error policy above `e_max`,
`ionization.rs:242-289`). Julia's own `IonRatePPTAccel` (`Ionisation.jl:480-518`)
is the same idea one level up — a `Maths.CSpline` fit to the same
log-rate data, cached to disk (`~/.luna/pptcache`, `Ionisation.jl:528-570`)
so the expensive fit is amortized across process launches. The item
proposes evaluating the true PPT series (`IonRatePPT`, `Ionisation.jl:367-422`)
directly on every call, on both sides, removing interpolation error
entirely.

#### Accuracy argument: real, but immaterial

Evaluating the PPT formula directly is unambiguously "more correct" than
interpolating a spline through it, in the same direction as β1 (Rust
computing something exact instead of an approximation). But unlike β1,
where Julia's FD error against the true derivative was ~1e-12 and
accumulated *coherently* over a broadband, long-propagation to a physically
meaningful ~1e-4–1e-7 full-solve effect, the LUT here is fit at
`N=2^16` = 65536 knots in log-space over the field-strength axis. A
natural cubic spline at that knot density interpolates a smooth,
monotonic-ish function (`IonRatePPT` output) to an error far below any
physical tolerance the simulation carries elsewhere (this is exactly why
`TESTING.md` §2 classifies "LUT/spline interpolation" as its own
~1e-8-tier bucket, already tighter than the ~1e-6 floor tier most
full-solve tests live at). There is no evidence, measured or structural, of
a coherent, propagation-scale error mechanism here the way there was for
β1 — the interpolation error is uncorrelated, essentially white noise
across the field-strength axis, at a magnitude below the noise floor the
suite already tolerates. **"More correct" is true; "buys anything" is not
demonstrated,** which is the opposite of β1's case.

#### Cost argument: this is where it fails

`amalthea/src/ionization.rs`'s `PptIonizationRate::rate` today is a binary
search (`~log2(65536)≈16` comparisons) plus a cubic evaluation
(`a + dx*(b + dx*(c + dx*d))`, ~6 flops) — call it O(10ns) per grid point,
per stage, and it vectorizes/parallelizes trivially (`rate_vector`,
`ionization.rs:291-306`, with an optional GPU path).

The PPT formula it replaces (`Ionisation.jl:367-422`) is materially more
expensive, and in its default configuration has an **unbounded** tail:

- `sum_integral` defaults to `false` (`Ionisation.jl:347-348`), meaning the
  rate is not the closed-form integral approximation but an explicit
  series summed via `Maths.converge_series` to `rtol=1e-6`
  (`Ionisation.jl:409-413`) — an iteration count that depends on the
  Keldysh parameter `γ` and isn't bounded a priori.
- Each series term calls `φ(m, x)` (`Ionisation.jl:427-461`), which for
  `m=0` is a Dawson-function evaluation, for `m≠0, x≤26` is a confluent
  hypergeometric function (`pFq`) times gamma functions, and **for
  `m≠0, x>26` falls back to adaptive quadrature (`hquadrature`) over a
  `BigFloat` integrand** (`Ionisation.jl:453-459`). That last branch is
  categorically impossible to run in a Rust hot loop without either
  reimplementing arbitrary-precision arithmetic and adaptive quadrature in
  Rust (a large, separate porting project with no existing primitive to
  reuse — `cubature.rs`'s `libcubature` binding is `f64`-only) or accepting
  silent divergence from Julia's answer exactly in the regime it was added
  to handle.
- Even the cheaper `sum_integral=true` closed-form variant
  (`Ionisation.jl:407`, `s = sqrt(π)*factorial(mabs)*β^mabs/...`) requires
  several `pow`/`exp`/`asinh`/`sqrt`/`gamma`-adjacent evaluations per call
  — call it conservatively 20-30 transcendental-function-class operations,
  each tens of ns, i.e. **~0.5-1.5µs per call**, a rough 50-150x per-call
  slowdown versus the spline even before counting the accuracy cost of
  using a cheaper formula than Julia's own default oracle uses (see below).

The plasma path is not a cold path: `apply_plasma_radial`/the mode-averaged
plasma step calls the ionisation rate once per time-domain grid point,
every RHS stage, every accepted-or-rejected step. Measured
(`docs/dev/BACKLOG.md`, S2 Phase 2 item 2, `Instant`-based profiling,
add/measure/revert discipline): **plasma is 24.5-40.5% of `rhs_radial`
wall time today, on the cheap spline path**
(N=32 r-points: 40.5%; N=128 r-points: 24.5%). A 50-150x per-call slowdown
on a term that is already a quarter-to-two-fifths of RHS time would not
shave a few percent off a solve — it would make plasma-enabled propagation
dominated by ionisation-rate evaluation, likely a multi-x regression on
total wall time for exactly the configs (strong-field, plasma-heavy) where
the native port's performance case matters most. And that estimate is for
the *cheap* variant; the *default* (`sum_integral=false`, unbounded series
+ occasional BigFloat quadrature) is not boundable at all — it cannot be
given a worst-case per-call cost, which alone disqualifies it from a
allocation-free, real-time-budgeted hot loop.

There is also a version-mismatch problem baked into "replace the LUT with
direct evaluation": if the *cheap* (`sum_integral=true`) formula is used
for speed, that is no longer the same ground truth the LUT was fit to
(which uses Julia's default `sum_integral=false`) — so switching would
trade a well-characterized, sub-noise-floor interpolation error for an
uncharacterized, physically real "neglects the multiphoton thresholds"
approximation error (the docstring's own words, `Ionisation.jl:285-286`).
That is not obviously a net accuracy win, and is not the β1 pattern
("more correct, unambiguously") at all.

#### The stated secondary motivation is already moot

`SUGGESTIONS.md` #16 and `MATH.md` §8.2 both cite "structurally eliminates
the out-of-range segfault (BACKLOG Phase J item 2)" as a benefit. That
segfault is **already resolved** (`ARCHIVE.md`, Phase J archived item 2,
"Resolved 2026-07-09"): the root cause was a NaN/±inf comparison gap (IEEE
754 makes every comparison with NaN false, silently falling through both
the `< e_min`/`> e_max` guards), fixed by an explicit `is_finite()` check
in both `CubicSplineLUT::evaluate` and `PptIonizationRate::rate`
(`ionization.rs:242-289`), backed by 11 new Rust unit tests
(`ionization.rs:449-591`, including `ppt_non_finite_inputs_never_panic`,
`ppt_extreme_sweep_never_panics`) sweeping ±1e10× the fitted range in both
signs without a panic. The LUT already has no fitted-range-based failure
mode left to eliminate; a direct-eval rewrite would not be closing an open
safety gap, it would be redundant with a cheaper fix already shipped.

#### Acceptance test / ground truth, if pursued anyway

The direct sum against a `BigFloat` evaluation of the same series (as
`SUGGESTIONS.md` #16 itself proposes) is the right ground truth *for the
accuracy question* — but accuracy was never the blocker here. Cost is. Any
future revisit should lead with a Criterion microbenchmark of
`IonRatePPT`-direct (both `sum_integral` settings) against
`PptIonizationRate::rate`, at the actual per-RHS-call rate (once per
`n_time × n_r` point, or `n_time` for mode-averaged), gated the same way
S5 item 1's mixed-precision spike was: **measure first, in a
timeboxed/reverted benchmark, before writing any production code**, against
an explicit threshold (e.g. "total plasma-path time must not exceed
current + X% of a representative solve's wall time" — pick X based on
how much of the native port's advertised speedup budget plasma is allowed
to consume, not an arbitrary number invented here).

#### Validation blast radius

If measurement somehow cleared the cost bar (unlikely given the analysis
above, and only conceivable for the `sum_integral=true` variant with a
hard cap disallowing the `m≠0,x>26` BigFloat branch — i.e., a genuinely
different, narrower formula than Julia's default, which would itself need
its own accuracy sign-off before being called a drop-in replacement): every
plasma-enabled test in `sim-propagation`, `sim-multimode`, and the `rust`
group's ionisation-specific tests (`test_ionisation_rust.jl`) would need
re-validation at whatever tier the new method/ground-truth pairing
justifies — likely the "method/spline" ~1e-8 tier stays, since both old
and new paths interpolate or approximate the same underlying physics, just
via a different intermediate representation.

---

### 6.3 Short-kernel overlap-save Raman convolution

**Verdict: recommend, narrowly scoped to pad-shortening — not full
block-based overlap-save.**

#### Scope: which Raman kernel this touches

There are two distinct native Raman kernels (`CLAUDE.md`'s kernel-wiring
table, `MATH.md` §5.3, §8.4):

1. The **default carrier-field path** (`raman.rs`'s
   `TimeDomainRamanSolver`, an explicit exponential-integrator ADE
   solve) — already O(N), allocation-free, no FFT at all. **Out of scope**
   for this item; there is no convolution to shorten.
2. The **intermediate-broadening `:SiO2` path** (`native.rs`'s
   `raman_fft_e2`/`raman_fft_hw`/`raman_fft_plan`, `native.rs:301-325,
   1490-1523`, wired via `native_set_raman_fft_params`) and its Julia
   counterpart `RamanPolarEnv` (`Nonlinear.jl:327` onward) — both use a
   **full c2c FFT over a zero-padded double-length grid**
   (`2·n_time_over`), every RHS call. This is the one the item, and BACKLOG
   open remainder "Phase J.3," both target.

#### Why the grid is doubled today, and why truncation is safe in principle

Luna's own comment (`Nonlinear.jl:443-447`) explains the doubling: it gives
an "accurate full convolution between the full field grid and the full
extent of the impulse response function, with no truncation of the
response function... this is safe, until we come up with [something more
efficient]." For the SiO2 intermediate-broadening (Hollenbeck & Cantrell
Gaussian-damped) response, `h ≈ 0` beyond roughly 100 fs on a
multi-picosecond simulation grid — already noted as the reason a stale-tail
bug in the padded buffer was numerically invisible in every existing test
(`ARCHIVE.md`, Phase I archived item 1). That is the concrete basis for
truncating the kernel: measure the index `M` where `|h|` drops below (say)
a few ULPs of its peak in `f64`, and use a pad of `~n_time_over + M`
instead of `2·n_time_over`.

#### Decomposing the actual speedup — the "overlap-save" name overclaims

The naive framing ("kernel is a small fraction of the grid → convolution
gets proportionally cheaper") is wrong because FFT cost is `O(N log N)`,
not `O(N)`, and there are two structurally different ways to exploit a
short kernel:

- **Pad-shortening only (recommended here):** keep a single FFT per RHS
  call, but size the zero-pad to `n_time_over + M` (rounded to an
  FFT-friendly length) instead of `2·n_time_over`. This is a **clean,
  low-risk ~2x** in the typical case where `M ≈ n_time_over` shrinks the
  transform length by roughly half — no algorithmic change, same
  convolution theorem, same single resident FFTW plan (just shorter),
  same code shape as the existing `raman_fft_plan`. It **stacks
  multiplicatively** with item J.3's r2c/c2r halving (also ~2x, from using
  a real-valued transform instead of c2c on data that's real in both
  domains) — combined, roughly **~4x** on this specific kernel.
- **Full block-based overlap-save** (many short FFTs, one per
  `M`-sized input block, standard fast-convolution machinery): adds only
  the *additional* `log(N)/log(M)` factor on top of the above — for
  realistic `M/N` ratios here (kernel maybe 5-10% of the padded grid),
  that's roughly **another 1.3-1.5x**, at materially higher implementation
  complexity (block bookkeeping, save/discard logic, edge handling at
  segment boundaries) and more surface area for a correctness bug.

**The bulk of the achievable win is pad-shortening alone.** Full
block-based overlap-save is a real but much smaller additional
multiplier for a lot more code — worth deferring until pad-shortening
plus J.3's r2c/c2r halving are landed and measured, not worth bundling into
the same change.

#### Relationship to item J.3 (r2c/c2r halving)

**Complementary, not superseded by or exclusive with it.** J.3 attacks the
*representation* of the transform (complex-valued FFT on real-valued data →
real-valued FFT, same length, ~2x from skipping the redundant
conjugate-symmetric half). Pad-shortening attacks the *length* of the
transform (shorter effective kernel → shorter pad → smaller `N` in the
`O(N log N)` cost, another ~2x). Neither subsumes the other; a combined
r2c/c2r-on-a-shortened-pad implementation captures both multipliers
(~4x) with one coordinated change, and is a natural single unit of work
alongside — or immediately following — whichever agent lands J.3, so the
two don't produce a merge conflict on the same `native.rs` FFT-setup code.

#### Why this one, uniquely among the three, need not diverge from the oracle

Items 6.1 and 6.2 either found nothing to change (6.1) or would need a
genuine, physically real accuracy/cost tradeoff against the Julia oracle
(6.2). Item 6.3 is different in kind: **if the truncation cutoff `M` is
chosen so that everything discarded is below the f64 noise floor relative
to the kept kernel's peak, the truncated convolution and the full-grid
convolution are the same computation to within noise — not a "more
correct than Julia" divergence in the β1 sense, but a "provably
equivalent to the existing native SiO2 kernel and to `RamanPolarEnv`,
just computed over a shorter buffer" claim.** The ground truth is the
existing full-double-length-grid convolution (Rust-vs-Rust, or
Rust-vs-Julia's `RamanPolarEnv`), not a BigFloat/analytic derivation — a
materially easier bar than β1's, precisely because nothing here is
claimed to be *more accurate than Luna's physics*, only *cheaper to
compute the same physics*.

Two implementation choices follow from this:

- **Truncate only where measured, per-material, at setup** — not a fixed
  cutoff. `MATH.md` §8.5's own framing ("checked at setup, falling back to
  the full double grid otherwise") is right: compute `M` from the actual
  `h` array being used (SiO2's specific Gaussian-damping constants,
  already resident), not a hardcoded 100fs guess, so the mechanism is
  correct for any future material with a longer-tailed response.
- **If the cutoff is chosen conservatively enough that truncation error is
  below f64 noise, both `RamanPolarEnv` (Julia) and the native `:SiO2`
  kernel can adopt the shorter pad in the same commit**, preserving exact
  bit-parity between them (same discipline `SUGGESTIONS.md` #15 correctly
  insists on for the DP5(4) coefficients: land both sides together). This
  keeps the change inside the *existing* SiO2 equivalence tier
  (`~1e-6`-ish, FFT-method class, `TESTING.md` §2) rather than opening a
  new "deliberate divergence" tier.

**Hard boundary (already correctly stated in `MATH.md` §8.5, repeated
here for completeness):** do not go further and fit a recursive/IIR filter
to the Gaussian-damped response to avoid the FFT altogether — that is the
multi-SDO approximation trap `ARCHIVE.md`'s Phase I item 2 explicitly
rules out ("prefer analytic over LUT/fit-the-noise").

#### Acceptance test / ground truth

Rust-vs-Rust (truncated-pad kernel vs. the existing full-`2·n_time_over`
kernel, same inputs, single RHS evaluation) at whatever tier the measured
truncation error justifies — expect ~bitwise-to-reassociation
(`< 1e-13`) if the cutoff is chosen conservatively per the discussion
above, since the discarded tail is defined to be below the f64 noise
floor. If Julia's `RamanPolarEnv` is changed in the same commit, the
existing native-vs-Julia SiO2 full-solve test keeps its current tolerance
tier unchanged (fixed-step discipline per `TESTING.md` §3, since any
transform-length change alters summation order).

#### What to measure before implementing

1. **The effective kernel length `M`** for the SiO2 response actually used
   in the test suite and in representative user configs: the smallest `M`
   such that `max(|h[M:]|) < ε · max(|h|)` for `ε` a few ULPs of `f64`
   (e.g. `1e-15`-`1e-14` relative). This single number decides both
   pad-shortening's multiplier (`M / n_time_over`, roughly) and whether
   full block-based overlap-save's extra `log(N)/log(M)` factor is worth
   its complexity — if `M` turns out close to `n_time_over` (i.e. the
   response barely decays on the grid), there is little to gain and this
   item should be shelved for that config.
2. **A wall-clock benchmark of the shortened-pad FFT vs. the current
   `2·n_time_over` FFT**, isolated (Criterion, add/measure/revert
   discipline per S1.6/S5.1), to confirm the ~2x (or combined ~4x with
   r2c/c2r) actually materializes on this codebase's FFTW binding and
   plan-reuse pattern, before committing to a production change.

#### Validation blast radius

Narrow by construction: only the `:SiO2`/`RamanPolarEnv` path is touched
(`prop_gnlse(...; ramanmodel=:SiO2)`'s reachable configs, per `native.rs`'s
own comment at `native.rs:5516-5519`). The default carrier-field SDO
path, every Kerr-only/plasma-only config, and every other geometry are
untouched. If both Julia and Rust change together (recommended), the
existing SiO2-specific tests are the entire blast radius — no
suite-wide adaptive-step-path perturbation the way 6.1's micro-fix would
cause, because the *result* is designed to be numerically unchanged, only
the FFT length differs.
