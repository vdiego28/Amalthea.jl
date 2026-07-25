# Inbox note — BACKLOG resume-queue item 3: repair the 7 known-broken examples

Agent: worktree `agent-ac6e34a3affa51929`. Date: 2026-07-25.
Scope: the seven `examples/low_level_interface/` files flagged broken by
`docs/dev/native-port/portlog-inbox/hygiene.md` (2026-07-22 audit) plus
regression coverage in `test/test_examples_smoke.jl`. Exclusive file zone
for this task: the seven example files, `test/test_examples_smoke.jl`, this
new inbox file. Per instructions, `PORT_LOG.md`, `BACKLOG.md`, `ARCHIVE.md`,
`hygiene.md`, `SUGGESTIONS.md` were NOT edited.

## 1. What was verified before fixing

Re-audited all 44 `.jl` files under `examples/low_level_interface/` (not
just the 7 named) for the two documented bug classes:
- `grep -rn "norm_modal"` — only the 3 files hygiene.md already named
  (`full_modal/basic_modal_full.jl`, `full_modal/basic_modal_full_bothpolarisations.jl`,
  `polarisation/elliptical_env.jl`) call it. No other file misuses it.
- `grep -rn "linop"` across all 44, checked first-use vs. assignment line
  order for every file that mentions `linop` — only the 6 files hygiene.md
  named use it before assignment. Every other file (37 total, including all
  8 maintained smoke examples) assigns `linop` before its first use.

So the backlog's file list for these two classes is exactly right, no
additions or removals needed there. However, the audit's classification of
*why* each file was broken undersold three files — see §3.

## 2. Fixes applied (the two documented bug classes)

**Class 1 — `linop` used before assignment** (moved the
`linop = LinearOps.make_const_linop(...)` line to immediately before its
first use in `Stats.default(...)`, matching every maintained smoke example,
e.g. `examples/low_level_interface/basic_modal.jl:35-37`):
- `full_modal/basic_modal_full.jl:38-40`
- `full_modal/basic_modal_full_bothpolarisations.jl:38-40`
- `polarisation/modal_vector_plasma.jl:37-39`
- `polarisation/modal_nonvector_plasma.jl:37-39`
- `polarisation/modal_vector_plasma_CP.jl:39-41`
- `polarisation/modal_vector_plasma_45deg.jl:39-41`

**Class 2 — `NonlinearRHS.norm_modal(grid.ω)` instead of
`NonlinearRHS.norm_modal(grid)`** (`norm_modal(grid; shock=true)` reads
`grid.referenceλ` internally — `src/NonlinearRHS.jl:476-481`):
- `full_modal/basic_modal_full.jl:21`
- `full_modal/basic_modal_full_bothpolarisations.jl:21`
- `polarisation/elliptical_env.jl:20`

## 3. Bugs the original audit's harness never reached (found while
verifying each file actually *runs*, not just that the documented line
matches — the task instructions required verifying claims against the
file, and running each file end-to-end after the documented fix surfaced
these)

The hygiene.md audit's own harness stopped at the *first* thrown error in
each file. Fixing only the documented bug let execution proceed further in
three files and immediately hit a **second, previously-undiscovered**
error in each — proven by running the pristine (`git show HEAD:...`)
original file through the harness and confirming which error fires first:

- **`polarisation/modal_vector_plasma_CP.jl`**: the *actual* first error on
  the unfixed file is not the linop bug at all — it's
  `Fields.GaussField(...; ϕ=π/2)` at line 34, which fires during `inputs =
  (...)` construction, textually and temporally *before* the file ever
  reaches the `linop`/`Stats.default` line. `PulseField.ϕ::Vector{Float64}`
  (`src/Fields.jl:70,77`) is a vector of spectral-phase coefficients
  (CEP, GD, GDD, TOD, ...), not a scalar — confirmed by
  `test/test_fields.jl:266` etc.'s `ϕ=[ϕCEO]` convention, and by
  `test/test_interface.jl:73,76,241`'s `ϕ=[0, 0, φ2]` /
  `ϕ=[0, 100e-15]`. Fixed: `ϕ=π/2` → `ϕ=[π/2]` (line 34) — this is what
  "circular polarisation via a 90° phase offset between two modes" means
  under the real API. So this file's backlog classification under "Finding
  2" (linop bug) is technically true (the linop bug is *also* present and
  was fixed) but not why it fails at runtime — the ϕ bug fires first.
- **`polarisation/elliptical_env.jl`**: fixing the documented
  `norm_modal(grid.ω)` → `norm_modal(grid)` (line 20) let execution reach
  three further, independent bugs, each confirmed by re-running after each
  incremental fix:
  1. `gausspulse(t)` (line 22-25) used an undefined global `τ` (line 23)
     instead of `τfwhm` — `UndefVarError`. Every other τ-reference in the
     same file, and the sibling `polarisation/elliptical.jl:27`, use
     `τfwhm`. Fixed: `fwhm=τ` → `fwhm=τfwhm`.
  2. Same line: `Maths.gauss(t, fwhm=τfwhm)` called without broadcasting —
     `Maths.gauss(x, σ; x0=0, power=2) = exp(-1/2 * ((x-x0)/σ)^power)`
     (`src/Maths.jl:40`) is scalar-only; `t` is a `Vector{Float64}`, so
     `t - x0` (no broadcast) throws `MethodError: no method matching
     -(::Vector{Float64}, ::Int64)`. The sibling `elliptical.jl:27` uses
     `Maths.gauss.(t; fwhm=τfwhm)` (broadcast dot). Fixed: added the dot —
     `Maths.gauss.(t, fwhm=τfwhm)`.
  3. `FFTW.fft(Et, 1)` used at (then-)line 33, but `import FFTW` only
     appeared after `Amalthea.run` in the plotting section (originally
     line 63) — `UndefVarError: FFTW not defined`. Sibling
     `elliptical.jl:4` has `import FFTW` up top. Fixed: added `import
     FFTW` to the top import block (line 2).
  4. `Amalthea.setup(grid, densityfun, normfun, responses, inputs, modes,
     :xy; full=false)` (then-lines 52-53) passes `normfun` as an errant
     extra **positional** argument. The real modal-EnvGrid `setup` overload
     (`src/Amalthea.jl:220-245`) is
     `setup(grid, densityfun, responses, inputs, modes, components; full,
     norm!=NonlinearRHS.norm_modal(grid), ...)` — `norm!` is a keyword
     with a default **identical** to what `normfun` computes, and there is
     no positional slot for it. Confirmed against the maintained smoke
     example `basic_modal_env.jl:34`
     (`Amalthea.setup(grid, densityfun, responses, inputs, modes, :y;
     full=false)` — 6 positional args, no normfun). This means `normfun`
     is genuinely dead code once removed from the call, same as it already
     was (unused) in `basic_modal_full.jl` and
     `basic_modal_full_bothpolarisations.jl` after their class-2 fix —
     consistent across all three files. Fixed: dropped `normfun,` from the
     `Amalthea.setup` call (now 6 positional args, matching
     `basic_modal_env.jl`'s modal-EnvGrid convention).
  After all four fixes, `elliptical_env.jl` runs to completion under the
  smoke harness (confirmed).
  This file was **not** flagged as fixed for a third/fourth/fifth bug class
  in the backlog — recording it here per "the backlog is the lead's note,
  not ground truth."

**Consulted the advisor tool mid-task** on whether to expand scope beyond
the two documented classes; it recommended fixing the τ/ϕ API-drift bugs
(single-token, matching sibling-file/test-suite convention) and NOT
chasing the deeper `TransModal` bug below, which is a library defect, not
an example typo.

## 4. Left broken (deliberately, out of scope)

**`full_modal/basic_modal_full_bothpolarisations.jl`** still throws after
both documented-class fixes:
```
DimensionMismatch: cannot broadcast array to have fewer non-singleton dimensions
```
Stack: `Cubature.hcubature_v` → `TransModal`'s `(t::TransModal)(nl, Eω, z)`
(`src/NonlinearRHS.jl:453`) → `RK45.PreconStepper`'s constructor
(`src/RK45.jl:269`, `fbar!(k1, y0, t, t)` — the initial FSAL RHS
evaluation, called with `t1=t2=t`, i.e. **before any step is taken**). This
proves the throw is independent of `flength`/`SMOKE_LENGTH` — it is not a
smoke-harness artifact, it reproduces at the file's original 15 cm length
too (same code path, same inputs at z=0). This is `full=true` modal
propagation with 2 polarisations (`:xy`) and `PlasmaCumtrapz` — a
combination that appears to have never worked. This is a library-level
`TransModal`/plasma-response shape bug, not an example-file typo, and is
out of the "minimal, idiomatic" fix mandate for this task. Not fixed.
Documented in `test/test_examples_smoke.jl`'s header comment and excluded
from the smoke set. Per this task's doc-ownership restriction, not added to
`BACKLOG.md` — flagging here for the lead to file if they want it tracked.

## 5. Regression coverage added

Extended `test/test_examples_smoke.jl`'s `examples` list (now 10, was 8):
- `full_modal/basic_modal_full.jl` — covers **both** bug classes.
- `polarisation/modal_nonvector_plasma.jl` — covers class 1 only (chosen
  over the other three class-1-only polarisation files because it's the
  simplest: single mode, single `GaussField` input, no vector-CP/45°-input
  complexity).

**Harness addition**: `polarisation/modal_nonvector_plasma.jl` (like all
four `modal_*_plasma*.jl` files) uses `Output.HDF5Output("<hardcoded
name>.h5", ...)`, which would otherwise write a stray file into the test
process's CWD on every smoke run. Added `_rewrite_hdf5_tempdir(e)` to the
harness (same AST-rewrite precedent as the existing `flength`/`L`
rewrite, no edits to the example file itself): detects an
`output = Output.HDF5Output("...", ...)` assignment and rewrites the
string-literal path argument to `joinpath(mktempdir(), "...")`.

**Revert-and-rerun proof that these are real regression tests** (not
tests that would pass either way): reconstructed each file's pristine
content via `git show HEAD:<path>` and ran it standalone through the exact
harness logic (flength/HDF5 rewrites included):
- `full_modal/basic_modal_full.jl` (original, unfixed) →
  `FieldError: type Array has no field \`referenceλ\`, available fields:
  \`ref\`, \`size\`` — this is class 2's error, and it fires *first* (before
  the file ever reaches the `linop` line), so this file's revert-proof
  demonstrates class 2 coverage specifically.
- `polarisation/modal_nonvector_plasma.jl` (original, unfixed) →
  `UndefVarError: \`linop\` not defined in \`Main.ExampleSmoke\`` — class
  1's error, demonstrating class 1 coverage.

Both reverts were run as isolated single-file checks (not a combined
stash), each producing the expected class-specific error.

## 6. Tests run

- `LUNA_TEST_GROUP=examples julia --project test/runtests.jl`: **20/20
  pass**, wall time **1m54s** (previously ~45-58s for 8 files per
  hygiene.md's measurement on a loaded host) — **+~56s for the two added
  cases combined** (not split per-file in this pass). The added
  `full_modal/basic_modal_full.jl` case is `full=true`, so its RHS uses
  2-D `Cubature.hcubature_v` rather than the 1-D `pcubature_v` path the
  existing modal cases use, and is very likely most of that delta;
  flagging for the lead in case the group's runtime budget matters.
- `LUNA_TEST_GROUP=sim-multimode julia --project test/runtests.jl`:
  **33/33 pass**, wall time **11m02.4s**. Run to confirm no regression
  from touching modal/polarisation example files — these are example
  scripts, not `src/`, so no regression was expected, and none appeared.
- Per-file smoke checks (standalone Julia scripts, not part of the
  committed test suite) confirmed 6 of 7 originally-broken files now run
  to completion at `SMOKE_LENGTH=5e-3`; `basic_modal_full_bothpolarisations.jl`
  still throws the documented residual `DimensionMismatch` (§4).

## 7. Environment note

This worktree had no `amalthea/target/release/libamalthea.so` and no
`Manifest.toml`. Copied the `.so` from the main checkout
(`/home/diego/Documents/fernando_luz/Luna-Rust.jl/amalthea/target/release/libamalthea.so`,
no Rust source touched) and copied `Manifest.toml` from the main checkout,
then ran `Pkg.instantiate()` (succeeded, package precompiled cleanly).

## 8. Next

Nothing further planned for this item. Residual known-broken file (§4) is
the only thing the lead might want tracked in `BACKLOG.md` (left as their
call per doc-ownership rules for this task).
