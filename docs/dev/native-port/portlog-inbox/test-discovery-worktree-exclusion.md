# Inbox note — BACKLOG.md item 10: test discovery recurses into `.claude/worktrees/`

Branch: `test-discovery-claude-exclusion` (off `main`, not merged, not pushed).

---

## Root cause

`test/runtests.jl` calls `@run_package_tests` (TestItemRunner.jl). The macro
expands to `run_tests(joinpath(dirname(@__FILE__), ".."))` — i.e. it resolves
the package root once, then `run_tests` (`TestItemRunner.jl`'s `walkdir(path)`
loop) recurses into *every* subdirectory under that root with no exclusion at
all, collecting every `.jl` file and parsing it for `@testitem`s. There was no
directory-argument or exclude-glob on the macro itself — the only lever
`@run_package_tests` exposes is a post-discovery `filter=` predicate over
`(filename, name, tags)`.

Any git worktree checked out under `.claude/worktrees/<name>/` (the location
this repo's agent-worktree tooling uses) contains its own full copy of
`test/`. `walkdir` does not know or care that this is a worktree — it just
finds more `.jl` files with the same `@testitem tags=[:group]` markers, so
every worktree present at test-run time multiplies the group's assertion
count by however many checkouts exist under `.claude/worktrees/`.

This was already known and already fixed on the *parallel* discovery path
(see "Python shard scripts" below) — `test/run_group_bucket.jl` has carried an
`in_this_checkout` guard since commit `fe08fa9` ("Test harness: fix two bugs
that made the full gate report a false 0/0 FAIL"). The gap was specifically
the plain serial entry point, `test/runtests.jl`, invoked directly via
`LUNA_TEST_GROUP=<group> julia --project test/runtests.jl` — which is also
what `AGENTS.md` §3 step 5 tells every agent to run after a change. That path
had no such guard in either of its two branches.

## The fix

`test/runtests.jl`, both branches (`group == "All"` and the tagged `else`):

```julia
# `@run_package_tests` (TestItemRunner) resolves the package root as
# `joinpath(dirname(@__FILE__), "..")` from *this* file and then `walkdir`s
# it with no exclusion (TestItemRunner.jl's `run_tests`), so it also
# recurses into any git worktree checked out under `.claude/worktrees/`.
# Each such worktree carries its own full copy of `test/`, so a stray
# worktree makes every group's assertion count a multiple of the true
# count (docs/dev/BACKLOG.md item 10; measured 2026-07-25: `examples`
# reported 120/120 from the root vs. the true 20/20 in a clean worktree).
# Match `.claude` as a whole path *component* relative to the package root
# (not `occursin(".claude", ti.filename)`, which would misfire the day
# someone checks this repo out under a path containing the literal
# substring ".claude") so legitimate auto-discovered files elsewhere in the
# tree — e.g. `amalthea/tests/*.jl` (BACKLOG.md line ~1182, the `:rust`
# safety net) — are still picked up.
const _package_root = normpath(joinpath(testdir, ".."))
_under_claude_dir(filename) = ".claude" in splitpath(relpath(String(filename), _package_root))

if group == "All"
    @run_package_tests filter=ti->!_under_claude_dir(ti.filename)
else
    tag_sym = Symbol(replace(group, "-" => "_"))
    @run_package_tests filter=ti->(tag_sym in ti.tags) && !_under_claude_dir(ti.filename)
end
```

### Why a `filter=` predicate, not a path/directory argument

Confirmed against the installed `TestItemRunner` (`~/.julia/packages/TestItemRunner/GnoVt/src/TestItemRunner.jl`):
- `@run_package_tests` only accepts `filter=` and `verbose=` keyword
  arguments (the macro body rejects anything else — `error("Invalid
  argument")`). There is no way to pass a restricted root or an exclude-glob
  through the macro.
- The underlying `run_tests(path; filter, verbose)` function *does* take a
  directory, but the macro hardcodes `path = joinpath(dirname(@__FILE__),
  "..")` — not overridable without calling `run_tests` directly instead of
  the macro, which would be a bigger surface change than this item calls for.
- The `filter` predicate TestItemRunner invokes is a `NamedTuple`
  `(filename=file, name=i.name, tags=i.option_tags)` — `ti.filename` is
  real (confirmed by reading `run_tests`'s `Base.filter(i -> filter((filename=file, ...` line), not
  a guess. `filename` is the absolute, `normpath`-ed path assembled during
  `walkdir`.

So a `filter=` predicate over `ti.filename` was the only lever available, and
is also exactly the mechanism `test/run_group_bucket.jl` already uses for its
own (different) exclusion problem.

### Why a path-component match, not a substring

`occursin(".claude", ti.filename)` would also match a checkout at, say,
`/home/alice/projects/my.claude.fork/Amalthea.jl/...` — a real risk given
`.claude` is a generic dotfile name, not something namespaced to this repo.
`splitpath(relpath(filename, _package_root))` breaks the path *relative to
the package root* into components and checks for an exact `".claude"`
component, so it only fires on an actual `.claude/` directory inside this
checkout, regardless of what the checkout's own absolute path looks like.

### What stays discoverable

`amalthea/tests/*.jl` (four `@testitem tags=[:rust]` files — the auto-discovered
half of the `rust` group's safety net per `CLAUDE.md`/`BACKLOG.md` line ~1182)
sits at `amalthea/tests/`, which has no `.claude` path component, so it is
untouched by this filter. Confirmed the fix does not regress this by
inspection — no `rust`-group run was needed to prove this, since the filter
predicate is a straightforward path-component check and `amalthea/tests/`
does not contain a `.claude` segment under any relpath from the package root.

## Constructed reproducer (deleted after use — never `git add`ed)

There were no leftover worktrees in the tree as of 2026-07-26 (the prior day's
sweep removed them), so the inflation had to be reconstructed by hand:

1. `mkdir -p .claude/worktrees/fake/test/`
2. `cp test/test_noise.jl .claude/worktrees/fake/test/test_noise.jl` — one
   small `:fields`-tagged file (`@testitem "Noise" tags=[:fields]`).
3. Measured `LUNA_TEST_GROUP=fields julia --project test/runtests.jl`
   **with the fix reverted** (`git stash push -- test/runtests.jl`, i.e. the
   pre-fix two-branch code with no filter at all in the `All`/no-tag-filter
   path — same bare-filter shape in the tagged branch too):
   **432/432** — 98 assertions higher than the 334 baseline, i.e. exactly the
   duplicated `test_noise.jl`'s own contribution counted twice.
4. Restored the fix (`git stash pop`), re-ran the identical command with the
   confounder directory still present: **334/334** — unchanged from the
   documented baseline; the fake worktree copy is now silently skipped.
5. Removed the throwaway directory (`rm -rf .claude/worktrees/fake`) and
   re-ran once more on the fully clean tree: **334/334**, matching
   BACKLOG.md's gate-table baseline for `fields` exactly.
6. `git status --short` after cleanup shows only `test/runtests.jl` modified
   — the confounder directory was never `git add`ed and leaves no trace.

| Run | Confounder present | Fix applied | Result |
|---|---|---|---|
| Baseline sanity check | no | yes | 334/334 |
| Reproducer "before" | yes | **no** (reverted) | **432/432 (inflated)** |
| Reproducer "after" | yes | yes | 334/334 (correct) |
| Final clean-tree check | no | yes | 334/334 |

## Python shard scripts (BACKLOG item 10's "second discovery surface")

Checked both `test/parallel_group_tests.py` and `test/run_full_gate.py` for
their own file-discovery logic (this is how the gate actually runs, per
`CLAUDE.md`):

- `test/parallel_group_tests.py`'s `discover_group_files()` calls
  `TEST_DIR.glob("*.jl")` — a **non-recursive** glob scoped to the literal
  `test/` directory (`TEST_DIR = REPO_ROOT / "test"`). `pathlib.Path.glob`
  with a non-`**` pattern does not descend into subdirectories at all, so it
  can never see `.claude/worktrees/<x>/test/*.jl` regardless of how many
  worktrees exist. This script is not affected by the bug.
- `test/run_full_gate.py` does no file discovery of its own — it only
  imports `prepare_group_bins`/`run_groups` from `parallel_group_tests.py`
  (`from parallel_group_tests import (prepare_group_bins, run_groups,
  DEFAULT_MAX_WORKERS)`), so it inherits the same non-recursive-glob safety.
- The actual test execution for both scripts happens in a subprocess running
  `test/run_group_bucket.jl`, which **does** call `@run_package_tests`
  (so it does re-walk the whole repo internally) — but that file already
  carries its own guard, predating this fix:
  ```julia
  const THIS_TEST_DIR = @__DIR__
  in_this_checkout(f) = dirname(abspath(String(f))) == THIS_TEST_DIR
  @run_package_tests filter=ti->(tag_sym in ti.tags &&
                                basename(String(ti.filename)) in targets &&
                                in_this_checkout(ti.filename))
  ```
  added in commit `fe08fa9`, with its own comment already naming
  `.claude/worktrees/` as the reason. So the parallel/gate path was already
  closed; only the plain serial entry point (`test/runtests.jl`, invoked
  directly, e.g. by `AGENTS.md` §3 step 5's "run
  `LUNA_TEST_GROUP=rust julia --project test/runtests.jl`") had the gap.

**Conclusion: no changes needed to either Python script or to
`run_group_bucket.jl`.** Only `test/runtests.jl` was patched.

## Decided against

- **Restricting discovery via a stricter `dirname(filename) == testdir`
  check** (the same style `run_group_bucket.jl` uses, which pins to exactly
  one directory). Rejected: `run_group_bucket.jl` can afford this because it
  is only ever invoked once per bucket, on file names it already knows live
  directly in `test/`. `test/runtests.jl` (via `@run_package_tests`'s "All"
  branch and the auto-discovery `CLAUDE.md` documents for `amalthea/tests/*.jl`)
  is relied on to discover `:rust`-tagged files outside `test/` — a
  `dirname == testdir` filter would have silently dropped those four files
  from the `rust` group's serial run, understating that group's true count
  instead of overstating it. The path-component match was chosen precisely
  because it is the narrower exclusion that only removes `.claude/` and
  nothing else auto-discovered elsewhere in the tree.
- **Pruning stale worktrees before gating, or always running the gate from
  inside a worktree** (the other two options BACKLOG.md item 10 offered as
  alternatives). Both are process/discipline fixes with no enforcement — a
  forgotten worktree or a serial `runtests.jl` invocation from the repo root
  (which is exactly what `AGENTS.md` tells agents to do) would silently
  reintroduce the inflation. Excluding `.claude/` at the source is the only
  option that holds regardless of what state the tree happens to be in.

## PORT_LOG.md entry

Per repo policy (`MEMORY.md`: "inbox files, not `PORT_LOG.md` edits"), this
file is the durable record; it is **not** appended to `PORT_LOG.md`.
