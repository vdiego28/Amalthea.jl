import Logging: @info
using TestItemRunner

testdir = dirname(@__FILE__)

import Amalthea: set_fftw_mode, set_fftw_threads
set_fftw_mode(:estimate)

# On Windows, FFTW's internal thread pool is unstable when Julia uses many threads
# simultaneously, leading to EXCEPTION_ACCESS_VIOLATION crashes in libfftw3-3.dll.
# Restrict FFTW to a single thread to avoid this.
if Sys.iswindows()
    set_fftw_threads(1)
end

group = get(ENV, "LUNA_TEST_GROUP", "All")
@info "Running test group: $group"

# Disable strict HDF5 file locking ONLY on Windows runners
if Sys.iswindows()
    ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
end

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
    # Run only tests matching the specified group tag
    tag_sym = Symbol(replace(group, "-" => "_"))
    @run_package_tests filter=ti->(tag_sym in ti.tags) && !_under_claude_dir(ti.filename)
end
