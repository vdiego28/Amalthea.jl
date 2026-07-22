using TestItemRunner
import Amalthea: set_fftw_mode

# Runs the test items belonging to one or more files, filtered to a single
# CI test-group tag, in this single process. Used by
# `parallel_group_tests.py` to distribute any test group across several
# load-balanced worker processes. Must live in `test/` (not e.g. a scratch
# dir) — `@run_package_tests` resolves the package root from this file's
# own location.
#
# Deliberately configured via ENV, not command-line ARGS: `Scans.jl`'s
# `Scan()` constructor defaults to reading Julia's own `ARGS` global and
# parsing it as CLI flags (-l/-r/-b/-q/-p via ArgParse). Any bucket file
# that calls `Scan()` with no explicit args (e.g. test_processing.jl) would
# otherwise see our tag/filenames as unexpected positional arguments and
# fail with "too many arguments" — confirmed via the `fields` group's
# first run under this script.
set_fftw_mode(:estimate)

tag_sym = Symbol(ENV["LUNA_BUCKET_TAG"])
targets = Set(split(ENV["LUNA_BUCKET_FILES"], "\n"))

# `@run_package_tests` walks the package root recursively, which sweeps in
# any git worktree living under `.claude/worktrees/<name>/test/`. Those hold
# their own copies of every test file, so a bucket matching only on
# `basename` would run each item once per checkout (inflating assertion
# counts 2-3x) and would fail on worktrees that never built amalthea. Pin
# the bucket to *this* checkout's test directory.
const THIS_TEST_DIR = @__DIR__
in_this_checkout(f) = dirname(abspath(String(f))) == THIS_TEST_DIR

@run_package_tests filter=ti->(tag_sym in ti.tags &&
                              basename(String(ti.filename)) in targets &&
                              in_this_checkout(ti.filename))
