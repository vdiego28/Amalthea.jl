import Logging: @info
using TestItemRunner

testdir = dirname(@__FILE__)

import Amalthea: set_fftw_mode, set_fftw_threads
set_fftw_mode(:estimate)

# On Windows and macOS, FFTW's internal thread pool is unstable when Julia
# uses many threads simultaneously. Windows produced
# EXCEPTION_ACCESS_VIOLATION in libfftw3-3.dll; macOS 26 arm64 intermittently
# produced SIGBUS in test_rk45.jl's plain-Julia solve even with fresh,
# host-local wisdom (BACKLOG item 11 / native-port/PLANS.md §7.2). Keep Julia
# threads enabled for threaded coverage, but restrict FFTW itself to one.
if Sys.iswindows() || Sys.isapple()
    set_fftw_threads(1)
end

group = get(ENV, "LUNA_TEST_GROUP", "All")
@info "Running test group: $group"

# Disable strict HDF5 file locking ONLY on Windows runners
if Sys.iswindows()
    ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
end

# `@run_package_tests` walks the package root recursively. Filter that walk
# through the same explicit roots used by the Python bucket runner so serial
# and parallel execution discover exactly the same maintained test surface,
# while nested agent worktrees remain excluded.
const _package_root = normpath(joinpath(testdir, ".."))
const _test_roots_file = joinpath(testdir, "test_roots.txt")
const _test_roots = [
    normpath(joinpath(_package_root, line))
    for line in strip.(readlines(_test_roots_file))
    if !isempty(line) && !startswith(line, "#")
]
function _under_test_root(filename, root)
    parts = splitpath(relpath(abspath(String(filename)), root))
    !isempty(parts) && first(parts) != ".."
end
_in_test_manifest(filename) = any(root -> _under_test_root(filename, root), _test_roots)

if group == "All"
    @run_package_tests filter=ti->_in_test_manifest(ti.filename)
else
    # Run only tests matching the specified group tag
    tag_sym = Symbol(replace(group, "-" => "_"))
    @run_package_tests filter=ti->(tag_sym in ti.tags) && _in_test_manifest(ti.filename)
end
