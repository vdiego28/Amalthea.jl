using TestItemRunner
import Amalthea: set_fftw_mode, set_fftw_threads

# Runs one or more exact test items, filtered to a single
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
if Sys.iswindows() || Sys.isapple()
    set_fftw_threads(1)
end
if Sys.iswindows()
    ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
end

tag_sym = Symbol(ENV["LUNA_BUCKET_TAG"])
target_specs = filter(!isempty, split(
    get(ENV, "LUNA_BUCKET_ITEMS", get(ENV, "LUNA_BUCKET_FILES", "")),
    "\n",
))

# Resolve the checkout-relative manifest names emitted by
# `parallel_group_tests.py`. Historical entries for files directly under
# `test/` remain basenames; secondary/nested roots use repository-relative
# names so cross-root filename collisions cannot alias.
const THIS_TEST_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(THIS_TEST_DIR, ".."))
target_path(name) = normpath(abspath(joinpath(
    occursin('/', name) || occursin('\\', name) ? REPO_ROOT : THIS_TEST_DIR,
    name,
)))

const TARGET_FILES = Set{String}()
const TARGET_ITEMS = Set{Tuple{String,String}}()
for spec in target_specs
    parts = split(spec, "::"; limit=2)
    path = target_path(parts[1])
    if length(parts) == 1
        push!(TARGET_FILES, path)
    else
        push!(TARGET_ITEMS, (path, parts[2]))
    end
end

function is_target(ti)
    path = normpath(abspath(String(ti.filename)))
    path in TARGET_FILES || (path, String(ti.name)) in TARGET_ITEMS
end

@run_package_tests filter=ti->(tag_sym in ti.tags &&
                              is_target(ti))
