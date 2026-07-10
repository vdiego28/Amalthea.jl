using TestItemRunner
import Luna: set_fftw_mode

# Runs the `rust`-tagged test items belonging to one or more files, in this
# single process. Used by `parallel_rust_tests.py` to distribute the `rust`
# test group across several load-balanced worker processes. Must live in
# `test/` (not e.g. a scratch dir) — `@run_package_tests` resolves the
# package root from this file's own location.
set_fftw_mode(:estimate)

targets = Set(ARGS)
@run_package_tests filter=ti->(:rust in ti.tags && basename(String(ti.filename)) in targets)
