import Logging: @info
using TestItemRunner

testdir = dirname(@__FILE__)

import Luna: set_fftw_mode, set_fftw_threads
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

if group == "All"
    @run_package_tests
else
    # Run only tests matching the specified group tag
    tag_sym = Symbol(replace(group, "-" => "_"))
    @run_package_tests filter=ti->(tag_sym in ti.tags)
end