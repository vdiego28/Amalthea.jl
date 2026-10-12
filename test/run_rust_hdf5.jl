# Run in a fresh Rust process: loading HDF5 in this parent must not hide
# initialization defects in the standalone library. Use Julia's resolved
# library and transitive dependency paths instead of guessing artifact paths.
using HDF5

repo = dirname(@__DIR__)
libpath_env = Sys.iswindows() ? "PATH" :
              Sys.isapple() ? "DYLD_FALLBACK_LIBRARY_PATH" : "LD_LIBRARY_PATH"
separator = Sys.iswindows() ? ';' : ':'
dependency_paths = isdefined(HDF5.API, :HDF5_jll) ? HDF5.API.HDF5_jll.LIBPATH[] : ""
libpaths = join(filter(!isempty, [dirname(HDF5.API.libhdf5), dependency_paths,
                                 get(ENV, libpath_env, "")]), separator)
println("Required standalone Rust HDF5 tests using ", HDF5.API.libhdf5)
function required_hdf5_tests(arguments, count)
    command = Cmd(`cargo test --locked --release $arguments -- --test-threads=1 --nocapture`;
                  dir=joinpath(repo, "amalthea"))
    output = IOBuffer()
    process = run(pipeline(ignorestatus(addenv(command,
        "AMALTHEA_HDF5_LIB" => HDF5.API.libhdf5,
        "AMALTHEA_REQUIRE_HDF5_TESTS" => "1", libpath_env => libpaths)); stdout=output))
    results = String(take!(output))
    print(results)
    success(process) || error("Required standalone Rust HDF5 tests failed")
    occursin("test result: ok. $count passed; 0 failed; 0 ignored;", results) ||
        error("Required standalone Rust HDF5 gate must exercise all $count tests without skips")
end
required_hdf5_tests(["--lib", "test_hdf5_"], 5)
required_hdf5_tests(["--test", "hdf5_queue_process", "test_hdf5_queue_processes"], 1)
