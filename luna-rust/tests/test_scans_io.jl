using TestItems

# Safely disable locking for local Windows testing before HDF5 is loaded.
Sys.iswindows() && (ENV["HDF5_USE_FILE_LOCKING"] = "FALSE")

@testitem "Julia-Rust Phase 4 Integration (Scans & I/O)" tags=[:rust] begin
    import HDF5

    # Resolve the platform-correct shared library extension
    _LIB_EXT = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"
    LIB_PATH = if Sys.iswindows()
        joinpath(@__DIR__, "../target/release/luna_rust.dll")
    else
        joinpath(@__DIR__, "../target/release/libluna_rust.$_LIB_EXT")
    end

    # Explicitly load `libsz` and other dependent libraries implicitly needed by `HDF5.API.libhdf5` before calling Rust.
    # When `dlopen` is called from Rust on `libhdf5.so`, it frequently fails due to missing transitive
    # dependencies that are normally resolved by Julia's package manager when `using HDF5` is done at the module level.
    # By forcing Julia to fully materialize HDF5 bindings *before* Rust tries to load it, we sidestep the dlopen error.
    HDF5.h5open

    # Pass HDF5 path to Rust side so it doesn't have to guess or search for it
    ENV["LUNA_HDF5_LIB"] = HDF5.API.libhdf5
    
    @test isfile(LIB_PATH)
    
    qfile = "julia_test_queue.h5"
    lock_file = qfile * "_lock"
    
    # Ensure clean state
    rm(qfile, force=true)
    rm(lock_file, force=true)
    
    total_points = 3
    
    # Initialize queue
    queue_ptr = ccall(
        (:init_scan_queue, LIB_PATH),
        Ptr{Cvoid},
        (Cstring, Csize_t),
        qfile,
        total_points
    )
    @test queue_ptr != C_NULL
    
    # First checkout should get index 0
    idx0 = ccall(
        (:checkout_next_index, LIB_PATH),
        Cssize_t,
        (Ptr{Cvoid},),
        queue_ptr
    )
    @test idx0 == 0
    
    # Second checkout should get index 1
    idx1 = ccall(
        (:checkout_next_index, LIB_PATH),
        Cssize_t,
        (Ptr{Cvoid},),
        queue_ptr
    )
    @test idx1 == 1
    
    # Mark index 0 as completed successfully (success = 1)
    res0 = ccall(
        (:mark_completed, LIB_PATH),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cint),
        queue_ptr,
        0,
        1
    )
    @test res0 == 0
    
    # Mark index 1 as failed (success = 0)
    res1 = ccall(
        (:mark_completed, LIB_PATH),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cint),
        queue_ptr,
        1,
        0
    )
    @test res1 == 0
    
    # Checkout third item
    idx2 = ccall(
        (:checkout_next_index, LIB_PATH),
        Cssize_t,
        (Ptr{Cvoid},),
        queue_ptr
    )
    @test idx2 == 2
    
    # Mark index 2 as success
    res2 = ccall(
        (:mark_completed, LIB_PATH),
        Cint,
        (Ptr{Cvoid}, Csize_t, Cint),
        queue_ptr,
        2,
        1
    )
    @test res2 == 0

    # Checkout when none left should return -1
    idx_none = ccall(
        (:checkout_next_index, LIB_PATH),
        Cssize_t,
        (Ptr{Cvoid},),
        queue_ptr
    )
    @test idx_none == -1

    # Destroy the Rust queue object so all Rust file handles are dropped before Julia reopens the file.
    ccall(
        (:free_scan_queue, LIB_PATH),
        Cvoid,
        (Ptr{Cvoid},),
        queue_ptr
    )

    # Now Julia can safely inspect the final queue state sequentially.
    @test isfile(qfile)
    HDF5.h5open(qfile, "r") do file
        qdata = read(file["qdata"])
        @test qdata[1] == 2  # Completed successfully (Julia is 1-indexed)
        @test qdata[2] == 3  # Failed
        @test qdata[3] == 2  # Completed successfully
    end

    # Clean up test artifacts explicitly after validation.
    rm(qfile, force=true)
    rm(lock_file, force=true)

    println("Successfully validated Phase 4 Scans & I/O FFI bindings between Julia and Rust.")
end
