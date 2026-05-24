using Test
import HDF5

# Resolve the platform-correct shared library extension
const _LIB_EXT = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"
const LIB_PATH = joinpath(@__DIR__, "../target/release/libluna_rust.$_LIB_EXT")

@testset "Julia-Rust Phase 4 Integration (Scans & I/O)" begin
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
    
    # Read the queue status from Julia HDF5 to verify the data was correctly written by Rust
    @test isfile(qfile)
    HDF5.h5open(qfile, "r") do file
        qdata = read(file["qdata"])
        @test qdata[1] == 2  # Completed successfully (Julia is 1-indexed)
        @test qdata[2] == 3  # Failed
        @test qdata[3] == 0  # Not started
    end
    
    # Checkout third item
    idx2 = ccall(
        (:checkout_next_index, LIB_PATH),
        Cssize_t,
        (Ptr{Cvoid},),
        queue_ptr
    )
    @test idx2 == 2
    
    # Checkout when none left should return -1
    idx_none = ccall(
        (:checkout_next_index, LIB_PATH),
        Cssize_t,
        (Ptr{Cvoid},),
        queue_ptr
    )
    @test idx_none == -1
    
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
    
    # Free queue
    ccall(
        (:free_scan_queue, LIB_PATH),
        Cvoid,
        (Ptr{Cvoid},),
        queue_ptr
    )
    
    # All tasks are finished, the queue files should be cleaned up
    @test !isfile(qfile)
    @test !isfile(lock_file)
    
    println("Successfully validated Phase 4 Scans & I/O FFI bindings between Julia and Rust.")
end
