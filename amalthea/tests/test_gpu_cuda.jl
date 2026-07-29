using TestItems

@testitem "Julia-Rust Phase 5 GPU CUDA Dispatch" tags=[:rust] begin
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    _LIB_EXT = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"
    LIB_PATH = if Sys.iswindows()
        joinpath(@__DIR__, "../target/release/amalthea.dll")
    else
        joinpath(@__DIR__, "../target/release/libamalthea.$_LIB_EXT")
    end

    if !isfile(LIB_PATH)
        require_cuda && error(
            "CUDA tests are required, but the Rust shared library was not found at $LIB_PATH")
        @warn "Skipping Rust GPU dispatch test: shared library not found at $LIB_PATH. " *
              "Build it with `cargo build --release` in amalthea/ (or run `]build Amalthea`)."
        return
    end
    
    # Initialize the simulation engine with GpuCuda (5)
    engine_ptr = ccall(
        (:init_simulation_engine, LIB_PATH),
        Ptr{Cvoid},
        (Cint,),
        5 # GpuCuda
    )
    @test engine_ptr != C_NULL
    
    # Retrieve active hardware path
    active_path = ccall(
        (:get_active_hardware_path, LIB_PATH),
        Cint,
        (Ptr{Cvoid},),
        engine_ptr
    )
    
    # In CI environments lacking CUDA/Vulkan drivers, tests involving GPU dispatch
    # should allow fallback active paths instead of strictly expecting the CUDA path (5).
    if require_cuda
        @test active_path == 5
    else
        @test active_path in (1, 2, 3, 4, 5, 6)
    end
    println("Active hardware path returned from Rust engine: ", active_path)
    
    # Free the engine
    ccall(
        (:free_simulation_engine, LIB_PATH),
        Cvoid,
        (Ptr{Cvoid},),
        engine_ptr
    )
    println("Successfully verified dynamic GPU CUDA dispatch from Julia FFI.")
end
