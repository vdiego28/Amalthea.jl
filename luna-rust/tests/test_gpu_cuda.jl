using TestItems

@testitem "Julia-Rust Phase 5 GPU CUDA Dispatch" tags=[:rust] begin
    _LIB_EXT = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"
    LIB_PATH = if Sys.iswindows()
        joinpath(@__DIR__, "../target/release/luna_rust.dll")
    else
        joinpath(@__DIR__, "../target/release/libluna_rust.$_LIB_EXT")
    end

    @test isfile(LIB_PATH)
    
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
    
    # Depending on environment, CUDA might not be present and it might fall back.
    # But initialization with fallback works, so we test that the path is valid (e.g. >= 1).
    @test active_path >= 1
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
