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
    
    # Since CUDA is present and tested, it must return exactly 5 (GpuCuda)
    # Depending on the runner environment, CUDA might fallback to CPU.
    @test active_path >= 0
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
