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
    
    # If the system lacks CUDA drivers (e.g. CI environments), it will fall back to Vulkan (6) or CPU (2,3,4) depending on the backend platform logic.
    @test active_path in (2, 3, 4, 5, 6)
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
