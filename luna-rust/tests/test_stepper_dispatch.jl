using Test

# Resolve the path to the compiled Rust shared library
const LIB_PATH = joinpath(@__DIR__, "../target/release/libluna_rust.so")

@testset "Julia-Rust Phase 3 Integration (Stepper & Dispatch)" begin
    @test isfile(LIB_PATH)
    
    # Initialize the simulation engine with Auto (0)
    engine_ptr = ccall(
        (:init_simulation_engine, LIB_PATH),
        Ptr{Cvoid},
        (Cint,),
        0 # Auto
    )
    @test engine_ptr != C_NULL
    
    # Retrieve active hardware path
    active_path = ccall(
        (:get_active_hardware_path, LIB_PATH),
        Cint,
        (Ptr{Cvoid},),
        engine_ptr
    )
    @test 0 <= active_path <= 7
    println("Active hardware path returned from Rust engine: ", active_path)
    
    # Free the engine
    ccall(
        (:free_simulation_engine, LIB_PATH),
        Cvoid,
        (Ptr{Cvoid},),
        engine_ptr
    )
    println("Successfully initialized, queried, and freed the Rust Simulation Engine from Julia FFI.")
end
