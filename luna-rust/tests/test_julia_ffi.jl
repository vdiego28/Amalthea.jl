using Test

# Resolve the path to the compiled Rust shared library
const LIB_PATH = joinpath(@__DIR__, "../target/release/libluna_rust.so")

@testset "Julia-Rust FFI Verification" begin
    # Verify the compiled library exists
    @test isfile(LIB_PATH)
    
    # Create a complex array in Julia
    original_data = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im]
    test_data = copy(original_data)
    
    # Define scaling factor
    factor = 2.5
    
    # Call the Rust FFI function
    # process_field_inplace(data: *mut c_double, len: size_t, factor: c_double)
    ccall(
        (:process_field_inplace, LIB_PATH),
        Cvoid,
        (Ptr{ComplexF64}, Csize_t, Float64),
        test_data,
        length(test_data),
        factor
    )
    
    # Assert that the array was modified in-place and correctly scaled
    expected_data = original_data .* factor
    @test test_data == expected_data
    
    println("Julia-Rust FFI test completed successfully. Data modified in-place: $test_data")
end
