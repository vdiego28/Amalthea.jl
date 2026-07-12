@testitem "Rust FFI" tags=[:rust] begin
    using Amalthea

    # Locate the shared library
    libname = if Sys.iswindows()
        "luna_rust.dll"
    elseif Sys.isapple()
        "libluna_rust.dylib"
    else
        "libluna_rust.so"
    end
    libpath = joinpath(@__DIR__, "..", "luna-rust", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust FFI test: shared library not found at $libpath. " *
              "Build it with `cargo build --release` in luna-rust/ (or run `]build Amalthea`)."
        return
    end

    # Test process_field_inplace FFI call
    data = ComplexF64[1.0+2.0im, 3.0+4.0im]
    GC.@preserve data begin
        ptr = pointer(reinterpret(Float64, data))
        @ccall libpath.process_field_inplace(ptr::Ptr{Float64}, 2::Csize_t, 2.0::Float64)::Cvoid
    end
    @test data[1] ≈ 2.0 + 4.0im
    @test data[2] ≈ 6.0 + 8.0im
end
