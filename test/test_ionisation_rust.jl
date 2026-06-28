using TestItems

# Julia ↔ Rust PPT ionization equivalence test.
#
# Tagged :rust so it runs in the `rust` CI group and is automatically skipped
# (not failed) when the Rust shared library has not been built yet — matching
# the skip-guard pattern used in test_rust_ffi.jl.
@testitem "Rust PPT ionization equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Luna
    import Logging: with_logger, NullLogger

    # ── locate the shared library ──────────────────────────────────────────────
    libname = if Sys.iswindows()
        "luna_rust.dll"
    elseif Sys.isapple()
        "libluna_rust.dylib"
    else
        "libluna_rust.so"
    end
    libpath = joinpath(@__DIR__, "..", "luna-rust", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust ionization equivalence test: shared library not found at $libpath. " *
              "Build it with `cargo build --release` in luna-rust/ (or run `]build Luna`)."
        return
    end

    # ── build reference Julia IonRatePPTAccel (toggle OFF) ────────────────────
    # We build the LUT manually so the test is hermetic (no disk caching needed)
    # and reproduces what IonRatePPTAccel(material, λ0) would do internally.
    λ0    = 800e-9
    gas   = :He
    Emin_full = 1e9
    Emax_full = 5e10
    N = 512

    E_full    = collect(range(Emin_full, Emax_full, length=N))
    rate_full = with_logger(NullLogger()) do
        Ionisation.ionrate_PPT.(gas, λ0, E_full)
    end

    # Julia-only reference (explicitly pass toggle off)
    ir_julia = with_logger(NullLogger()) do
        Ionisation.IonRatePPTAccel(E_full, rate_full)
    end
    # Confirm no Rust handle was attached (toggle default = off)
    @test ir_julia.rust_handle === nothing

    # ── build Rust-backed IonRatePPTAccel (toggle ON) ─────────────────────────
    ir_rust = withenv("LUNA_USE_RUST_IONISATION" => "1") do
        with_logger(NullLogger()) do
            Ionisation.IonRatePPTAccel(E_full, rate_full)
        end
    end
    @test ir_rust.rust_handle !== nothing  # handle must be non-null
    @test ir_rust.rust_handle.ptr != C_NULL

    # ── test field array spanning Emin..Emax ──────────────────────────────────
    # Both paths are cubic splines fitted to ln(rate); we expect agreement to
    # the spline interpolation precision (~1e-8 relative) — NOT machine epsilon,
    # because the two implementations (Julia Maths.CSpline vs Rust CubicSplineLUT)
    # use identical natural cubic spline coefficients on the same knots.
    E_test = collect(range(ir_julia.Emin * 1.01, ir_julia.Emax * 0.99, length=200))
    out_julia = Vector{Float64}(undef, length(E_test))
    out_rust  = Vector{Float64}(undef, length(E_test))

    ir_julia(out_julia, E_test)
    ir_rust(out_rust,   E_test)

    @testset "Julia vs Rust rate values" begin
        for i in eachindex(E_test)
            rel_err = abs(out_rust[i] - out_julia[i]) / out_julia[i]
            @test rel_err < 1e-8
        end
    end

    # ── test scalar path (always Julia, even when Rust handle is present) ─────
    @testset "Scalar path" begin
        E_scalar = 2.0e10
        @test ir_julia(E_scalar) ≈ ir_rust(E_scalar)
    end

    # ── below-cutoff: both must return exactly 0.0 ────────────────────────────
    @testset "Below-cutoff returns 0.0" begin
        out_lo_julia = Vector{Float64}(undef, 5)
        out_lo_rust  = Vector{Float64}(undef, 5)
        E_lo = fill(ir_julia.Emin * 0.1, 5)
        ir_julia(out_lo_julia, E_lo)
        ir_rust(out_lo_rust,   E_lo)
        @test all(iszero, out_lo_julia)
        @test all(iszero, out_lo_rust)
    end

    # ── above-cutoff: Rust falls back to Julia path (both clamp) ─────────────
    @testset "Above-cutoff clamps (no error)" begin
        out_hi_julia = Vector{Float64}(undef, 3)
        out_hi_rust  = Vector{Float64}(undef, 3)
        E_hi = fill(ir_julia.Emax * 1.5, 3)
        ir_julia(out_hi_julia, E_hi)
        ir_rust(out_hi_rust,   E_hi)
        # The Rust -2 error code triggers a fallback to Julia, so outputs agree
        @test all(out_hi_rust .≈ out_hi_julia)
    end
end
