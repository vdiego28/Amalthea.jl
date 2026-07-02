using TestItems

# Julia ↔ Rust PPT ionization equivalence test.
#
# Tagged :rust so it runs in the `rust` CI group and is automatically skipped
# (not failed) when the Rust shared library has not been built yet — matching
# the skip-guard pattern used in test_rust_ffi.jl.
@testitem "Rust PPT ionization equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Luna
    import Logging
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

    # ── build reference Julia IonRatePPTAccel (both toggles OFF) ──────────────
    # We build the LUT manually so the test is hermetic (no disk caching needed)
    # and reproduces what IonRatePPTAccel(material, λ0) would do internally.
    #
    # Since Phase C (BACKLOG.md), the Rust ionisation LUT is built whenever
    # EITHER `LUNA_USE_RUST_IONISATION=1` OR the native stepper is enabled
    # (`LUNA_USE_RUST_NATIVE` defaults to `"1"` since Phase 8) — the native
    # plasma wiring needs this handle to exist for the fork's default
    # workload to actually run natively (REVIEW.md §3.2). So a true
    # Julia-only reference must explicitly force BOTH toggles off.
    λ0    = 800e-9
    gas   = :He
    Emin_full = 1e9
    Emax_full = 5e10
    N = 512

    E_full    = collect(range(Emin_full, Emax_full, length=N))
    rate_full = with_logger(NullLogger()) do
        Ionisation.ionrate_PPT.(gas, λ0, E_full)
    end

    ir_julia = withenv("LUNA_USE_RUST_IONISATION" => "0", "LUNA_USE_RUST_NATIVE" => "0") do
        with_logger(NullLogger()) do
            Ionisation.IonRatePPTAccel(E_full, rate_full)
        end
    end
    @test ir_julia.rust_handle === nothing

    # ── build Rust-backed IonRatePPTAccel (explicit toggle) ───────────────────
    ir_rust = withenv("LUNA_USE_RUST_IONISATION" => "1") do
        with_logger(NullLogger()) do
            Ionisation.IonRatePPTAccel(E_full, rate_full)
        end
    end
    @test ir_rust.rust_handle !== nothing  # handle must be non-null
    @test ir_rust.rust_handle.ptr != C_NULL

    # ── Phase C: native-default alone (no explicit LUNA_USE_RUST_IONISATION)
    # must ALSO build the handle, since LUNA_USE_RUST_NATIVE defaults to "1".
    @testset "Native default builds the ionisation handle without the opt-in toggle" begin
        ir_native_default = withenv("LUNA_USE_RUST_IONISATION" => nothing,
                                     "LUNA_USE_RUST_NATIVE" => nothing) do
            with_logger(NullLogger()) do
                Ionisation.IonRatePPTAccel(E_full, rate_full)
            end
        end
        @test ir_native_default.rust_handle !== nothing
        @test ir_native_default.rust_handle.ptr != C_NULL

        ir_native_explicit_off = withenv("LUNA_USE_RUST_IONISATION" => nothing,
                                          "LUNA_USE_RUST_NATIVE" => "0") do
            with_logger(NullLogger()) do
                Ionisation.IonRatePPTAccel(E_full, rate_full)
            end
        end
        @test ir_native_explicit_off.rust_handle === nothing
    end

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

    # ── above-cutoff: Rust's own PptIonizationRate::rate clamps to rate(Emax),
    # matching Julia's IonRatePPTAccel, instead of erroring and silently
    # falling back to the Julia path (the fallback-and-warn-spam bug fixed in
    # BACKLOG.md Phase B.2). Assert both the value and that no fallback
    # warning fires (Logging.with_logger + a custom logger that records
    # whether any @warn was emitted).
    @testset "Above-cutoff clamps (no error, no fallback)" begin
        out_hi_julia = Vector{Float64}(undef, 3)
        out_hi_rust  = Vector{Float64}(undef, 3)
        E_hi = fill(ir_julia.Emax * 1.5, 3)
        ir_julia(out_hi_julia, E_hi)

        warned = Ref(false)
        logger = Logging.SimpleLogger(IOBuffer())
        with_logger(logger) do
            ir_rust(out_hi_rust, E_hi)
        end
        # Direct check on the Rust handle: no fallback warning means the Rust
        # vector call itself succeeded (returned 0), not a Julia fallback.
        take!(logger.stream) |> String |> s -> (warned[] = occursin("falling back", s))
        @test !warned[]
        @test all(out_hi_rust .≈ out_hi_julia)
        @test out_hi_rust[1] ≈ exp(ir_julia.spline(ir_julia.Emax))
    end
end
