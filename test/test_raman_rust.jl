using TestItems

# Julia ↔ Rust time-domain Raman equivalence test.
#
# Tagged :rust so it runs in the `rust` CI group and is automatically skipped
# (not failed) when the Rust shared library has not been built — matching the
# skip-guard pattern used in test_rust_ffi.jl and test_ionisation_rust.jl.
@testitem "Rust Raman equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Amalthea
    import Amalthea: Nonlinear, Raman
    import Logging: with_logger, NullLogger

    # ── locate the shared library ──────────────────────────────────────────────
    libname = if Sys.iswindows()
        "amalthea.dll"
    elseif Sys.isapple()
        "libamalthea.dylib"
    else
        "libamalthea.so"
    end
    libpath = joinpath(@__DIR__, "..", "amalthea", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping Rust Raman equivalence test: shared library not found at $libpath. " *
              "Build it with `cargo build --release` in amalthea/ (or run `]build Amalthea`)."
        return
    end

    # ── SDO response parameters ────────────────────────────────────────────────
    K_raw = 1.0   # arbitrary (will be normalised inside RamanRespNormedSingleDampedOscillator)
    Ω     = 1e13  # 10 THz oscillator frequency [rad/s]
    τ2    = 50e-15 # 50 fs coherence time [s]

    # Time grid: 1024 points × 1 fs = 1.024 ps total
    dt = 1e-15
    Nt = 1024
    t  = collect(range(0.0, step=dt, length=Nt))

    rr = Raman.CombinedRamanResponse(t,
             [Raman.RamanRespNormedSingleDampedOscillator(K_raw, Ω, τ2)])

    # ── Julia-only path (no rust_handle, default) ─────────────────────────────
    RP_julia = with_logger(NullLogger()) do
        Nonlinear.RamanPolarField(t, rr)
    end
    @test RP_julia.rust_handle === nothing

    # ── Rust-backed path: build handle manually from SDO parameters ───────────
    # Extract normalised K stored in the SDO after normalisation by (Ω² + 1/τ2²)/Ω.
    # Parameter mapping: omega=Ω, gamma=1/τ2, coupling=K_normed (see CLAUDE.md).
    K_normed  = rr.Rs[1].K
    omegas    = Float64[Ω]
    gammas    = Float64[1.0/τ2]
    couplings = Float64[K_normed]

    rust_handle = withenv("AMALTHEA_USE_RUST_RAMAN" => "1") do
        with_logger(NullLogger()) do
            Nonlinear._make_rust_raman_handle(omegas, gammas, couplings, dt)
        end
    end
    @test rust_handle !== nothing
    @test rust_handle.ptr != C_NULL

    RP_rust = with_logger(NullLogger()) do
        Nonlinear.RamanPolarField(t, rr; rust_handle)
    end
    @test RP_rust.rust_handle !== nothing

    # ── drive both paths with the same Gaussian test field ────────────────────
    # Use a wide pulse (τw=80 fs > dt=1 fs) so the piecewise-linear intensity
    # approximation in the Rust ADE is accurate: error ≈ O(dt/τw)² ≈ O(1.6e-4).
    t0 = 300e-15  # pulse centre [s] — centred in the 1024 fs grid
    τw = 80e-15   # pulse 1/e half-width [s]
    E  = @. exp(-((t - t0) / τw)^2)

    out_julia = zeros(Float64, Nt)
    out_rust  = zeros(Float64, Nt)
    ρ = 1.0

    RP_julia(out_julia, E, ρ)
    RP_rust(out_rust,   E, ρ)

    @testset "Julia vs Rust Raman polarisation" begin
        # The two methods are fundamentally different numerical schemes:
        #   Julia:  windowed FFT convolution on a doubled grid (O(Nt log Nt))
        #   Rust:   exponential-integrator ADE, no window (O(Nt), phase-exact)
        # Their outputs differ by:
        #   (a) Planck-taper window applied in Julia (negligible here: τ2=50fs ≪ grid)
        #   (b) Piecewise-linear intensity interpolation error in the ADE ≈ O(dt/τw)²
        # For dt=1fs, τw=80fs: expected relative agreement ≈ 2e-4 → we allow 3e-3.
        peak = maximum(abs.(out_julia))
        threshold = peak * 1e-3  # skip near-zero points where division is unstable
        n_compared = 0
        for i in eachindex(out_julia)
            abs(out_julia[i]) < threshold && continue
            rel_err = abs(out_rust[i] - out_julia[i]) / abs(out_julia[i])
            @test rel_err < 3e-3
            n_compared += 1
        end
        @test n_compared > 10  # sanity: at least 10 signal points were checked
    end

    # ── idempotent: solver resets state each call ─────────────────────────────
    @testset "Idempotent: solver resets state each call" begin
        out_rust2 = zeros(Float64, Nt)
        RP_rust(out_rust2, E, ρ)
        @test out_rust ≈ out_rust2
    end

    # ── toggle off: default constructor always returns julia path ─────────────
    @testset "Toggle off: no Rust handle by default" begin
        RP_norust = with_logger(NullLogger()) do
            Nonlinear.RamanPolarField(t, rr)
        end
        @test RP_norust.rust_handle === nothing
    end
end
