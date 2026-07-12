using TestItems

@testitem "Native-Rust Phase 4 (Raman, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # RealGrid, Kerr + Raman only (no plasma, no shot-noise), thg=true —
        # Phase 4's first gate. Molecular gas (:N2) auto-enables the Raman
        # response (Interface.makeresponse). vibration=true, rotation=false:
        # N2's vibrational line is a single SDO with density-independent τ2
        # (eligible for the native path); its rotational line uses a
        # multi-line RamanRespRotationalNonRigid model with density-dependent
        # τ2 (ineligible — same limitation the existing LUNA_USE_RUST_RAMAN
        # FFI wiring already has, not something Phase 4 newly solves).
        radius = 125e-6
        flength = 0.05
        gas = :N2
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange,
               raman=true, rotation=false, vibration=true,
               plasma=false, kerr=true, shotnoise=false,
               energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        @assert grid isa Luna.Grid.RealGrid "Raman test uses RealGrid"
        @assert any(r isa Luna.Nonlinear.RamanPolarField for r in transform.resp) "Expected a RamanPolarField response"

        t0 = 0.0
        dt = 0.01

        # Raman's per-step contribution here is genuinely tiny relative to
        # Kerr (raw RHS ratio ~2e-16 — at the double-precision floor), so a
        # *single* z-step shows an uninformative exact-zero difference
        # between Raman-on and Raman-off (verified: this is expected physics
        # for this pulse regime, not a bug — Raman self-frequency-shift
        # effects need propagation distance to become visible, unlike Kerr
        # self-phase-modulation). The single-step test below therefore only
        # confirms Rust doesn't *diverge* at one step; the meaningful
        # Raman-specific equivalence check is the full-solve testset, which
        # explicitly asserts Raman changes the Julia oracle's result (see
        # PORT_LOG 2026-07-01 for the diagnostic that established this).
        @testset "Single-step equivalence (RealGrid + Raman, ~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Raman single-step rel: ", rel_step)
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (RealGrid + Raman, fixed dt)" begin
            # Fixed step size (max_dt=min_dt=dt): see docs/dev/native-port/TESTING.md
            # §3 and PORT_LOG 2026-07-01 — the adaptive PI controller's
            # near-cancellation error estimate makes raw adaptive-dt agreement
            # an unreliable equivalence signal. Applied from the outset here.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, flength)
            solve(s_ru, flength)

            # Self-validating sanity check: a test that passes identically
            # whether Raman is included or not is testing nothing (see
            # PORT_LOG 2026-07-01). Build a no-Raman reference over the same
            # propagation and confirm Raman actually changes the Julia
            # oracle's result by a margin far above the FP-noise floor
            # (~1e-13, per Phase 1-3) before trusting the Rust-vs-Julia
            # comparison below.
            kw_noraman = (; λ0, λlims, trange, raman=false,
                          plasma=false, kerr=true, shotnoise=false,
                          energy=1e-6, τfwhm=30e-15)
            Eω_nr, _, linop_nr, transform_nr, _, _ = with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw_noraman...)
            end
            s_jl_noraman = PreconStepper(transform_nr, linop_nr, copy(Eω_nr), t0, dt,
                                         rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_jl_noraman, flength)
            rel_raman_matters = norm(s_jl.yn - s_jl_noraman.yn) / norm(s_jl.yn)
            println("Raman on-vs-off (Julia, sanity check): ", rel_raman_matters)
            @test rel_raman_matters > 1e-6

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Raman full-solve rel (Rust vs Julia, both with Raman): ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
