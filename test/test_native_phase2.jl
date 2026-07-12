using TestItems

@testitem "Native-Rust Phase 2a (mode-avg Kerr, EnvGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # EnvGrid (envelope propagation) with Kerr only, no plasma
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange,
               raman=false, plasma=false, kerr=true, shotnoise=false,
               energy=1e-6, τfwhm=30e-15, envelope=true)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        @assert grid isa Amalthea.Grid.EnvGrid "Expected EnvGrid for envelope=true"

        t0 = 0.0
        dt = 0.01

        @testset "Single-step equivalence (EnvGrid Kerr, ~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel_step < 1e-13
        end

        @testset "Full-solve equivalence (EnvGrid Kerr, ~1e-6)" begin
            # Fixed step size (max_dt=min_dt=dt): the adaptive PI controller's
            # embedded error estimate is a near-total cancellation (b5-b4=0),
            # so tiny FP-summation-order differences between Julia and Rust
            # can amplify into different dt choices, sending the two adaptive
            # integrators down different step paths that land at different z
            # (see docs/dev/native-port/PORT_LOG.md). Forcing an identical step-size
            # sequence isolates genuine multi-step state-accumulation error.
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)

            solve(s_jl, flength)
            solve(s_ru, flength)

            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("EnvGrid Kerr full-solve rel: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end

@testitem "Native-Rust Phase 2b (plasma, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        radius = 125e-6
        flength = 0.10
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (150e-9, 4e-6)
        trange = 500e-15

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange,
               raman=false, plasma=:PPT, kerr=true, shotnoise=false,
               energy=2e-6, τfwhm=30e-15)

        # The native plasma path needs a Rust-backed ionization-rate handle,
        # which is only wired up if LUNA_USE_RUST_IONISATION=1 is set BEFORE
        # the ionization LUT is built inside prop_capillary_args (not just
        # around the later RustNativeStepper construction below) — so the
        # whole setup is wrapped in one withenv rather than relying on an
        # ambient CI-wide env var (which would conflict with
        # test_ionisation_rust.jl's check that the default is off).
        Eω, grid, linop, transform, FT, output = withenv("LUNA_USE_RUST_IONISATION" => "1") do
            with_logger(NullLogger()) do
                Interface.prop_capillary_args(args...; kw...)
            end
        end

        let
            @assert grid isa Amalthea.Grid.RealGrid "plasma test uses RealGrid"

            t0 = 0.0
            dt = 0.01

            @testset "Single-step equivalence (RealGrid + plasma, ~1e-13)" begin
                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
                s_ru = withenv("LUNA_USE_RUST_NATIVE" => "1",
                               "LUNA_USE_RUST_IONISATION" => "1") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
                end

                step!(s_jl)
                step!(s_ru)

                rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("Plasma single-step rel: ", rel_step)
                @test rel_step < 1e-13
            end

            @testset "Full-solve equivalence (RealGrid + plasma, ~1e-6)" begin
                # Fixed step size (max_dt=min_dt=dt): see comment in the
                # EnvGrid Kerr full-solve testset above for why the adaptive
                # PI controller's near-cancellation error estimate makes raw
                # adaptive dt agreement an unreliable test of physics
                # correctness. Forcing an identical step-size sequence isolates
                # genuine multi-step state-accumulation error.
                s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                s_ru = withenv("LUNA_USE_RUST_NATIVE" => "1",
                               "LUNA_USE_RUST_IONISATION" => "1") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                end

                solve(s_jl, flength)
                solve(s_ru, flength)

                rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
                println("Plasma full-solve rel: ", rel_solve)
                @test rel_solve < 1e-6
            end

            @testset "Full-solve triangulation at a plasma-matters energy (density-factor regression guard)" begin
                # docs/dev/BACKLOG.md Phase I item 3 postmortem (2026-07-08): the
                # equivalence test above runs at energy=2e-6 J, weak enough
                # that PPT's ionization rate stays near zero throughout —
                # exactly the blind spot that let a real bug (native's
                # plasma polarization missing a density factor entirely,
                # silently contributing ~0 instead of ρ·P) go undetected for
                # every native mode-averaged/radial plasma sim since Phase C.
                # Re-run the same geometry at a much stronger energy where
                # ionization is unambiguously non-vacuous, and triangulate:
                # native must match Julia's plasma-ON result, not its
                # plasma-OFF one.
                kw_strong = (; λ0, λlims, trange, raman=false, plasma=:PPT, kerr=true,
                             shotnoise=false, energy=1.6e-3, τfwhm=30e-15)
                kw_strong_noplasma = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true,
                             shotnoise=false, energy=1.6e-3, τfwhm=30e-15)
                Eω_s, grid_s, linop_s, transform_s, FT_s, output_s = withenv("LUNA_USE_RUST_IONISATION" => "1") do
                    with_logger(NullLogger()) do
                        Interface.prop_capillary_args(args...; kw_strong...)
                    end
                end
                Eω_sn, grid_sn, linop_sn, transform_sn, FT_sn, output_sn = with_logger(NullLogger()) do
                    Interface.prop_capillary_args(args...; kw_strong_noplasma...)
                end
                dt_s = 0.005
                s_jl_p = PreconStepper(transform_s, linop_s, copy(Eω_s), t0, dt_s, rtol=1e-6, atol=1e-10,
                                        max_dt=dt_s, min_dt=dt_s)
                s_jl_np = PreconStepper(transform_sn, linop_sn, copy(Eω_sn), t0, dt_s, rtol=1e-6, atol=1e-10,
                                         max_dt=dt_s, min_dt=dt_s)
                s_ru_p = withenv("LUNA_USE_RUST_NATIVE" => "1", "LUNA_USE_RUST_IONISATION" => "1") do
                    RustNativeStepper(transform_s, linop_s, copy(Eω_s), t0, dt_s, rtol=1e-6, atol=1e-10,
                                       max_dt=dt_s, min_dt=dt_s)
                end
                solve(s_jl_p, flength)
                solve(s_jl_np, flength)
                solve(s_ru_p, flength)

                rel_effect = norm(s_jl_p.yn - s_jl_np.yn) / norm(s_jl_np.yn)
                rel_native_vs_jl = norm(s_ru_p.yn - s_jl_p.yn) / norm(s_jl_p.yn)
                rel_native_vs_noplasma = norm(s_ru_p.yn - s_jl_np.yn) / norm(s_jl_np.yn)
                println("Strong-energy plasma-on vs plasma-off rel (Julia, non-vacuousness): ", rel_effect)
                println("Strong-energy native vs Julia (both plasma-on): ", rel_native_vs_jl)
                println("Strong-energy native vs Julia no-plasma (must NOT match): ", rel_native_vs_noplasma)
                @test rel_effect > 1e-4
                @test rel_native_vs_jl < 1e-10
                @test rel_native_vs_noplasma > 1e-4
            end
        end
    end
end
