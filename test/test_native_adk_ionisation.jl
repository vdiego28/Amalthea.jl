using TestItems

@testitem "Native-Rust Phase I item 3 (ADK ionisation, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase I item 3: `ionisation=:ADK` builds `Ionisation.IonRateADK`
        # (closed-form), not `IonRatePPTAccel` (spline LUT) — the native plasma
        # wiring previously required the latter, so :ADK always fell back.
        #
        # `energy=1.6e-3` (not a weaker value) is deliberate: ADK's rate is
        # exp(-const/E) — at weak fields the rate is zero everywhere (below
        # `ADK_threshold`) and a plasma-vs-no-plasma comparison spuriously
        # passes with *no* ionization physics actually exercised. This energy
        # was chosen (see docs/dev/BACKLOG.md's Phase I item 3 postmortem) to put a
        # measurable (~0.5%) but not chaotic ionization effect on the
        # propagation, which is what actually caught a real bug: a missing
        # density factor in the native plasma polarization accumulation
        # (`apply_plasma_real`/`apply_plasma_radial` in native.rs) that
        # silently affected BOTH the PPT and ADK native paths.
        radius = 125e-6
        flength = 0.10
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (150e-9, 4e-6)
        trange = 500e-15
        energy = 1.6e-3

        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange,
               raman=false, plasma=:ADK, kerr=true, shotnoise=false,
               energy, τfwhm=30e-15)
        kw_noplasma = (; λ0, λlims, trange,
               raman=false, plasma=false, kerr=true, shotnoise=false,
               energy, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end
        Eω2, grid2, linop2, transform_noplasma, FT2, output2 = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw_noplasma...)
        end

        @assert grid isa Amalthea.Grid.RealGrid "plasma test uses RealGrid"

        t0 = 0.0
        dt = 0.005

        @testset "Not silently ineligible: native stepper actually builds" begin
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            @test s_ru isa RustNativeStepper
        end

        @testset "Full-solve triangulation (native vs Julia vs no-plasma)" begin
            s_jl_plasma = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                         max_dt=dt, min_dt=dt)
            s_jl_noplasma = PreconStepper(transform_noplasma, linop2, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10,
                                           max_dt=dt, min_dt=dt)
            s_ru_plasma = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                             max_dt=dt, min_dt=dt)

            solve(s_jl_plasma, flength)
            solve(s_jl_noplasma, flength)
            solve(s_ru_plasma, flength)

            rel_effect = norm(s_jl_plasma.yn - s_jl_noplasma.yn) / norm(s_jl_noplasma.yn)
            rel_native_vs_jl = norm(s_ru_plasma.yn - s_jl_plasma.yn) / norm(s_jl_plasma.yn)
            rel_native_vs_noplasma = norm(s_ru_plasma.yn - s_jl_noplasma.yn) / norm(s_jl_noplasma.yn)
            println("ADK plasma-on vs plasma-off rel (Julia, non-vacuousness): ", rel_effect)
            println("ADK native vs Julia (both plasma-on): ", rel_native_vs_jl)
            println("ADK native vs Julia no-plasma (must NOT match — regression guard): ",
                     rel_native_vs_noplasma)

            # Real ionization effect present in Julia (not vacuous).
            @test rel_effect > 1e-4
            # Native matches Julia's plasma-on result, not its plasma-off
            # result — this is exactly the check that catches "native plasma
            # silently contributes nothing" (the density-factor bug found
            # 2026-07-08), which a plasma-vs-no-plasma check on native alone,
            # or an equivalence check at too weak an energy, would both miss.
            @test rel_native_vs_jl < 1e-10
            @test rel_native_vs_noplasma > 1e-4
        end
    end
end
