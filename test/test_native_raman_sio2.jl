using TestItems

@testitem "Native-Rust Phase I item 2 (ramanmodel=:SiO2, mode-avg EnvGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Luna
    using Luna.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBLUNA_RUST_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # docs/dev/BACKLOG.md Phase I item 2: `ramanmodel=:SiO2` builds a bare
        # `RamanRespIntermediateBroadening` (Hollenbeck & Cantrell
        # Gaussian-damped multi-line model, Raman.jl:78-113) — no finite sum
        # of single-damped-oscillators reproduces its `exp(-Γᵢ²t²/4)` envelope,
        # so it can't reuse the resident ADE solver wired for
        # `CombinedRamanResponse`. Ported instead as a second, resident
        # FFT-convolution kernel (`native_set_raman_fft_params`/
        # `rhs_mode_avg_env`'s new Step 3c in native.rs), mirroring the
        # generic Julia path in `(R::RamanPolar)(out, Et, ρ)`
        # (Nonlinear.jl:395-417) exactly. Only reachable via `prop_gnlse`
        # (mode-averaged EnvGrid, `RamanPolarEnv`) — `:SiO2` is not wired for
        # `prop_capillary`.
        γ = 0.1
        β2 = -1e-26
        N = 4.0
        τ0 = 280e-15
        τfwhm = (2*log(1 + sqrt(2)))*τ0
        fr = 0.18
        P0 = N^2*abs(β2)/((1-fr)*γ*τ0^2)
        flength = pi*τ0^2/abs(β2)
        βs =  [0.0, 0.0, β2]
        λ0 = 835e-9
        λlims = [450e-9, 8000e-9]
        trange = 4e-12

        # shock/shotnoise deliberately left at `prop_gnlse`'s defaults
        # (both `true`) — Phase I item 1's postmortem (docs/dev/BACKLOG.md) is that
        # the EnvGrid noise gap stayed invisible until the *default* gnlse
        # workload was tested, not a hand-picked easy config. This is the
        # config a real `prop_gnlse(...; ramanmodel=:SiO2)` call actually
        # hits.
        kw = (; λ0, τfwhm, power=P0, pulseshape=:sech, λlims, trange,
               raman=true, ramanmodel=:SiO2, fr)
        kw_noraman = (; λ0, τfwhm, power=P0, pulseshape=:sech, λlims, trange,
               raman=false)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_gnlse_args(γ, flength, βs; kw...)
        end
        Eω2, grid2, linop2, transform_nr, FT2, output2 = with_logger(NullLogger()) do
            Interface.prop_gnlse_args(γ, flength, βs; kw_noraman...)
        end

        @assert grid isa Luna.Grid.EnvGrid "prop_gnlse uses EnvGrid"

        t0 = 0.0
        dt = flength / 2000

        @testset "Not silently ineligible: native stepper actually builds" begin
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            @test s_ru isa RustNativeStepper
        end

        @testset "Single-step equivalence (~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)

            step!(s_jl)
            step!(s_ru)

            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println(":SiO2 Raman single-step rel: ", rel_step)
            @test rel_step < 1e-13
        end

        @testset "Full-solve triangulation (native vs Julia vs no-raman)" begin
            s_jl_r = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                    max_dt=dt, min_dt=dt)
            s_jl_nr = PreconStepper(transform_nr, linop2, copy(Eω2), t0, dt, rtol=1e-6, atol=1e-10,
                                     max_dt=dt, min_dt=dt)
            s_ru_r = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                        max_dt=dt, min_dt=dt)

            solve(s_jl_r, flength)
            solve(s_jl_nr, flength)
            solve(s_ru_r, flength)

            rel_effect = norm(s_jl_r.yn - s_jl_nr.yn) / norm(s_jl_nr.yn)
            rel_native_vs_jl = norm(s_ru_r.yn - s_jl_r.yn) / norm(s_jl_r.yn)
            rel_native_vs_noraman = norm(s_ru_r.yn - s_jl_nr.yn) / norm(s_jl_nr.yn)
            println(":SiO2 Raman-on vs raman-off rel (Julia, non-vacuousness): ", rel_effect)
            println(":SiO2 native vs Julia (both raman-on): ", rel_native_vs_jl)
            println(":SiO2 native vs Julia no-raman (must NOT match): ", rel_native_vs_noraman)

            # Real Raman effect present in Julia (not vacuous — a soliton
            # over many dispersion lengths self-shifts substantially under
            # SiO2 Raman gain).
            @test rel_effect > 1e-2
            # Native matches Julia's raman-on result, not its raman-off
            # result.
            @test rel_native_vs_jl < 1e-10
            @test rel_native_vs_noraman > 1e-2
        end
    end
end
