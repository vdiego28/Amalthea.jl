using TestItems

@testitem "Native-Rust Phase 1 (mode-avg Kerr, RealGrid)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # ── Test Geometry & Setup ─────────────────────────────────────────────
        radius = 125e-6
        flength = 0.15
        gas = :He
        pressure = 1.0
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12
        
        args = (radius, flength, gas, pressure)
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true, shotnoise=false, energy=1e-6, τfwhm=30e-15)
        
        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end
        
        t0 = 0.0
        dt = 0.01 # 10 mm step size to get error far above machine precision floor
        
        @testset "Single-step equivalence (~1e-13)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10)
            
            ok_jl = step!(s_jl)
            ok_ru = step!(s_ru)
            
            @test ok_jl == ok_ru
            @test s_jl.tn ≈ s_ru.tn
            @test s_jl.dtn ≈ s_ru.dtn rtol=1e-11
            @test s_jl.err ≈ s_ru.err rtol=1e-11
            @test s_jl.errlast ≈ s_ru.errlast rtol=1e-11
            
            rel_step = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            @test rel_step < 1e-13
        end
        
        @testset "Full-solve equivalence (~1e-6)" begin
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
            println("Full solve rel_solve: ", rel_solve)
            @test rel_solve < 1e-6
        end
    end
end
