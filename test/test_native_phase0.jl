using TestItems

@testitem "Native-Rust Phase 0 (callback-free stepper)" tags=[:rust] begin
    import Test: @test, @test_skip
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!
    import FFTW

    use_native = get(ENV, "AMALTHEA_USE_RUST_NATIVE", "0") == "1"
    
    # We only run if we can load the library
    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # 1. No-op RHS reproduction
        n = 128
        linop = ComplexF64[ -0.1*i + 0.5im*i for i in 1:n ]
        y0 = ComplexF64[ sin(0.1*i) + 1im*cos(0.1*i) for i in 1:n ]
        
        # A true no-op RHS: f!(out, y, t) = fill!(out, 0)
        function f_noop!(out, y, t)
            fill!(out, 0)
        end
        
        t0 = 0.0
        dt = 1e-3
        
        s_jl = PreconStepper(f_noop!, linop, copy(y0), t0, dt, rtol=1e-6, atol=1e-10)
        s_ru = RustNativeStepper(linop, copy(y0), t0, dt, rtol=1e-6, atol=1e-10)
        
        # Take a few steps and compare
        for step in 1:5
            ok_jl = step!(s_jl)
            ok_ru = step!(s_ru)
            
            @test ok_jl == ok_ru
            @test s_jl.tn ≈ s_ru.tn
            @test s_jl.dtn ≈ s_ru.dtn
            @test s_jl.err ≈ s_ru.err
            @test s_jl.errlast ≈ s_ru.errlast
            
            # The solutions should match exactly or very closely
            @test isapprox(s_jl.yn, s_ru.yn, rtol=1e-13)
        end
        
        # 2. Full-solve equivalence test at the floor tier
        tmax = 5.0
        
        # Reset steppers
        s_jl = PreconStepper(f_noop!, linop, copy(y0), t0, dt, rtol=1e-6, atol=1e-10)
        s_ru = RustNativeStepper(linop, copy(y0), t0, dt, rtol=1e-6, atol=1e-10)
        
        # RK45.solve updates the stepper in-place.
        # We don't use output=true because interpolate() isn't implemented for Phase 0.
        RK45.solve(s_jl, tmax)
        RK45.solve(s_ru, tmax)
        
        # Compare final state
        using LinearAlgebra
        rel = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
        @test rel < 1e-6
    end
end
