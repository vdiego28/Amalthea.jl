using TestItems

@testitem "Native-Rust Phase F.3 (multi-point pressure gradient)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm
    import Amalthea.Capillary: gradient, MarcatiliMode, MultiPointGradient
    import Amalthea.PhysData: wlfreq, c
    import Amalthea: Maths

    libpath = RK45._LIBAMALTHEA_RK45
    if !isfile(libpath)
        @test_skip "Rust library not found"
    else
        # ── Test Geometry & Setup ─────────────────────────────────────────────
        # A 3-segment piecewise pressure profile: `pressure = (Z, P)` with
        # length(Z) > 2 triggers Interface.jl's `makemode_s`/`makedensity`
        # general-tuple branch (`Capillary.gradient(gas, Z, P)`), which now
        # (docs/dev/BACKLOG.md Phase F item 3) builds a `MultiPointGradient` instead of
        # `nothing`, so `LinearOps.make_linop`'s Capillary specialization
        # wraps it in the same `ZDepLinopMarcatili` the two-point gradient
        # (Phase 7) uses -- `ensure_linop_at` (Rust) selects the segment
        # containing `z` at every resident call instead of the single
        # closed-form two-point formula.
        radius = 125e-6
        flength = 0.5
        gas = :Ar
        Z = [0.0, 0.15, 0.35, flength]
        P = [0.5, 3.0, 1.5, 5.0]  # non-monotonic on purpose -- exercises >1 segment shape
        λ0 = 800e-9
        λlims = (200e-9, 4e-6)
        trange = 1e-12

        args = (radius, flength, gas, (Z, P))
        kw = (; λ0, λlims, trange, raman=false, plasma=false, kerr=true, shotnoise=false,
                energy=1e-6, τfwhm=30e-15)

        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            Interface.prop_capillary_args(args...; kw...)
        end

        @test linop isa Amalthea.Capillary.ZDepLinopMarcatili
        @test linop.Z == Float64.(Z)
        @test linop.P == Float64.(P)

        t0 = 0.0
        dt = 0.01

        @testset "β1(z) analytic closed form vs BigFloat ground truth" begin
            # Same two-tier discipline as test_native_zdep_linop.jl's Phase 7
            # check: compares against a BigFloat derivative of the exact same
            # formula, not Julia's own (slightly imprecise) `dispersion`. The
            # 4 closed-form constants are z-independent regardless of how
            # many pressure-gradient segments feed dens(z) (BETA1_ANALYTIC.md
            # §2 -- the derivation only assumes εco(ω;z)-1 is separable into
            # γ(λ(ω))·dens(z), true for any pfun).
            coren, _ = gradient(gas, Z, P)
            mode = MarcatiliMode(radius, coren)
            @test mode.coren.pfun isa MultiPointGradient
            ω0 = wlfreq(λ0)

            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      flength=flength)
            dens_r = Ref{Float64}(0.0)
            beta1_r = Ref{Float64}(0.0)
            for z in (0.0, 0.05, 0.15, 0.2, 0.35, 0.4, 0.49, 0.5)
                ccall((:native_debug_beta1_at, RK45._LIBAMALTHEA_RK45), Cint,
                      (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
                      s_ru._handle.ptr, z, dens_r, beta1_r)
                β_of_ω(ω) = ω/c*real(Amalthea.Capillary.neff(mode, ω; z=z))
                β1_bigfloat = Float64(Maths.derivative(β_of_ω, BigFloat(ω0), 1))
                rel = abs(beta1_r[] - β1_bigfloat) / abs(β1_bigfloat)
                @test rel < 1e-9
            end
        end

        @testset "Full-solve equivalence (fixed step size)" begin
            s_jl = PreconStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            s_ru = RustNativeStepper(transform, linop, copy(Eω), t0, dt, rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt, flength=flength)

            solve(s_jl, flength)
            solve(s_ru, flength)

            # Self-validating: the multi-point profile actually changes the
            # Julia oracle's result relative to a constant-pressure run
            # (same discipline as test_native_zdep_linop.jl's Phase 7 check).
            args_const = (radius, flength, gas, P[end])
            Eω_c, grid_c, linop_c, transform_c, FT_c, _ = with_logger(NullLogger()) do
                Interface.prop_capillary_args(args_const...; kw...)
            end
            s_const = PreconStepper(transform_c, linop_c, copy(Eω_c), t0, dt, rtol=1e-6,
                                     atol=1e-10, max_dt=dt, min_dt=dt)
            solve(s_const, flength)
            rel_gradient_effect = norm(s_jl.yn - s_const.yn) / norm(s_jl.yn)
            println("Phase F.3 multi-point-gradient effect (vs constant fill): ", rel_gradient_effect)
            @test rel_gradient_effect > 1e-6

            # Same ~1e-4 systematic-β1-offset tier as Phase 7 (test_native_zdep_linop.jl)
            # -- unrelated to the number of pressure-gradient segments.
            rel_solve = norm(s_ru.yn - s_jl.yn) / norm(s_jl.yn)
            println("Phase F.3 full-solve rel_solve: ", rel_solve)
            @test rel_solve < 1e-3
        end
    end
end
